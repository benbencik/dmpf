use crate::OmrDmpf;
use crate::OmrDmpfKey;
// use super::BITS_OF_SECURITY;
use crate::prg::double_prg;
use crate::prg::prg_eval_select;
use crate::prg::prg_eval_select_vectorized;
// use crate::prg::double_prg_many;
// use crate::prg::many_prg;
use crate::prg::DOUBLE_PRG_CHILDREN;
// use crate::utils::BitVec;
use crate::utils::Node;
use crate::DmpfSession;
use crate::DpfOutput;
use core::simd::u64x4;
use rayon::prelude::*;

// TODO: tune this based on the seed size and L2 cache
// used in eval_dpf to chunk the AES calls efficiently
const PRG_CHUNK: usize = 1 << 8;

// prg_eval_select requires an even-length slice, assume PRG_CHUNK is even
const _: () = assert!(PRG_CHUNK % 2 == 0, "PRG_CHUNK must be even");

const MASK_ALL_128: u128 = u128::MAX;
const MASK_ALL_64: u64 = u64::MAX;
const MASK_TWO_BITS_64: u64 = !3u64;
const MASK_TWO_BITS_64X4: u64x4 = u64x4::from_array([!3u64, u64::MAX, !3u64, u64::MAX]);

#[derive(Clone, Copy)]
pub struct CorrectionWord {
    node: Node,
}
pub struct LvlDpfDmpf;
impl LvlDpfDmpf {
    pub fn new() -> Self {
        Self
    }
}
impl<Output: DpfOutput> OmrDmpf<Output> for LvlDpfDmpf {
    type Key = LvlDpfDmpfKey<Output>;
    fn try_gen<R: rand::prelude::CryptoRng + rand::prelude::RngCore>(
        &self,
        input_length: usize,
        inputs: &[(u128, Output)],
        mut rng: &mut R,
    ) -> Option<(Self::Key, Self::Key)> {
        let mut first_keys = Vec::with_capacity(inputs.len());
        let mut second_keys = Vec::with_capacity(inputs.len());
        inputs.iter().for_each(|(k, v)| {
            let init_seeds = (Node::random(&mut rng), Node::random(&mut rng));
            let (f, s) = DpfKey::gen(&init_seeds, k, input_length, v);
            first_keys.push(f);
            second_keys.push(s);
        });
        Some((
            LvlDpfDmpfKey {
                dpf_keys: first_keys,
            },
            LvlDpfDmpfKey {
                dpf_keys: second_keys,
            },
        ))
    }
}

// server database of all message shares
pub struct LvlDpfDmpfDb<Output> {
    input_bits: usize,               // tree depth
    num_points: usize,               // K points of the DMPF
    num_messages: usize,             // N max number of messages
    messages_count: usize,           // current number of messages <= N
    init_seeds: Vec<Node>,           // len K*N, index [k*N + n]
    init_correction_bits: Vec<bool>, // len K*N, index [k*N + n] (this is also server index)
    cws: Vec<Vec<CorrectionWord>>,   // len K, each len input_bits*N, index [k][level*N + n]
    last_cw: Vec<Output>,            // len K*N, index [k*N + n]
}

impl<Output: DpfOutput> LvlDpfDmpfDb<Output> {
    pub fn new(input_bits: usize, num_points: usize, num_messages: usize) -> Self {
        Self {
            input_bits,
            num_points,
            num_messages,
            messages_count: 0,
            init_seeds: vec![Node::default(); num_points * num_messages],
            init_correction_bits: vec![false; num_points * num_messages],
            cws: vec![
                vec![
                    CorrectionWord::new(Node::default(), false, false);
                    input_bits * num_messages
                ];
                num_points
            ],
            last_cw: vec![Output::default(); num_points * num_messages],
        }
    }

    // builds a db directly from values (used for benchmarks)
    pub fn new_from_values(
        input_bits: usize,
        num_points: usize,
        num_messages: usize,
        init_seeds: Vec<Node>,
        init_correction_bits: Vec<bool>,
        cws: Vec<Vec<CorrectionWord>>,
        last_cw: Vec<Output>,
    ) -> Self {
        assert_eq!(init_seeds.len(), num_points * num_messages);
        assert_eq!(init_correction_bits.len(), num_points * num_messages);
        assert_eq!(cws.len(), num_points);
        assert!(cws.iter().all(|i| i.len() == input_bits * num_messages));
        assert_eq!(last_cw.len(), num_points * num_messages);

        Self {
            input_bits,
            num_points,
            num_messages,
            messages_count: num_messages,
            init_seeds,
            init_correction_bits,
            cws,
            last_cw,
        }
    }

    pub fn insert(&mut self, key: LvlDpfDmpfKey<Output>) {
        assert!(self.messages_count < self.num_messages, "db full capacity");
        for k in 0..self.num_points {
            let dpf_key = &key.dpf_keys[k];
            self.init_seeds[k * self.num_messages + self.messages_count] = dpf_key.init_seed;
            self.init_correction_bits[k * self.num_messages + self.messages_count] =
                dpf_key.init_correction_bit;
            self.last_cw[k * self.num_messages + self.messages_count] = dpf_key.last_cw;
            for level in 0..self.input_bits {
                self.cws[k][level * self.num_messages + self.messages_count] = dpf_key.cws[level];
            }
        }
        self.messages_count += 1;
    }

    // one single-point DPF, evaluated for all message shares
    fn eval_dpf(
        &self,
        k: usize,
        input: &u128,
        offset_n: usize,
        cur_seeds: &mut [Node],
        cur_correction_bits: &mut [bool],
        output: &mut [Output],
    ) {
        let n = output.len();
        let start = k * self.num_messages + offset_n;

        cur_seeds[..n].copy_from_slice(&self.init_seeds[start..start + n]);
        cur_correction_bits[..n].copy_from_slice(&self.init_correction_bits[start..start + n]);
        let cur_cw = &self.cws[k]; // correction words for this point

        let mut prg_out = [Node::default(); PRG_CHUNK];
        for level in 0..self.input_bits {
            let path_bit = get_bit(*input, level);
            let ctr = path_bit as u8;

            // get slice of correction words for this level for faster indexing
            let level_cw = &cur_cw
                [level * self.num_messages + offset_n..level * self.num_messages + offset_n + n];

            for chunk_start in (0..n).step_by(PRG_CHUNK) {
                let chunk_end = (chunk_start + PRG_CHUNK).min(n);
                let chunk_len = chunk_end - chunk_start;

                // prg_eval_select requires an even-length slice, for final simd xor
                let chunk_len_padded = chunk_len + (chunk_len % 2);
                let chunk_end_padded = chunk_start + chunk_len_padded;

                // aesenc calls are batched the chunking should be set such that the calls
                // fit inside cache (ideally L2)
                prg_eval_select(
                    &mut cur_seeds[chunk_start..chunk_end_padded],
                    ctr,
                    &mut prg_out[..chunk_len_padded],
                );
                for (idx, i) in (chunk_start..chunk_end).enumerate() {
                    let t = cur_correction_bits[i];

                    let mut new_s = prg_out[idx];
                    let mut new_t = new_s.pop_first_bit();

                    let mut cw = level_cw[i].node;
                    let (left_bit, right_bit) = cw.pop_first_two_bits();

                    // branchless correction only applied if t is 1
                    let mask_correction_bits: u128 = (t as u128) * MASK_ALL_128;
                    new_s ^= &Node::from(u128::from(cw) & mask_correction_bits);
                    let selected_bit = (left_bit & !path_bit) ^ (right_bit & path_bit);
                    new_t ^= selected_bit & t;

                    cur_seeds[i] = new_s;
                    cur_correction_bits[i] = new_t;
                }
            }
        }

        for i in 0..n {
            let mut val = Output::from(cur_seeds[i]); // convert step of the DPF
            if cur_correction_bits[i] {
                val += self.last_cw[start + i];
            }
            if self.init_correction_bits[start + i] {
                val = val.neg();
            }
            output[i] += val; // accumulate the share of the output
        }
    }

    // split to pair of u64 to load into simd register
    #[inline(always)]
    fn nodes_u64(s: &[Node]) -> &[u64] {
        unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u64>(), s.len() * 2) }
    }

    // split to pair of u64 to load into simd register
    #[inline(always)]
    fn nodes_u64_mut(s: &mut [Node]) -> &mut [u64] {
        unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u64>(), s.len() * 2) }
    }

    // split to pair of u64 to load into simd register
    #[inline(always)]
    fn cws_u64(s: &[CorrectionWord]) -> &[u64] {
        unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u64>(), s.len() * 2) }
    }

    // same update as in eval_dpf 
    #[inline(always)]
    fn update_correction_bit(prg_lo: u64, cw_lo: u64, t: bool, path_bit: bool) -> bool {
        let left_bit = cw_lo & 1 == 1;
        let right_bit = (cw_lo >> 1) & 1 == 1;
        let selected_bit = (left_bit & !path_bit) ^ (right_bit & path_bit);
        let new_t = (prg_lo & 1) == 1;
        new_t ^ (selected_bit & t)
    }

    // same as eval_dpf but uses simd u64x4
    fn eval_dpf_vectorized(
        &self,
        k: usize,
        input: &u128,
        offset_n: usize,
        cur_seeds: &mut [Node],
        cur_correction_bits: &mut [bool],
        output: &mut [Output],
    ) {
        let n = output.len();
        let start = k * self.num_messages + offset_n;

        cur_seeds[..n].copy_from_slice(&self.init_seeds[start..start + n]);
        cur_correction_bits[..n].copy_from_slice(&self.init_correction_bits[start..start + n]);
        let cur_cw = &self.cws[k]; // correction words for this point

        // clears the two tag bits in the low u64 of each 128-bit node

        let mut prg_out = [Node::default(); PRG_CHUNK];
        for level in 0..self.input_bits {
            let path_bit = get_bit(*input, level);
            let ctr = path_bit as u8;

            // get slice of correction words for this level for faster indexing
            let level_cw = &cur_cw
                [level * self.num_messages + offset_n..level * self.num_messages + offset_n + n];

            for chunk_start in (0..n).step_by(PRG_CHUNK) {
                let chunk_end = (chunk_start + PRG_CHUNK).min(n);
                let chunk_len = chunk_end - chunk_start;

                // prg_eval_select_vectorized requires an even-length slice, for final simd xor
                let chunk_len_padded = chunk_len + (chunk_len % 2);
                let chunk_end_padded = chunk_start + chunk_len_padded;

                // aesenc calls are batched the chunking should be set such that the calls
                // fit inside cache (ideally L2)
                prg_eval_select_vectorized(
                    &mut cur_seeds[chunk_start..chunk_end_padded],
                    ctr,
                    &mut prg_out[..chunk_len_padded],
                );

                let mut prg = Self::nodes_u64(&prg_out[..chunk_len]).chunks_exact(4); // [lo,hi] per node
                let mut seeds =
                    Self::nodes_u64_mut(&mut cur_seeds[chunk_start..chunk_end]).chunks_exact_mut(4);
                let mut cw = Self::cws_u64(&level_cw[chunk_start..chunk_end]).chunks_exact(4); // [lo,hi] per word
                let mut t = cur_correction_bits[chunk_start..chunk_end].chunks_exact_mut(2);

                for (((seed_pair, prg_pair), cw_pair), t_pair) in
                    (&mut seeds).zip(&mut prg).zip(&mut cw).zip(&mut t)
                {
                    let prg_simd = u64x4::from_slice(prg_pair);
                    let cw_simd = u64x4::from_slice(cw_pair);

                    let mask_t0_64 = (t_pair[0] as u64) * MASK_ALL_64;
                    let mask_t1_64 = (t_pair[1] as u64) * MASK_ALL_64;
                    let mask_correction_bits = u64x4::from_array([mask_t0_64, mask_t0_64, mask_t1_64, mask_t1_64]);

                    let new_seed = (prg_simd ^ (cw_simd & mask_correction_bits)) & MASK_TWO_BITS_64X4;
                    new_seed.copy_to_slice(seed_pair);

                    t_pair[0] = Self::update_correction_bit(prg_pair[0], cw_pair[0], t_pair[0], path_bit);
                    t_pair[1] = Self::update_correction_bit(prg_pair[2], cw_pair[2], t_pair[1], path_bit);
                }

                // handle odd number of chunks, without simd
                // we can also pad but that would require chanding LvlDpfDmpfDb this seems simpler
                if let ([seed_lo, seed_hi], [prg_lo, prg_hi], [cw_lo, cw_hi], [t0]) = (
                    seeds.into_remainder(),
                    prg.remainder(),
                    cw.remainder(),
                    t.into_remainder(),
                ) {
                    let mask_correction_bits = (*t0 as u64) * MASK_ALL_64;
                    *seed_lo = (*prg_lo ^ (*cw_lo & mask_correction_bits)) & MASK_TWO_BITS_64;
                    *seed_hi = prg_hi ^ (cw_hi & mask_correction_bits);
                    *t0 = Self::update_correction_bit(*prg_lo, *cw_lo, *t0, path_bit);
                }
            }
        }

        // TODO: this part is not vectorized, but it should not be a bottleneck
        for i in 0..n {
            let mut val = Output::from(cur_seeds[i]); // convert step of the DPF
            if cur_correction_bits[i] {
                val += self.last_cw[start + i];
            }
            if self.init_correction_bits[start + i] {
                val = val.neg();
            }
            output[i] += val; // accumulate the share of the output
        }
    }

    #[allow(clippy::type_complexity)]
    fn pick_eval_dpf() -> fn(&Self, usize, &u128, usize, &mut [Node], &mut [bool], &mut [Output]) {
        // true if CPU has 256-bit SIMD registers (AVX2), needed for eval_dpf_vectorized
        #[cfg(target_arch = "x86_64")]
        let has_u64x4_simd = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let has_u64x4_simd = false;

        if has_u64x4_simd {
            Self::eval_dpf_vectorized
        } else {
            Self::eval_dpf
        }
    }

    // outer loop, sequential: runs K single point DPFs
    pub fn eval_dmpf_seq(&self, input: &u128) -> Vec<Output> {
        // prg_eval_select is doing unsafe pointer casts
        // the Node and Block types must have the same size
        assert!(std::mem::size_of::<Node>() == std::mem::size_of::<aes::Block>());

        let n = self.messages_count;

        // reuse across all K runs
        // +1 slot so the last PRG chunk can pad to even length
        let mut cur_seeds = vec![Node::default(); n + (n % 2)];
        let mut cur_correction_bits = vec![false; n];

        let mut output = vec![Output::default(); n];
        let eval_dpf = Self::pick_eval_dpf();
        for k in 0..self.num_points {
            eval_dpf(
                self,
                k,
                input,
                0,
                &mut cur_seeds,
                &mut cur_correction_bits,
                &mut output,
            );
        }
        output
    }

    // outer loop, parallel over number of messages
    // splits message shares into chunks depending on number of threads
    pub fn eval_dmpf_par(&self, input: &u128) -> Vec<Output>
    where
        Output: Send + Sync,
    {
        let n = self.messages_count;
        let mut output = vec![Output::default(); n];
        let num_threads = rayon::current_num_threads();
        // TODO: make sure that PRG_CHUNK nicely divides the chunk size
        let chunk_size = n.div_ceil(num_threads).max(1);
        let eval_dpf = Self::pick_eval_dpf();

        output
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                // each thread owns a disjoint slice of seeds and correction bits
                let chunk_len = out_chunk.len();
                // +1 slot so the last PRG chunk can pad to even length
                let mut cur_seeds = vec![Node::default(); chunk_len + (chunk_len % 2)];
                let mut cur_correction_bits = vec![false; chunk_len];

                let offset_n = chunk_idx * chunk_size;
                for k in 0..self.num_points {
                    eval_dpf(
                        self,
                        k,
                        input,
                        offset_n,
                        &mut cur_seeds,
                        &mut cur_correction_bits,
                        out_chunk,
                    );
                }
            });
        output
    }
}

// TODO: fields unused...struct based on the original implementation, might change later
#[allow(unused)]
pub struct LvlDpfDmpfSession {
    cur_seeds: Vec<Node>,
    next_seeds: Vec<Node>,
    cur_signs: Vec<bool>,
    next_signs: Vec<bool>,
}
impl DmpfSession for LvlDpfDmpfSession {
    fn get_session(_: usize, mut input_bits: usize) -> Self {
        if input_bits > 27 {
            input_bits = 1;
        }
        let mut cur_seeds = Vec::with_capacity(1 << input_bits);
        let mut next_seeds = Vec::with_capacity(1 << input_bits);
        let mut cur_signs = Vec::with_capacity(1 << input_bits);
        let mut next_signs = Vec::with_capacity(1 << input_bits);
        unsafe { cur_seeds.set_len(1 << input_bits) };
        unsafe { next_seeds.set_len(1 << input_bits) };
        unsafe { cur_signs.set_len(1 << input_bits) };
        unsafe { next_signs.set_len(1 << input_bits) };
        Self {
            cur_seeds,
            cur_signs,
            next_seeds,
            next_signs,
        }
    }
}

// DMPF key given by the sender
pub struct LvlDpfDmpfKey<Output> {
    dpf_keys: Vec<DpfKey<Output>>,
}
impl<Output: DpfOutput> OmrDmpfKey<Output> for LvlDpfDmpfKey<Output> {
    type Session = LvlDpfDmpfSession;
    fn input_length(&self) -> usize {
        self.dpf_keys[0].cws.len()
    }
    fn eval_with_session(&self, input: &u128, output: &mut Output, _: &mut Self::Session) {
        *output = self
            .dpf_keys
            .iter()
            .map(|k| {
                let mut cur_out = Output::default();
                k.eval(input, &mut cur_out);
                cur_out
            })
            .sum();
    }
    fn point_count(&self) -> usize {
        self.dpf_keys.len()
    }
}

// TODO: input_bits unused...struct based on the original implementation, might change later
#[allow(unused)]
pub struct DpfKey<Output> {
    init_seed: Node,
    init_correction_bit: bool,
    cws: Vec<CorrectionWord>,
    last_cw: Output,
    input_bits: usize,
}
impl CorrectionWord {
    pub fn new(mut node: Node, left_bit: bool, right_bit: bool) -> Self {
        node.push_first_two_bits(left_bit, right_bit);
        Self { node }
    }
}
// pub(crate) fn tree_and_leaf_depth(alpha_len: usize, beta_len: usize) -> (usize, usize) {
//     let max_betas_in_node = BITS_OF_SECURITY / beta_len;
//     let max_leaf_depth = if max_betas_in_node > 0 {
//         usize::ilog2(max_betas_in_node) as usize
//     } else {
//         0
//     };
//     let tree_depth = if max_leaf_depth > alpha_len {
//         0
//     } else {
//         alpha_len - max_leaf_depth
//     };
//     let leaf_depth = alpha_len - tree_depth;
//     (tree_depth, leaf_depth)
// }

// pub(crate) fn convert(node: &Node, bits: usize) -> BitVec {
//     let mut output = BitVec::new(bits);
//     convert_into(node, &mut output.as_mut());
//     output
// }
// pub(crate) fn convert_into(node: &Node, output: &mut [Node]) {
//     let len = output.len();
//     if len > 1 {
//         many_prg(node, 0..len as u16, output);
//     } else {
//         // We don't have to expand node
//         output[0] = *node;
//     }
// }
fn get_bit(v: u128, bit_idx: usize) -> bool {
    (v >> (127 - bit_idx)) & 1 == 1
}
impl<Output: DpfOutput> DpfKey<Output> {
    pub fn gen(
        init_seeds: &(Node, Node),
        alpha: &u128,
        input_len: usize,
        beta: &Output,
    ) -> (DpfKey<Output>, DpfKey<Output>) {
        // TODO: the init seeds were hardcoded in the original implementation, shouldnt they be random?
        let mut t_0 = false;
        let mut t_1 = true;
        let mut seed_0 = init_seeds.0;
        let mut seed_1 = init_seeds.1;
        let mut cws = Vec::with_capacity(input_len);
        for i in 0..input_len {
            let [mut seeds_l_0, mut seeds_r_0] = double_prg(&seed_0, &DOUBLE_PRG_CHILDREN);
            let [mut seeds_l_1, mut seeds_r_1] = double_prg(&seed_1, &DOUBLE_PRG_CHILDREN);
            let path_bit = get_bit(*alpha, i);
            let (t_l_0, _) = seeds_l_0.pop_first_two_bits();
            let (t_l_1, _) = seeds_l_1.pop_first_two_bits();
            let (t_r_0, _) = seeds_r_0.pop_first_two_bits();
            let (t_r_1, _) = seeds_r_1.pop_first_two_bits();
            let diff_bit_left = !(t_l_0 ^ t_l_1 ^ path_bit);
            let diff_bit_right = t_r_0 ^ t_r_1 ^ path_bit;
            let cw_node = [seeds_l_0 ^ seeds_l_1, seeds_r_0 ^ seeds_r_1][!path_bit as usize];
            seed_0 = [seeds_l_0, seeds_r_0][path_bit as usize];
            seed_1 = [seeds_l_1, seeds_r_1][path_bit as usize];
            if t_0 {
                seed_0 ^= &cw_node;
            }
            if t_1 {
                seed_1 ^= &cw_node;
            }
            (t_0, t_1) = if path_bit {
                (
                    t_r_0 ^ (t_0 & diff_bit_right),
                    t_r_1 ^ (t_1 & diff_bit_right),
                )
            } else {
                (t_l_0 ^ (t_0 & diff_bit_left), t_l_1 ^ (t_1 & diff_bit_left))
            };
            cws.push(CorrectionWord::new(cw_node, diff_bit_left, diff_bit_right));
        }
        let conv_0 = Output::from(seed_0);
        let conv_1 = Output::from(seed_1);
        let mut last_cw: Output = conv_0 - conv_1 - *beta;
        if t_0 {
            last_cw = last_cw.neg();
        }
        let first_key = DpfKey {
            init_seed: init_seeds.0,
            init_correction_bit: false,
            cws: cws.clone(),
            last_cw: last_cw.into(),
            input_bits: input_len,
        };
        let second_key = DpfKey {
            init_seed: init_seeds.1,
            init_correction_bit: true,
            cws,
            last_cw: last_cw.into(),
            input_bits: input_len,
        };
        (first_key, second_key)
    }

    pub fn eval(&self, x: &u128, output: &mut Output) {
        let mut t = self.init_correction_bit;
        let mut s = self.init_seed;
        for (idx, cw) in self.cws.iter().enumerate() {
            let path_bit = get_bit(*x, idx);
            let seeds = double_prg(&s, &DOUBLE_PRG_CHILDREN);
            s = seeds[path_bit as usize];
            let (mut new_t, _) = s.pop_first_two_bits();
            if t {
                let mut cw = cw.node;
                let (left_bit, right_bit) = cw.pop_first_two_bits();
                s ^= &cw;
                new_t ^= (left_bit & !path_bit) ^ (right_bit & path_bit);
            }
            t = new_t;
        }
        *output = Output::from(s);
        if t {
            *output += self.last_cw;
        }
        if self.init_correction_bit {
            *output = output.neg()
        }
    }
}

#[cfg(test)]
mod db_tests {
    use super::*;
    // for now use field from this project migh need to change later
    use crate::OmrDmpf;
    use crate::{field::PrimeField64x2, SmallFieldContainer};
    use rand::{thread_rng, RngCore};

    const INPUT_BITS: usize = 128;
    const K: usize = 5;

    fn rand_point(rng: &mut impl RngCore) -> u128 {
        let lo = rng.next_u64() as u128;
        let hi = rng.next_u64() as u128;
        lo | (hi << 64)
    }

    // sender picks K random (index, value) points per message,
    // and sends keys to both servers
    fn populate_db(
        n: usize,
    ) -> (
        LvlDpfDmpfDb<PrimeField64x2>,
        LvlDpfDmpfDb<PrimeField64x2>,
        Vec<Vec<(u128, PrimeField64x2)>>,
    ) {
        let mut rng = thread_rng();
        let dmpf = LvlDpfDmpf::new();

        let mut db0 = LvlDpfDmpfDb::new(INPUT_BITS, K, n);
        let mut db1 = LvlDpfDmpfDb::new(INPUT_BITS, K, n);
        let mut messages = Vec::with_capacity(n);

        for _ in 0..n {
            let points: Vec<(u128, PrimeField64x2)> = (0..K)
                .map(|_| (rand_point(&mut rng), PrimeField64x2::random(&mut rng)))
                .collect();

            let (key0, key1) = dmpf
                .try_gen(INPUT_BITS, &points, &mut rng)
                .expect("try_gen failed");
            db0.insert(key0);
            db1.insert(key1);
            messages.push(points);
        }
        (db0, db1, messages)
    }

    #[test]
    fn test_inserted_points() {
        for n in [10, 15, 20] {
            let (db0, db1, messages) = populate_db(n);
            for (i, points) in messages.iter().enumerate() {
                for &(idx, expected) in points {
                    let out0 = db0.eval_dmpf_seq(&idx);
                    let out1 = db1.eval_dmpf_seq(&idx);
                    let out0_par = db0.eval_dmpf_par(&idx);
                    let out1_par = db1.eval_dmpf_par(&idx);
                    assert_eq!(out0[i] + out1[i], expected);
                    assert_eq!(out0_par[i] + out1_par[i], expected);
                }
            }
        }
    }

    #[test]
    fn test_not_inseted_pointsc() {
        let n = 25;
        let (db0, db1, messages) = populate_db(n);

        let mut rng = thread_rng();
        // pick 10 random points that are not in any message
        for _ in 0..10 {
            let mut not_message: u128 = 0;
            while not_message == 0 {
                let rand = rand_point(&mut rng);
                let is_used = messages
                    .iter()
                    .any(|points| points.iter().any(|&(idx, _)| idx == rand));
                if !is_used {
                    not_message = rand;
                }
            }

            let out0 = db0.eval_dmpf_seq(&not_message);
            let out1 = db1.eval_dmpf_seq(&not_message);
            let out0_par = db0.eval_dmpf_par(&not_message);
            let out1_par = db1.eval_dmpf_par(&not_message);
            for i in 0..n {
                assert_eq!(out0[i] + out1[i], PrimeField64x2::zero());
                assert_eq!(out0_par[i] + out1_par[i], PrimeField64x2::zero());
            }
        }
    }
}
