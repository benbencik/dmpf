use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dmpf::{CorrectionWord, LvlDpfDmpfDb, Node, PrimeField64x2};
use rand::{thread_rng, RngCore};

const K: usize = 8;
const INPUT_BITS: usize = 128;

// bypass gen, build a db with random values
fn random_db<R: RngCore>(
    input_bits: usize,
    num_points: usize,
    num_messages: usize,
    rng: &mut R,
) -> LvlDpfDmpfDb<PrimeField64x2> {
    // N*K random PRG seeds
    let init_seeds: Vec<Node> = (0..num_points * num_messages)
        .map(|_| {
            let lo = rng.next_u64() as u128;
            let hi = rng.next_u64() as u128;
            Node::from(lo | (hi << 64))
        })
        .collect();

    // N*K bits, this is either 0 or 1 depending on the server
    // for 1 we do slightly more work, bench that
    let init_correction_bits: Vec<bool> = vec![true; num_points * num_messages];

    // N*K*128 random correction words
    let cws: Vec<Vec<CorrectionWord>> = (0..num_points)
        .map(|_| {
            (0..input_bits * num_messages)
                .map(|_| {
                    let lo = rng.next_u64() as u128;
                    let hi = rng.next_u64() as u128;
                    CorrectionWord::new(
                        Node::from(lo | (hi << 64)),
                        rng.next_u32() & 1 == 1,
                        rng.next_u32() & 1 == 1,
                    )
                })
                .collect()
        })
        .collect();

    // N*K random last correction words
    let last_cw: Vec<PrimeField64x2> = (0..num_points * num_messages)
        .map(|_| PrimeField64x2::from(Node::from(rng.next_u64() as u128)))
        .collect();

    LvlDpfDmpfDb::new_from_values(
        input_bits,
        num_points,
        num_messages,
        init_seeds,
        init_correction_bits,
        cws,
        last_cw,
    )
}

fn bench_eval_dmpf_seq(c: &mut Criterion) {
    let mut rng = thread_rng();
    let mut group = c.benchmark_group(format!("lvl_dmpf_eval_seq/K/{}", K));

    for &logn in [10, 12, 14, 16, 18].iter() {
        let n = 1 << logn;

        let db = random_db(INPUT_BITS, K, n, &mut rng);
        let input = ((rng.next_u64() as u128) << 64) | rng.next_u64() as u128;

        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("seq/logN", logn), &n, |b, _| {
            b.iter(|| db.eval_dmpf_seq(&input));
        });
    }
    group.finish();
}

fn bench_eval_dmpf_par(c: &mut Criterion) {
    let mut rng = thread_rng();
    let mut group = c.benchmark_group(format!("lvl_dmpf_eval_par/K/{}", K));

    for &logn in [10, 12, 14, 16, 18].iter() {
        let n = 1 << logn;

        let db = random_db(INPUT_BITS, K, n, &mut rng);
        let input = ((rng.next_u64() as u128) << 64) | rng.next_u64() as u128;

        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("par/logN", logn), &n, |b, _| {
            b.iter(|| db.eval_dmpf_par(&input));
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_eval_dmpf_seq
);
criterion_main!(benches);
