//! Recover the public tweak of a split-key vanity address from the base scan
//! key alone: baby-step giant-step over `t < 2^max_bits`.
//!
//! For each of the six `(e, s)` variants the query is `Q = s·λ^{-e}·B − D`
//! and we need `t` with `Q = t·G`. Baby steps tabulate `x(j·G)` for
//! `j < m = 2^K`; giant steps walk `Q − i·m·G` with the same batched-inversion
//! machinery as the key search (generator `−m·G`, base `Q`) and look every
//! visited x up in the table: `x(Q − i·m·G) = x(j·G)` means `t = i·m ± j`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use k256::elliptic_curve::Group;
use k256::{ProjectivePoint, Scalar};

use crate::field::Fe;
use crate::search::{Counter, Table, Visit, Walk, affine_xy};
use crate::tweak::{Tweak, VARIANTS, lambda_pow};

const MIN_BABY_BITS: u32 = 4;
const MAX_BABY_BITS: u32 = 28;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// `M`: search `t < 2^M`; `baby.bits <= M <= 63`.
    pub max_bits: u32,
    pub threads: usize,
    /// Half-batch size of the giant-step walk.
    pub half: usize,
}

/// Giant-step progress for one variant.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub variant: usize,
    pub endo: u8,
    pub negate: bool,
    /// Giant steps done / total over all six variants.
    pub done: u64,
    pub total: u64,
}

/// `x(j·G)` for `0 < j < 2^bits`, keyed by the top 64 bits of x.
pub struct BabyTable {
    bits: u32,
    /// Sorted top limbs and the matching `j`.
    keys: Vec<u64>,
    index: Vec<u32>,
    /// `buckets[b]..buckets[b+1]` is the key range with top bits `b`.
    buckets: Vec<u32>,
    bucket_shift: u32,
    /// Bitmap over the low `bits + 4` bits of the key: cheap negative answers
    /// for the 15 in 16 lookups that miss (8 MB at K = 22).
    filter: Vec<u64>,
    filter_mask: u64,
}

impl BabyTable {
    /// Builds the table with the batched walk from `(H+1)·G`.
    pub fn build(bits: u32, half: usize) -> Result<BabyTable, String> {
        if !(MIN_BABY_BITS..=MAX_BABY_BITS).contains(&bits) {
            return Err(format!(
                "--baby-bits must be between {MIN_BABY_BITS} and {MAX_BABY_BITS}"
            ));
        }
        let m = 1u64 << bits;
        let table = Table::new(half, ProjectivePoint::GENERATOR);
        let mut k0 = table.half as u64 + 1;
        let mut walk = Walk::new(&table, Scalar::from(k0), ProjectivePoint::IDENTITY);
        let mut entries: Vec<(u64, u32)> = Vec::with_capacity(m as usize);
        while (k0 - table.half as u64) < m {
            let visited = walk.batch(|offset: i64, x: &Fe| {
                let j = (k0 as i64 + offset) as u64;
                if j < m {
                    entries.push((x.top_limb(), j as u32));
                }
            });
            if !visited {
                slow_batch(&table, &ProjectivePoint::IDENTITY, k0, |offset, point| {
                    let j = (k0 as i64 + offset) as u64;
                    if j < m && !bool::from(point.is_identity()) {
                        entries.push((affine_xy(&point).0.top_limb(), j as u32));
                    }
                });
            }
            k0 += table.step();
        }
        entries.sort_unstable();
        let bucket_bits = bits.saturating_sub(2).max(1);
        let bucket_shift = 64 - bucket_bits;
        let mut buckets = vec![0u32; (1usize << bucket_bits) + 1];
        for (key, _) in &entries {
            buckets[(key >> bucket_shift) as usize + 1] += 1;
        }
        for b in 1..buckets.len() {
            buckets[b] += buckets[b - 1];
        }
        let filter_bits = bits + 4;
        let filter_mask = (1u64 << filter_bits) - 1;
        let mut filter = vec![0u64; 1usize << (filter_bits - 6)];
        for (key, _) in &entries {
            let bit = key & filter_mask;
            filter[(bit >> 6) as usize] |= 1 << (bit & 63);
        }
        let (keys, index) = entries.into_iter().unzip();
        Ok(BabyTable {
            bits,
            keys,
            index,
            buckets,
            bucket_shift,
            filter,
            filter_mask,
        })
    }

    /// `m = 2^K`.
    fn size(&self) -> u64 {
        1 << self.bits
    }

    /// Pushes every `j` with `x(j·G)` sharing the top 64 bits of `x`.
    fn matches(&self, x: &Fe, out: &mut Vec<u32>) {
        let key = x.top_limb();
        if self.filter_hit(key) {
            self.scan_bucket(key, out);
        }
    }

    /// Bitmap filter on a top limb: `false` means no entry has this key.
    #[inline(always)]
    fn filter_hit(&self, key: u64) -> bool {
        let bit = key & self.filter_mask;
        (self.filter[(bit >> 6) as usize] >> (bit & 63)) & 1 != 0
    }

    /// Bucket scan behind the bitmap filter (one lookup in 16 gets here).
    #[inline(never)]
    fn scan_bucket(&self, key: u64, out: &mut Vec<u32>) {
        let b = (key >> self.bucket_shift) as usize;
        let (lo, hi) = (self.buckets[b] as usize, self.buckets[b + 1] as usize);
        for (k, j) in self.keys[lo..hi].iter().zip(&self.index[lo..hi]) {
            if *k == key {
                out.push(*j);
            }
        }
    }
}

/// Recomputes one skipped batch with k256: `visit(offset, base + (k0 + offset)·P)`.
fn slow_batch<F: FnMut(i64, ProjectivePoint)>(
    table: &Table,
    base: &ProjectivePoint,
    k0: u64,
    mut visit: F,
) {
    let h = table.half as i64;
    for offset in -h..=h {
        let k = Scalar::from((k0 as i64 + offset) as u64);
        visit(offset, *base + table.generator * k);
    }
}

/// `Q = s·λ^{-e}·B − D`, the point that must equal `t·G`.
fn query(
    base: &ProjectivePoint,
    target: &ProjectivePoint,
    endo: u8,
    negate: bool,
) -> ProjectivePoint {
    let signed = if negate { -*target } else { *target };
    signed * lambda_pow(3 - endo % 3) - base
}

/// Giant-step index ranges searched in turn: `[0, 2^8)`, `[2^8, 2^16)`, … so
/// a small `t` is found quickly whatever its `(e, s)` variant. The total work
/// is unchanged (the levels partition the range).
const LEVEL_BITS: u32 = 8;

fn levels(total: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut lo = 0u64;
    let mut hi = 1u64 << LEVEL_BITS;
    while lo < total {
        out.push((lo, hi.min(total)));
        lo = hi;
        hi = hi.saturating_mul(1 << LEVEL_BITS);
    }
    out
}

/// Finds the tweak with `tweak.apply_point(base) == target`, trying the six
/// `(e, s)` variants level by level. `None` when no `t < 2^max_bits` works.
pub fn recover(
    base: &ProjectivePoint,
    target: &ProjectivePoint,
    params: &Params,
    baby: &BabyTable,
    progress: &mut dyn FnMut(Progress),
) -> Option<Tweak> {
    let queries = VARIANTS.map(|(endo, negate)| query(base, target, endo, negate));
    for (q, (endo, negate)) in queries.iter().zip(VARIANTS) {
        if bool::from(q.is_identity()) {
            return Some(Tweak { t: 0, endo, negate });
        }
    }
    let per_variant = 1u64 << (params.max_bits - baby.bits);
    let total = per_variant * VARIANTS.len() as u64;
    let giant = Giant::new(baby, params);
    let mut completed = 0u64;
    for (lo, hi) in levels(per_variant) {
        for (variant, (q, (endo, negate))) in queries.iter().zip(VARIANTS).enumerate() {
            let found = giant.steps(q, lo, hi, &mut |done| {
                progress(Progress {
                    variant,
                    endo,
                    negate,
                    done: completed + done,
                    total,
                })
            });
            completed += hi - lo;
            if let Some(t) = found {
                return Some(Tweak { t, endo, negate });
            }
        }
    }
    None
}

/// The giant-step walk: table for generator `−m·G`, baby table, bounds.
struct Giant<'a> {
    table: Table,
    baby: &'a BabyTable,
    threads: usize,
    /// `2^max_bits`: offsets at or above it are not reported.
    max: u64,
}

impl<'a> Giant<'a> {
    fn new(baby: &'a BabyTable, params: &Params) -> Giant<'a> {
        let m = baby.size();
        Giant {
            table: Table::new(params.half, -(ProjectivePoint::GENERATOR * Scalar::from(m))),
            baby,
            threads: params.threads.max(1),
            max: 1u64 << params.max_bits,
        }
    }

    /// Multi-threaded giant steps `i ∈ [lo, hi)` for one query; returns the
    /// first `t` found. `progress(done)` is called every 100 ms.
    fn steps(
        &self,
        q: &ProjectivePoint,
        lo: u64,
        hi: u64,
        progress: &mut dyn FnMut(u64),
    ) -> Option<u64> {
        let span = hi - lo;
        let chunk = span.div_ceil(self.threads as u64);
        let batches = chunk.div_ceil(self.table.step());
        let stop = AtomicBool::new(false);
        // One progress counter per thread (see `Counter`).
        let done: Vec<Counter> = (0..self.threads).map(|_| Counter::default()).collect();
        let (sender, receiver) = mpsc::channel::<u64>();
        let mut result = None;
        thread::scope(|scope| {
            for thread in 0..self.threads as u64 {
                let start = lo + thread * chunk;
                if start >= hi {
                    break;
                }
                let sender = sender.clone();
                let (stop, done) = (&stop, &done[thread as usize]);
                scope.spawn(move || {
                    let k0 = start + self.table.half as u64;
                    if let Some(t) = self.walk(q, k0, batches, stop, done) {
                        let _ = sender.send(t);
                    }
                });
            }
            drop(sender);
            loop {
                match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(t) => {
                        result = Some(t);
                        break;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        progress(done.iter().map(Counter::get).sum::<u64>().min(span));
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            stop.store(true, Ordering::Relaxed);
        });
        result
    }

    /// Walks `batches` batches of `Q − i·m·G` from `i = k0`; returns the
    /// smallest verified `t` of the first batch that has one.
    fn walk(
        &self,
        q: &ProjectivePoint,
        mut k0: u64,
        batches: u64,
        stop: &AtomicBool,
        done: &Counter,
    ) -> Option<u64> {
        let table = &self.table;
        let mut walk = Walk::new(table, Scalar::from(k0), *q);
        let mut hits: Vec<u64> = Vec::new();
        let mut js: Vec<u32> = Vec::new();
        let half = table.half as i64;
        // Top limbs of the batch, indexed by offset + H: the table lookups run
        // in a separate tight loop so their (random, cache-missing) loads
        // overlap instead of being serialised behind the field arithmetic.
        let mut tops = vec![0u64; 2 * table.half + 1];
        let mut filtered: Vec<(usize, u64)> = Vec::new();
        for _ in 0..batches {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let visited = walk.batch(TopCollector {
                tops: &mut tops,
                half,
            });
            if visited {
                // Filter pass first, bucket scans after: each pass is a tight
                // loop of independent loads, so the cache misses overlap.
                filtered.clear();
                filtered.extend(
                    tops.iter()
                        .enumerate()
                        .map(|(idx, &key)| (idx, key))
                        .filter(|&(_, key)| self.baby.filter_hit(key)),
                );
                for &(idx, key) in &filtered {
                    self.baby.scan_bucket(key, &mut js);
                    for j in js.drain(..) {
                        self.check(q, k0 as i64 + idx as i64 - half, j, &mut hits);
                    }
                }
            } else {
                slow_batch(table, q, k0, |offset, point| {
                    let i = k0 as i64 + offset;
                    if bool::from(point.is_identity()) {
                        self.check(q, i, 0, &mut hits);
                    } else {
                        self.baby.matches(&affine_xy(&point).0, &mut js);
                        for j in js.drain(..) {
                            self.check(q, i, j, &mut hits);
                        }
                    }
                });
            }
            done.add(table.step());
            k0 += table.step();
            if let Some(t) = hits.iter().min() {
                return Some(*t);
            }
        }
        None
    }

    /// x-match at giant step `i` against baby `j`: verifies `t = i·m ± j` with k256.
    fn check(&self, q: &ProjectivePoint, i: i64, j: u32, hits: &mut Vec<u64>) {
        if i < 0 {
            return;
        }
        let base = i as u64 * self.baby.size();
        for t in [base.checked_add(j as u64), base.checked_sub(j as u64)]
            .into_iter()
            .flatten()
        {
            if t < self.max && ProjectivePoint::GENERATOR * Scalar::from(t) == *q {
                hits.push(t);
            }
        }
    }
}

/// Records the top limb of every visited x, by offset.
struct TopCollector<'a> {
    tops: &'a mut [u64],
    half: i64,
}

impl Visit for TopCollector<'_> {
    #[inline(always)]
    fn visit(&mut self, offset: i64, x: &Fe) {
        self.tops[(offset + self.half) as usize] = x.top_limb();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::random_scalar;

    fn params(max_bits: u32, half: usize) -> Params {
        Params {
            max_bits,
            threads: 2,
            half,
        }
    }

    #[test]
    fn baby_table_lookup() {
        let baby = BabyTable::build(10, 8).unwrap();
        assert_eq!(baby.keys.len(), 1023);
        let mut out = Vec::new();
        for j in 1u64..1024 {
            let x = affine_xy(&(ProjectivePoint::GENERATOR * Scalar::from(j))).0;
            out.clear();
            baby.matches(&x, &mut out);
            assert_eq!(out, vec![j as u32], "j = {j}");
        }
        out.clear();
        baby.matches(
            &affine_xy(&(ProjectivePoint::GENERATOR * Scalar::from(5000u64))).0,
            &mut out,
        );
        assert!(out.is_empty());
        assert!(BabyTable::build(3, 8).is_err());
        assert!(BabyTable::build(29, 8).is_err());
    }

    #[test]
    fn query_inverts_apply_point() {
        let d = random_scalar().unwrap();
        let base = ProjectivePoint::GENERATOR * d;
        for tweak in Tweak::variants(123_456) {
            let target = tweak.apply_point(&base);
            let q = query(&base, &target, tweak.endo, tweak.negate);
            assert_eq!(
                q,
                ProjectivePoint::GENERATOR * Scalar::from(tweak.t),
                "{tweak}"
            );
        }
    }

    /// Random `t < 2^30` for all six variants: the recovered tweak reproduces
    /// the target point (small baby table, reduced range, 2 threads).
    #[test]
    fn recovers_all_variants() {
        let d = random_scalar().unwrap();
        let base = ProjectivePoint::GENERATOR * d;
        let baby = BabyTable::build(16, 64).unwrap();
        let p = params(30, 64);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (endo, negate) in VARIANTS {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let tweak = Tweak {
                t: seed & ((1 << 30) - 1),
                endo,
                negate,
            };
            let target = tweak.apply_point(&base);
            let got = recover(&base, &target, &p, &baby, &mut |_| {})
                .unwrap_or_else(|| panic!("no tweak for {tweak}"));
            assert_eq!(got.apply_point(&base), target, "{tweak} vs {got}");
        }
    }

    /// Edge offsets around the degenerate walk positions: `t = 0`, exact
    /// multiples of `m` (visited point at infinity), the initial centre at
    /// infinity (`t = H·m`) and the range end.
    #[test]
    fn recovers_degenerate_offsets() {
        let d = random_scalar().unwrap();
        let base = ProjectivePoint::GENERATOR * d;
        let half = 8usize;
        let baby = BabyTable::build(8, half).unwrap();
        let p = Params {
            max_bits: 20,
            threads: 3,
            half,
        };
        let m = 1u64 << 8;
        let h = half as u64;
        let step = 2 * h + 1;
        let offsets = [
            0,
            1,
            m - 1,
            m,
            m + 1,
            h * m,
            h * m + 3,
            (h + step) * m,
            (2 * h + 1) * m,
            (2 * h) * m + 5,
            (1 << 20) - 1,
            (1 << 20) - m,
            (1 << 19) + m * 7 - 1,
        ];
        for t in offsets {
            for tweak in Tweak::variants(t) {
                let target = tweak.apply_point(&base);
                let got = recover(&base, &target, &p, &baby, &mut |_| {})
                    .unwrap_or_else(|| panic!("no tweak for {tweak}"));
                assert_eq!(got.apply_point(&base), target, "{tweak} vs {got}");
            }
        }
    }

    #[test]
    fn levels_partition_the_range() {
        assert_eq!(
            levels(1 << 30),
            vec![(0, 256), (256, 65536), (65536, 1 << 24), (1 << 24, 1 << 30)]
        );
        assert_eq!(levels(100), vec![(0, 100)]);
        assert_eq!(levels(256), vec![(0, 256)]);
        assert_eq!(levels(257), vec![(0, 256), (256, 257)]);
        assert!(levels(0).is_empty());
    }

    #[test]
    fn out_of_range_offset_is_not_found() {
        let d = random_scalar().unwrap();
        let base = ProjectivePoint::GENERATOR * d;
        let baby = BabyTable::build(8, 8).unwrap();
        let p = params(16, 8);
        let tweak = Tweak {
            t: 1 << 16,
            endo: 1,
            negate: false,
        };
        let target = tweak.apply_point(&base);
        let mut reports = 0;
        assert!(recover(&base, &target, &p, &baby, &mut |_| reports += 1).is_none());
        // A tweak from a different base key is not found either.
        let other = ProjectivePoint::GENERATOR * random_scalar().unwrap();
        assert!(recover(&other, &target, &p, &baby, &mut |_| {}).is_none());
    }
}
