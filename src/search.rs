//! VanitySearch-style batched search over `C ± j·G`.
//!
//! One worker owns a moving centre point `C = k0·G`. Per batch it computes the
//! x coordinates of `C + j·G` and `C - j·G` for `j = 1..=H` with a single field
//! inversion (Montgomery's trick), tests `x`, `β·x` and `β²·x` (endomorphism:
//! scalars `k`, `λk`, `λ²k`) against the patterns, then jumps `C += (2H+1)·G`
//! using the same batched inversion. Result y coordinates are never computed:
//! the y parity is fixed afterwards by negating the scalar.

use crate::field::Fe;
use crate::pattern::PatternSet;
use crate::tweak::{MAX_TWEAK_BITS, Tweak, lambda_pow};
use k256::elliptic_curve::Group;
use k256::elliptic_curve::point::AffineCoordinates;
use k256::elliptic_curve::scalar::FromUintUnchecked;
use k256::elliptic_curve::sec1::ToSec1Point;
use k256::elliptic_curve::{Generate, PrimeField};
use k256::{NonZeroScalar, ProjectivePoint, Scalar, U256};

const LAMBDA_HEX: &str = "5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72";

/// λ such that λ·(x, y) = (β·x, y).
pub fn lambda() -> Scalar {
    Scalar::from_uint_unchecked(U256::from_be_hex(LAMBDA_HEX))
}

/// Affine x and y of a k256 point as field elements.
pub fn affine_xy(point: &ProjectivePoint) -> (Fe, Fe) {
    let affine = point.to_affine();
    let x: [u8; 32] = affine.x().into();
    let y: [u8; 32] = affine.y().into();
    (Fe::from_bytes_be(&x), Fe::from_bytes_be(&y))
}

/// Compressed SEC1 encoding of a k256 point.
pub fn compressed(point: &ProjectivePoint) -> [u8; 33] {
    let sec1 = point.to_affine().to_sec1_point(true);
    let mut out = [0u8; 33];
    out.copy_from_slice(sec1.as_bytes());
    out
}

/// Precomputed `j·P` for `j = 1..=H` plus the batch jump `(2H+1)·P`, for a
/// generator `P` (`G` for the key search, `−m·G` for the giant steps of
/// `recover`).
pub struct Table {
    pub half: usize,
    pub generator: ProjectivePoint,
    x: Vec<Fe>,
    y: Vec<Fe>,
    jump_x: Fe,
    jump_y: Fe,
}

impl Table {
    pub fn new(half: usize, generator: ProjectivePoint) -> Table {
        let half = half.max(1);
        let mut x = Vec::with_capacity(half);
        let mut y = Vec::with_capacity(half);
        let mut point = generator;
        for _ in 0..half {
            let (px, py) = affine_xy(&point);
            x.push(px);
            y.push(py);
            point += generator;
        }
        let jump = generator * Scalar::from(2 * half as u64 + 1);
        let (jump_x, jump_y) = affine_xy(&jump);
        Table {
            half,
            generator,
            x,
            y,
            jump_x,
            jump_y,
        }
    }

    /// Scalar distance between consecutive batch centres.
    pub fn step(&self) -> u64 {
        2 * self.half as u64 + 1
    }

    /// x candidates produced per batch (`x`, `βx`, `β²x` for `2H+1` points).
    pub fn candidates_per_batch(&self) -> u64 {
        3 * self.step()
    }
}

/// Independent product chains in `batch_invert`: one chain would make every
/// multiplication wait for the previous one (latency bound); `LANES`
/// interleaved chains keep the multiplier busy. Lane `k` holds the indices
/// `i ≡ k (mod LANES)`.
const LANES: usize = 4;

/// Inverts every element of `values` with one field inversion.
/// `scratch` and `out` must have the same length as `values`.
/// Returns `false` (leaving `out` unspecified) if any value is zero.
pub fn batch_invert(values: &[Fe], scratch: &mut [Fe], out: &mut [Fe]) -> bool {
    let n = values.len();
    if n == 0 {
        return true;
    }
    let scratch = &mut scratch[..n];
    let out = &mut out[..n];
    // Forward pass: scratch[i] = values[i mod LANES] · … · values[i - LANES] · values[i].
    let head = LANES.min(n);
    scratch[..head].copy_from_slice(&values[..head]);
    for i in LANES..n {
        scratch[i] = scratch[i - LANES].mul(&values[i]);
    }
    // The lane totals are the last `head` entries (one per lane); invert their
    // product once and split it with Montgomery's trick over the lanes.
    let totals = &scratch[n - head..];
    let mut prefix = [Fe::ONE; LANES];
    for k in 1..head {
        prefix[k] = prefix[k - 1].mul(&totals[k - 1]);
    }
    let all = prefix[head - 1].mul(&totals[head - 1]);
    if all.is_zero() {
        return false;
    }
    let mut inv = all.invert();
    // state[lane] = 1 / (product of that lane), indexed by the lane of the total.
    let mut state = [Fe::ZERO; LANES];
    for k in (0..head).rev() {
        state[(n - head + k) % LANES] = inv.mul(&prefix[k]);
        inv = inv.mul(&totals[k]);
    }
    // Backward pass: out[i] = state · scratch[i - LANES]; state ·= values[i].
    for i in (LANES..n).rev() {
        let lane = i % LANES;
        out[i] = state[lane].mul(&scratch[i - LANES]);
        state[lane] = state[lane].mul(&values[i]);
    }
    out[..head].copy_from_slice(&state[..head]);
    true
}

/// A pattern hit before scalar reconstruction.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// Batch start scalar.
    pub k0: Scalar,
    /// `k = k0 + offset` before the endomorphism.
    pub offset: i64,
    /// Power of λ applied (0, 1 or 2).
    pub endo: u8,
    /// x coordinate of `λ^endo · (k0 + offset) · G`.
    pub x: Fe,
    pub pattern: usize,
}

/// What a search produces: the scan key itself (random mode) or the public
/// tweak that turns the base key into it (split-key mode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Secret([u8; 32]),
    Tweak(Tweak),
}

/// A verified vanity scan key.
#[derive(Clone, Debug)]
pub struct Found {
    pub key: Key,
    pub pubkey: [u8; 33],
    pub pattern: usize,
}

/// How candidate scalars map to points: `k·G` (random start) or
/// `D + t·G` (split-key mode, `t` small).
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Random,
    Split { base: ProjectivePoint },
}

/// Per-point callback of [`Worker::batch`]. Implemented for closures; hot
/// visitors implement it directly with `#[inline(always)]` on `visit`: the
/// compiler does not inline a closure this large into the batch loop by
/// itself, and the call boundary (x through memory, spills) costs about 30%.
pub trait Visit {
    fn visit(&mut self, offset: i64, x: &Fe);

    /// `visit(offset, x_plus)` then `visit(-offset, x_minus)`. Hot visitors
    /// override it to do all their arithmetic before any branch: the compiler
    /// schedules within basic blocks, and the match checks split them.
    #[inline(always)]
    fn visit_pair(&mut self, offset: i64, x_plus: &Fe, x_minus: &Fe) {
        self.visit(offset, x_plus);
        self.visit(-offset, x_minus);
    }
}

impl<F: FnMut(i64, &Fe)> Visit for F {
    #[inline(always)]
    fn visit(&mut self, offset: i64, x: &Fe) {
        self(offset, x)
    }
}

pub struct Worker<'a> {
    table: &'a Table,
    /// Point added to every `k·P` of the walk (identity for the key search).
    base: ProjectivePoint,
    k0: Scalar,
    cx: Fe,
    cy: Fe,
    /// Centre is the point at infinity: the batch formulas do not apply.
    degenerate: bool,
    dx: Vec<Fe>,
    scratch: Vec<Fe>,
    inv: Vec<Fe>,
}

impl<'a> Worker<'a> {
    /// Worker starting at a random scalar.
    pub fn new(table: &'a Table) -> Result<Worker<'a>, String> {
        let start = NonZeroScalar::try_generate().map_err(|e| format!("rng failure: {e}"))?;
        Ok(Self::with_start(table, *start.as_ref()))
    }

    /// Worker walking `k·P` from `k0`.
    pub fn with_start(table: &'a Table, k0: Scalar) -> Worker<'a> {
        Self::with_base(table, k0, ProjectivePoint::IDENTITY)
    }

    /// Worker walking `base + k·P` from `k0`: the batch at `k0` visits
    /// `base + (k0 + offset)·P` for `offset ∈ -H..=H`.
    pub fn with_base(table: &'a Table, k0: Scalar, base: ProjectivePoint) -> Worker<'a> {
        let n = table.half + 1;
        let mut worker = Worker {
            table,
            base,
            k0,
            cx: Fe::ZERO,
            cy: Fe::ZERO,
            degenerate: false,
            dx: vec![Fe::ZERO; n],
            scratch: vec![Fe::ZERO; n],
            inv: vec![Fe::ZERO; n],
        };
        worker.recompute_centre();
        worker
    }

    /// Sets the centre to `base + k0·P` with k256.
    fn recompute_centre(&mut self) {
        let centre = self.base + self.table.generator * self.k0;
        self.degenerate = bool::from(centre.is_identity());
        let (cx, cy) = affine_xy(&centre);
        self.cx = cx;
        self.cy = cy;
    }

    /// Skips the current batch: advances the centre with k256 instead.
    fn skip_batch(&mut self) {
        self.k0 = self.k0.add(&Scalar::from(self.table.step()));
        self.recompute_centre();
    }

    /// Runs one batch, calling `visitor.visit(offset, x)` for `x(base + (k0 + offset)·P)`
    /// with `offset` in `-H..=H`, then moves to the next batch (`k0 += 2H+1`).
    /// Returns `false` if the batch was skipped because a visited point or the
    /// centre is the point at infinity (zero difference); `k0` still advances,
    /// so the caller can redo that batch with k256 if it matters.
    #[inline]
    pub fn batch<F: FnMut(i64, &Fe)>(&mut self, visit: F) -> bool {
        self.batch_with(visit)
    }

    /// [`Worker::batch`] with a [`Visit`] implementation (see the trait for
    /// why the hot visitors are not closures).
    #[inline]
    pub fn batch_with<V: Visit>(&mut self, mut visitor: V) -> bool {
        if self.degenerate {
            self.skip_batch();
            return false;
        }
        let table = self.table;
        let h = table.half;
        let (cx, cy) = (self.cx, self.cy);
        let xs = &table.x[..h];
        let ys = &table.y[..h];
        {
            let (dx, dx_jump) = self.dx.split_at_mut(h);
            for (d, tx) in dx.iter_mut().zip(xs) {
                *d = tx.sub(&cx);
            }
            dx_jump[0] = table.jump_x.sub(&cx);
        }
        if !batch_invert(&self.dx, &mut self.scratch, &mut self.inv) {
            // C == ±j·P for some j in the table (probability ~2^-247 for a
            // random start; a real case for the giant steps of `recover`):
            // jump via k256 instead and skip this batch.
            self.skip_batch();
            return false;
        }
        visitor.visit(0, &cx);
        for (j, ((tx, ty), inv)) in xs.iter().zip(ys).zip(&self.inv[..h]).enumerate() {
            let offset = j as i64 + 1;
            // x(C ± T) = λ² − cx − tx with λ = (±ty − cy) / (tx − cx); only λ²
            // is needed, so the slope of C − T is taken as (ty + cy) / dx.
            let sum = cx.add(tx);
            let lambda_plus = ty.sub(&cy).mul(inv);
            let lambda_minus = ty.add(&cy).mul(inv);
            let x_plus = lambda_plus.square().sub(&sum);
            let x_minus = lambda_minus.square().sub(&sum);
            visitor.visit_pair(offset, &x_plus, &x_minus);
        }
        // Jump: C += (2H+1)·G.
        let lambda = table.jump_y.sub(&cy).mul(&self.inv[h]);
        let nx = lambda.square().sub(&cx).sub(&table.jump_x);
        let ny = lambda.mul(&cx.sub(&nx)).sub(&cy);
        self.cx = nx;
        self.cy = ny;
        self.k0 = self.k0.add(&Scalar::from(table.step()));
        true
    }

    /// One batch of pattern matching; pushes hits into `out`.
    #[inline]
    pub fn search_batch(&mut self, patterns: &PatternSet, out: &mut Vec<Candidate>) -> bool {
        let k0 = self.k0;
        self.batch_with(Searcher {
            keys: &patterns.keys,
            patterns,
            out,
            k0,
        })
    }
}

/// Tests `x`, `β·x` and `β²·x` of every visited point against the patterns.
struct Searcher<'a> {
    /// Top-limb (mask, value) of every pattern, in a flat array so the check
    /// is one load per pattern and nothing is reloaded through `PatternSet`.
    keys: &'a [(u64, u64)],
    patterns: &'a PatternSet,
    out: &'a mut Vec<Candidate>,
    k0: Scalar,
}

impl Visit for Searcher<'_> {
    #[inline(always)]
    fn visit(&mut self, offset: i64, x: &Fe) {
        let [bx, b2x] = endomorphisms(x);
        self.check(offset, x, &bx, &b2x);
    }

    #[inline(always)]
    fn visit_pair(&mut self, offset: i64, x_plus: &Fe, x_minus: &Fe) {
        let [bx_plus, b2x_plus] = endomorphisms(x_plus);
        let [bx_minus, b2x_minus] = endomorphisms(x_minus);
        self.check(offset, x_plus, &bx_plus, &b2x_plus);
        self.check(-offset, x_minus, &bx_minus, &b2x_minus);
    }
}

/// `β·x` and `β²·x`. Since β² + β + 1 = 0, β²·x = −(x + β·x): an add and a
/// negation instead of a second multiplication.
#[inline(always)]
fn endomorphisms(x: &Fe) -> [Fe; 2] {
    let bx = Fe::BETA.mul(x);
    [bx, bx.add(x).neg()]
}

impl Searcher<'_> {
    /// Top-limb test of the three candidates of one point.
    #[inline(always)]
    fn check(&mut self, offset: i64, x: &Fe, bx: &Fe, b2x: &Fe) {
        for (endo, candidate) in [x, bx, b2x].into_iter().enumerate() {
            let top = candidate.top_limb();
            if self.keys.iter().any(|&(mask, value)| top & mask == value) {
                push_candidate(
                    self.out,
                    self.patterns,
                    self.k0,
                    offset,
                    endo as u8,
                    candidate,
                );
            }
        }
    }
}

/// Full check of a candidate whose top limb matched some pattern.
#[cold]
#[inline(never)]
fn push_candidate(
    out: &mut Vec<Candidate>,
    patterns: &PatternSet,
    k0: Scalar,
    offset: i64,
    endo: u8,
    x: &Fe,
) {
    if let Some(pattern) = patterns.find(x) {
        out.push(Candidate {
            k0,
            offset,
            endo,
            x: *x,
            pattern,
        });
    }
}

/// Reconstructs the scalar for a candidate, checks it against k256, and fixes
/// the y parity when the pattern requires one.
pub fn resolve(candidate: &Candidate, patterns: &PatternSet, mode: &Mode) -> Result<Found, String> {
    let magnitude = Scalar::from(candidate.offset.unsigned_abs());
    let k = if candidate.offset >= 0 {
        candidate.k0.add(&magnitude)
    } else {
        candidate.k0.sub(&magnitude)
    };
    let pattern = patterns
        .patterns
        .get(candidate.pattern)
        .ok_or_else(|| "internal: pattern index out of range".to_string())?;
    let (mut point, mut key) = match mode {
        Mode::Random => {
            let k = k.mul(&lambda_pow(candidate.endo));
            (
                ProjectivePoint::GENERATOR * k,
                Key::Secret(k.to_bytes().into()),
            )
        }
        Mode::Split { base } => {
            let t = scalar_to_u64(&k)
                .filter(|t| *t < 1 << MAX_TWEAK_BITS)
                .ok_or_else(|| "internal: split-key offset out of range".to_string())?;
            let tweak = Tweak {
                t,
                endo: candidate.endo,
                negate: false,
            };
            (tweak.apply_point(base), Key::Tweak(tweak))
        }
    };
    let (x, _) = affine_xy(&point);
    if x != candidate.x {
        return Err("internal: reconstructed key does not reproduce the candidate x".to_string());
    }
    let mut pubkey = compressed(&point);
    if let Some(want_odd) = pattern.parity
        && (pubkey[0] == 0x03) != want_odd
    {
        point = -point;
        key = match key {
            Key::Secret(bytes) => {
                let k = Scalar::from_repr_vartime(bytes.into())
                    .ok_or_else(|| "internal: secret out of range".to_string())?;
                Key::Secret(k.negate().to_bytes().into())
            }
            Key::Tweak(tweak) => Key::Tweak(Tweak {
                negate: true,
                ..tweak
            }),
        };
        pubkey = compressed(&point);
    }
    if !pattern.matches_bytes(&pubkey[1..].try_into().map_err(|_| "internal: bad pubkey")?) {
        return Err("internal: pubkey x does not match the pattern".to_string());
    }
    Ok(Found {
        key,
        pubkey,
        pattern: candidate.pattern,
    })
}

/// The scalar as a `u64` if it fits.
pub fn scalar_to_u64(k: &Scalar) -> Option<u64> {
    let bytes: [u8; 32] = k.to_bytes().into();
    if bytes[..24].iter().any(|&b| b != 0) {
        return None;
    }
    let mut low = [0u8; 8];
    low.copy_from_slice(&bytes[24..]);
    Some(u64::from_be_bytes(low))
}

/// Start offset of split-mode worker `thread`: `thread·2^range_bits + H`, so
/// its first batch visits `thread·2^range_bits ..= thread·2^range_bits + 2H`.
pub fn split_start(thread: usize, range_bits: u32, half: usize) -> u64 {
    ((thread as u64) << range_bits) + half as u64
}

/// Number of batches a split-mode worker may run before its highest visited
/// offset (`k0 + H`) would leave its `2^range_bits` range.
pub fn split_batches(range_bits: u32, half: usize) -> u64 {
    ((1u64 << range_bits) - 2 * half as u64) / (2 * half as u64 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address;
    use crate::pattern::Network;
    use k256::elliptic_curve::PrimeField;

    fn random_scalar() -> Scalar {
        *NonZeroScalar::try_generate().unwrap().as_ref()
    }

    fn x_of(k: &Scalar) -> Fe {
        affine_xy(&(ProjectivePoint::GENERATOR * k)).0
    }

    fn g_table(half: usize) -> Table {
        Table::new(half, ProjectivePoint::GENERATOR)
    }

    fn secret_of(found: &Found) -> Scalar {
        match found.key {
            Key::Secret(bytes) => Scalar::from_repr_vartime(bytes.into()).unwrap(),
            Key::Tweak(_) => panic!("expected a secret"),
        }
    }

    #[test]
    fn batch_invert_matches_individual_inverts() {
        let values: Vec<Fe> = (0..37).map(|_| x_of(&random_scalar())).collect();
        let mut scratch = vec![Fe::ZERO; values.len()];
        let mut out = vec![Fe::ZERO; values.len()];
        assert!(batch_invert(&values, &mut scratch, &mut out));
        for (v, inv) in values.iter().zip(&out) {
            assert_eq!(*inv, v.invert());
            assert_eq!(v.mul(inv), Fe::ONE);
        }
        let mut with_zero = values.clone();
        with_zero[5] = Fe::ZERO;
        assert!(!batch_invert(&with_zero, &mut scratch, &mut out));
        assert!(batch_invert(&[], &mut [], &mut []));
        // Lengths around the lane count, and a zero in every lane position.
        for n in 1..=2 * LANES + 1 {
            let values = &values[..n];
            let mut scratch = vec![Fe::ZERO; n];
            let mut out = vec![Fe::ZERO; n];
            assert!(batch_invert(values, &mut scratch, &mut out), "n = {n}");
            for (v, inv) in values.iter().zip(&out) {
                assert_eq!(v.mul(inv), Fe::ONE, "n = {n}");
            }
            for zero_at in 0..n {
                let mut with_zero = values.to_vec();
                with_zero[zero_at] = Fe::ZERO;
                assert!(!batch_invert(&with_zero, &mut scratch, &mut out));
            }
        }
    }

    #[test]
    fn batch_x_coordinates_match_k256() {
        let table = g_table(256);
        let k0 = random_scalar();
        let mut worker = Worker::with_start(&table, k0);
        let mut seen = vec![None; 2 * table.half + 1];
        assert!(worker.batch(|offset, x| {
            let idx = (offset + table.half as i64) as usize;
            assert!(seen[idx].is_none(), "offset {offset} visited twice");
            seen[idx] = Some(*x);
        }));
        for (idx, x) in seen.iter().enumerate() {
            let offset = idx as i64 - table.half as i64;
            let magnitude = Scalar::from(offset.unsigned_abs());
            let k = if offset >= 0 {
                k0.add(&magnitude)
            } else {
                k0.sub(&magnitude)
            };
            assert_eq!(x.unwrap(), x_of(&k), "offset {offset}");
        }
        // After the batch the centre moved by 2H+1.
        let next = k0.add(&Scalar::from(table.step()));
        assert_eq!(worker.k0, next);
        assert_eq!(
            affine_xy(&(ProjectivePoint::GENERATOR * next)),
            (worker.cx, worker.cy)
        );
        // And a second batch is consistent too.
        let mut count = 0;
        assert!(worker.batch(|offset, x| {
            count += 1;
            let k = if offset >= 0 {
                next.add(&Scalar::from(offset.unsigned_abs()))
            } else {
                next.sub(&Scalar::from(offset.unsigned_abs()))
            };
            assert_eq!(*x, x_of(&k));
        }));
        assert_eq!(count, 2 * table.half + 1);
    }

    /// Generic generator and base: the walk visits `base + (k0 + offset)·P`.
    #[test]
    fn batch_with_base_and_generator() {
        let m = random_scalar();
        let generator = ProjectivePoint::GENERATOR * m;
        let table = Table::new(16, generator);
        let base = ProjectivePoint::GENERATOR * random_scalar();
        let mut worker = Worker::with_base(&table, Scalar::from(1000u64), base);
        for batch in 0..3u64 {
            let k0 = 1000 + batch * table.step();
            let mut count = 0;
            assert!(worker.batch(|offset, x| {
                count += 1;
                let k = Scalar::from((k0 as i64 + offset) as u64);
                let expected = affine_xy(&(base + generator * k)).0;
                assert_eq!(*x, expected, "batch {batch} offset {offset}");
            }));
            assert_eq!(count, 2 * table.half + 1);
        }
    }

    /// Degenerate centres: the point at infinity and `±j·P` are skipped, the
    /// walk keeps its position, and the next batch is correct again.
    #[test]
    fn degenerate_batches_are_skipped_not_corrupted() {
        let table = g_table(8);
        // Centre = H·G: equals the table entry j = H.
        let mut worker = Worker::with_start(&table, Scalar::from(8u64));
        assert!(!worker.batch(|_, _| panic!("visited a degenerate batch")));
        assert_eq!(worker.k0, Scalar::from(8 + 17u64));
        assert!(worker.batch(|offset, x| {
            assert_eq!(*x, x_of(&Scalar::from((25 + offset) as u64)));
        }));
        // Centre at infinity: base = −k0·G.
        let base = -(ProjectivePoint::GENERATOR * Scalar::from(100u64));
        let mut worker = Worker::with_base(&table, Scalar::from(100u64), base);
        assert!(worker.degenerate);
        assert!(!worker.batch(|_, _| panic!("visited a degenerate batch")));
        assert!(!worker.degenerate);
        // The next centre is the jump point (2H+1)·G itself: skipped too.
        assert!(!worker.batch(|_, _| panic!("visited a degenerate batch")));
        assert!(worker.batch(|offset, x| {
            assert_eq!(*x, x_of(&Scalar::from((34 + offset) as u64)));
        }));
        // A visited point at infinity (offset ≠ 0): base = −(k0 + 3)·G.
        let base = -(ProjectivePoint::GENERATOR * Scalar::from(103u64));
        let mut worker = Worker::with_base(&table, Scalar::from(100u64), base);
        assert!(!worker.batch(|_, _| panic!("visited a degenerate batch")));
        assert!(worker.batch(|offset, x| {
            assert_eq!(*x, x_of(&Scalar::from((14 + offset) as u64)));
        }));
    }

    #[test]
    fn endomorphism_scalar_relation() {
        for _ in 0..8 {
            let k = random_scalar();
            let x = x_of(&k);
            let lk = k.mul(&lambda());
            let l2k = lk.mul(&lambda());
            assert_eq!(x_of(&lk), Fe::BETA.mul(&x));
            assert_eq!(x_of(&l2k), Fe::BETA2.mul(&x));
            assert_eq!(x_of(&l2k.mul(&lambda())), x);
        }
    }

    #[test]
    fn small_table_and_step() {
        let table = g_table(4);
        assert_eq!(table.step(), 9);
        assert_eq!(table.candidates_per_batch(), 27);
        assert_eq!(g_table(0).half, 1);
    }

    #[test]
    fn scalar_conversion_and_split_ranges() {
        assert_eq!(scalar_to_u64(&Scalar::from(u64::MAX)), Some(u64::MAX));
        assert_eq!(scalar_to_u64(&Scalar::ZERO), Some(0));
        assert_eq!(
            scalar_to_u64(&Scalar::from(u64::MAX).add(&Scalar::ONE)),
            None
        );
        assert_eq!(scalar_to_u64(&Scalar::ONE.negate()), None);
        assert_eq!(split_start(0, 44, 1024), 1024);
        assert_eq!(split_start(3, 44, 1024), 3 * (1 << 44) + 1024);
        // The last batch of a worker stays inside its range.
        for (bits, half) in [(44u32, 1024usize), (20, 64), (12, 8)] {
            let batches = split_batches(bits, half);
            let last_k0 = split_start(0, bits, half) + (batches - 1) * (2 * half as u64 + 1);
            assert!(last_k0 + (half as u64) < (1u64 << bits));
            assert!(
                last_k0 + (half as u64) + (2 * half as u64 + 1) >= (1u64 << bits) - 2 * half as u64
            );
        }
    }

    /// End-to-end: search a short pattern, then independently recompute the
    /// address from the secret key with k256 + bech32 and decode it.
    #[test]
    fn search_and_verify_independently() {
        let table = g_table(64);
        let spend = [2u8; 33];
        for (input, network, hrp) in [
            ("sp1qq?q", Network::Mainnet, address::HRP_MAINNET),
            ("sp1qqgp", Network::Mainnet, address::HRP_MAINNET),
            ("sp1qqvz", Network::Mainnet, address::HRP_MAINNET),
            ("tsp1qq?r", Network::Testnet, address::HRP_TESTNET),
        ] {
            let patterns = PatternSet::parse(&[input.to_string()], network).unwrap();
            let mut worker = Worker::new(&table).unwrap();
            let mut hits = Vec::new();
            let mut batches = 0;
            while hits.is_empty() {
                worker.search_batch(&patterns, &mut hits);
                batches += 1;
                assert!(batches < 10_000, "no hit for {input}");
            }
            for candidate in &hits {
                let found = resolve(candidate, &patterns, &Mode::Random).unwrap();
                // Independent path: scalar → point → compressed → bech32m.
                let point = ProjectivePoint::GENERATOR * secret_of(&found);
                let pubkey = compressed(&point);
                assert_eq!(pubkey, found.pubkey);
                let addr = address::encode(hrp, &pubkey, &spend);
                assert!(
                    patterns.patterns[found.pattern].matches_address(&addr),
                    "{input}: {addr}"
                );
                let (got_hrp, version, payload) = address::decode(&addr).unwrap();
                assert_eq!(got_hrp, hrp);
                assert_eq!(version, bech32::Fe32::Q);
                assert_eq!(&payload[..33], &pubkey);
            }
        }
    }

    /// Split-key mode with two workers on disjoint ranges: every tweak maps the
    /// base secret to the found key and stays inside its worker's range.
    #[test]
    fn split_mode_tweaks_reproduce_keys() {
        let table = g_table(64);
        let range_bits = 20;
        let d = random_scalar();
        let base = ProjectivePoint::GENERATOR * d;
        let mode = Mode::Split { base };
        for input in ["sp1qq?q", "sp1qqgp", "sp1qqvz"] {
            let patterns = PatternSet::parse(&[input.to_string()], Network::Mainnet).unwrap();
            let mut workers: Vec<Worker> = (0..2)
                .map(|i| {
                    let k0 = Scalar::from(split_start(i, range_bits, table.half));
                    Worker::with_base(&table, k0, base)
                })
                .collect();
            let mut hits = Vec::new();
            let mut batches = 0;
            while hits.is_empty() {
                for worker in &mut workers {
                    worker.search_batch(&patterns, &mut hits);
                }
                batches += 1;
                assert!(batches < 10_000, "no hit for {input}");
            }
            for candidate in &hits {
                let found = resolve(candidate, &patterns, &mode).unwrap();
                let Key::Tweak(tweak) = found.key else {
                    panic!("expected a tweak");
                };
                let range = tweak.t >> range_bits;
                assert!(range < 2, "{tweak}");
                let priv_key = tweak.apply(&d);
                assert_eq!(
                    compressed(&(ProjectivePoint::GENERATOR * priv_key)),
                    found.pubkey
                );
                assert_eq!(compressed(&tweak.apply_point(&base)), found.pubkey);
                let pattern = &patterns.patterns[found.pattern];
                if let Some(odd) = pattern.parity {
                    assert_eq!(found.pubkey[0] == 0x03, odd);
                }
                let addr = address::encode(address::HRP_MAINNET, &found.pubkey, &[2u8; 33]);
                assert!(pattern.matches_address(&addr), "{input}: {addr}");
                // Re-parse of the printed form gives the same point.
                let reparsed: Tweak = tweak.to_string().parse().unwrap();
                assert_eq!(reparsed, tweak);
            }
        }
    }

    #[test]
    fn resolve_rejects_wrong_x() {
        let table = g_table(4);
        let patterns = PatternSet::parse(&["sp1qq".to_string()], Network::Mainnet).unwrap();
        let worker = Worker::new(&table).unwrap();
        let candidate = Candidate {
            k0: worker.k0,
            offset: 1,
            endo: 0,
            x: Fe::ONE,
            pattern: 0,
        };
        assert!(resolve(&candidate, &patterns, &Mode::Random).is_err());
        let base = ProjectivePoint::GENERATOR * random_scalar();
        assert!(resolve(&candidate, &patterns, &Mode::Split { base }).is_err());
        // Split mode rejects offsets that do not fit the tweak range.
        let big = Candidate {
            k0: Scalar::from(1u64 << 52),
            offset: 0,
            endo: 0,
            x: affine_xy(&(base + ProjectivePoint::GENERATOR * Scalar::from(1u64 << 52))).0,
            pattern: 0,
        };
        let err = resolve(&big, &patterns, &Mode::Split { base }).unwrap_err();
        assert!(err.contains("out of range"), "{err}");
    }
}
