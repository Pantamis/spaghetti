//! VanitySearch-style batched walk over `C ± j·P`, and the key search built on it.
//!
//! A [`Walk`] owns a moving centre `C = base + k0·P`. Per batch it computes the
//! x coordinates of `C + j·P` and `C - j·P` for `j = 1..=H` with a single
//! field inversion (Montgomery's trick), hands every x to a [`Visit`]
//! implementation, then jumps `C += (2H+1)·P` using the same batched inversion.
//! Result y coordinates are never computed: the y parity is fixed afterwards by
//! negating the scalar.
//!
//! The key search ([`worker`]) walks with `P = G`, tests `x`, `β·x` and `β²·x`
//! (endomorphism: scalars `k`, `λk`, `λ²k`) against the patterns and
//! reconstructs hits with k256. `recover` reuses the walk with `P = −m·G`.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;

use k256::elliptic_curve::Generate;
use k256::elliptic_curve::Group;
use k256::elliptic_curve::point::AffineCoordinates;
use k256::elliptic_curve::sec1::ToSec1Point;
use k256::{NonZeroScalar, ProjectivePoint, Scalar};
use zeroize::Zeroizing;

use crate::field::Fe;
use crate::pattern::{Pattern, PatternSet};
use crate::tweak::{MAX_TWEAK_BITS, Tweak, lambda_pow};

/// Offsets covered by one split-mode range: `2^44`.
pub const RANGE_BITS: u32 = 44;
/// Number of split-mode ranges `[i·2^44, (i+1)·2^44)`; together they cover
/// every offset below `2^MAX_TWEAK_BITS`.
pub const SPLIT_RANGES: usize = 1 << (MAX_TWEAK_BITS - RANGE_BITS);

/// A random non-zero scalar from the OS RNG.
pub fn random_scalar() -> Result<Scalar, String> {
    NonZeroScalar::try_generate()
        .map(|k| *k.as_ref())
        .map_err(|e| format!("rng failure: {e}"))
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

/// Precomputed `j·P` for `j = 1..=H` plus the batch jump `(2H+1)·P`.
pub struct Table {
    /// `H >= 1`: points on each side of the centre per batch.
    pub half: usize,
    /// `P`: `G` for the key search, `−m·G` for the giant steps of `recover`.
    pub generator: ProjectivePoint,
    x: Vec<Fe>,
    y: Vec<Fe>,
    jump_x: Fe,
    jump_y: Fe,
}

impl Table {
    pub fn new(half: usize, generator: ProjectivePoint) -> Table {
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

    /// Scalar distance between consecutive batch centres, `2H+1`.
    pub fn step(&self) -> u64 {
        2 * self.half as u64 + 1
    }

    /// x candidates produced per batch (`x`, `βx`, `β²x` for `2H+1` points).
    fn candidates_per_batch(&self) -> u64 {
        3 * self.step()
    }
}

/// Independent product chains in `batch_invert`: one chain would make every
/// multiplication wait for the previous one (latency bound); `LANES`
/// interleaved chains keep the multiplier busy. Lane `k` holds the indices
/// `i ≡ k (mod LANES)`.
const LANES: usize = 4;

/// Inverts every element of `values` with one field inversion, in place into
/// `inv` (same length as `values`), which doubles as the scratch buffer.
/// Returns `false` (leaving `inv` unspecified) if any value is zero.
fn batch_invert(values: &[Fe], inv: &mut [Fe]) -> bool {
    let n = values.len();
    if n == 0 {
        return true;
    }
    let inv = &mut inv[..n];
    // Forward pass: inv[i] = values[i mod LANES] · … · values[i - LANES] · values[i].
    let head = LANES.min(n);
    inv[..head].copy_from_slice(&values[..head]);
    for i in LANES..n {
        inv[i] = inv[i - LANES].mul(&values[i]);
    }
    // The lane totals are the last `head` entries (one per lane); invert their
    // product once and split it with Montgomery's trick over the lanes.
    let totals = &inv[n - head..];
    let mut prefix = [Fe::ONE; LANES];
    for k in 1..head {
        prefix[k] = prefix[k - 1].mul(&totals[k - 1]);
    }
    let all = prefix[head - 1].mul(&totals[head - 1]);
    if all.is_zero() {
        return false;
    }
    let mut all_inv = all.invert();
    // state[lane] = 1 / (product of that lane), indexed by the lane of the total.
    let mut state = [Fe::ZERO; LANES];
    for k in (0..head).rev() {
        state[(n - head + k) % LANES] = all_inv.mul(&prefix[k]);
        all_inv = all_inv.mul(&totals[k]);
    }
    // Backward pass, in place: inv[i] = state · inv[i - LANES]; state ·= values[i].
    // Entry i is only ever read by step i + LANES, which ran before step i.
    for i in (LANES..n).rev() {
        let lane = i % LANES;
        inv[i] = state[lane].mul(&inv[i - LANES]);
        state[lane] = state[lane].mul(&values[i]);
    }
    inv[..head].copy_from_slice(&state[..head]);
    true
}

/// Per-point callback of [`Walk::batch`]. Implemented for closures; hot
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

/// The batched walk `base + k·P`, positioned at its next batch centre `k0`.
pub struct Walk<'a> {
    table: &'a Table,
    /// Added to every point of the walk (identity for the key search).
    base: ProjectivePoint,
    /// Random mode: the secret of every visited point minus a public offset,
    /// so it is wiped on drop.
    k0: Zeroizing<Scalar>,
    cx: Fe,
    cy: Fe,
    /// The centre is the point at infinity: the batch formulas do not apply.
    degenerate: bool,
    dx: Vec<Fe>,
    inv: Vec<Fe>,
}

impl<'a> Walk<'a> {
    /// Walk of `base + k·P` from `k0`: the first batch visits
    /// `base + (k0 + offset)·P` for `offset ∈ -H..=H`.
    pub fn new(table: &'a Table, k0: Scalar, base: ProjectivePoint) -> Walk<'a> {
        let n = table.half + 1;
        let mut walk = Walk {
            table,
            base,
            k0: Zeroizing::new(k0),
            cx: Fe::ZERO,
            cy: Fe::ZERO,
            degenerate: false,
            dx: vec![Fe::ZERO; n],
            inv: vec![Fe::ZERO; n],
        };
        walk.recompute_centre();
        walk
    }

    /// Walk of `k·P` from a random start.
    pub fn random(table: &'a Table) -> Result<Walk<'a>, String> {
        Ok(Self::new(
            table,
            random_scalar()?,
            ProjectivePoint::IDENTITY,
        ))
    }

    /// Moves the walk to a fresh random start (random mode, after a hit: keys
    /// from one run must not be related by a small public offset).
    pub fn reseed(&mut self) -> Result<(), String> {
        *self.k0 = random_scalar()?;
        self.recompute_centre();
        Ok(())
    }

    /// Sets the centre to `base + k0·P` with k256.
    fn recompute_centre(&mut self) {
        let centre = self.base + self.table.generator * *self.k0;
        self.degenerate = bool::from(centre.is_identity());
        (self.cx, self.cy) = affine_xy(&centre);
    }

    /// Skips the current batch: advances the centre with k256 instead.
    fn skip_batch(&mut self) {
        *self.k0 = self.k0.add(&Scalar::from(self.table.step()));
        self.recompute_centre();
    }

    /// Runs one batch, calling `visitor.visit(offset, x)` for `x(base + (k0 + offset)·P)`
    /// with `offset` in `-H..=H`, then moves to the next batch (`k0 += 2H+1`).
    /// Returns `false` if the batch was skipped because a visited point or the
    /// centre is the point at infinity (zero difference); `k0` still advances,
    /// so the caller can redo that batch with k256 if it matters.
    #[inline]
    pub fn batch<V: Visit>(&mut self, mut visitor: V) -> bool {
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
        if !batch_invert(&self.dx, &mut self.inv) {
            // C == ±j·P for some j in the table (probability ~2^-247 for a
            // random start; a real case for the giant steps of `recover`):
            // jump via k256 instead and skip this batch.
            self.skip_batch();
            return false;
        }
        visitor.visit(0, &cx);
        // ty + cy as ty − (−cy): a subtraction compiles shorter than an addition.
        let neg_cy = cy.neg();
        for (j, ((tx, ty), inv)) in xs.iter().zip(ys).zip(&self.inv[..h]).enumerate() {
            let offset = j as i64 + 1;
            // x(C ± T) = λ² − cx − tx with λ = (±ty − cy) / (tx − cx); only λ²
            // is needed, so the slope of C − T is taken as (ty + cy) / dx.
            let sum = cx.add(tx);
            let lambda_plus = ty.sub(&cy).mul(inv);
            let lambda_minus = ty.sub(&neg_cy).mul(inv);
            let x_plus = lambda_plus.square().sub(&sum);
            let x_minus = lambda_minus.square().sub(&sum);
            visitor.visit_pair(offset, &x_plus, &x_minus);
        }
        // Jump: C += (2H+1)·P.
        let lambda = table.jump_y.sub(&cy).mul(&self.inv[h]);
        let nx = lambda.square().sub(&cx).sub(&table.jump_x);
        let ny = lambda.mul(&cx.sub(&nx)).sub(&cy);
        self.cx = nx;
        self.cy = ny;
        *self.k0 = self.k0.add(&Scalar::from(table.step()));
        true
    }

    /// One batch of pattern matching; pushes hits into `out`.
    #[inline]
    fn search_batch<'p, K: Keys>(
        &mut self,
        keys: K,
        patterns: &'p PatternSet,
        out: &mut Vec<Candidate<'p>>,
    ) -> bool {
        self.batch(Searcher {
            keys,
            patterns,
            out,
            k0: self.k0.clone(),
        })
    }
}

/// The top-limb `(mask, value)` keys of the patterns, as seen by the batch
/// loop. Implemented for fixed-size arrays so that the common pattern counts
/// are monomorphised into straight-line compares (no length check, no loop),
/// and for a slice as the fallback for any count.
pub trait Keys: Copy {
    /// Does `top` (the top limb of an x coordinate) match any pattern?
    fn hit(&self, top: u64) -> bool;
}

impl<const N: usize> Keys for [(u64, u64); N] {
    #[inline(always)]
    fn hit(&self, top: u64) -> bool {
        self.iter().any(|&(mask, value)| top & mask == value)
    }
}

impl Keys for &[(u64, u64)] {
    #[inline(always)]
    fn hit(&self, top: u64) -> bool {
        self.iter().any(|&(mask, value)| top & mask == value)
    }
}

/// A pattern hit before scalar reconstruction. No `Debug`: `k0` is the
/// secret minus a public offset in random mode.
struct Candidate<'p> {
    /// Batch centre scalar.
    k0: Zeroizing<Scalar>,
    /// `k = k0 + offset` before the endomorphism.
    offset: i64,
    /// Power of λ applied (0, 1 or 2).
    endo: u8,
    /// x coordinate of `λ^endo · (k0 + offset) · G`.
    x: Fe,
    pattern: &'p Pattern,
}

/// Tests `x`, `β·x` and `β²·x` of every visited point against the patterns.
struct Searcher<'p, 'o, K: Keys> {
    /// Top-limb (mask, value) of every pattern, held by value so the check
    /// is a compare per pattern and nothing is reloaded through `PatternSet`.
    keys: K,
    patterns: &'p PatternSet,
    out: &'o mut Vec<Candidate<'p>>,
    k0: Zeroizing<Scalar>,
}

impl<K: Keys> Visit for Searcher<'_, '_, K> {
    #[inline(always)]
    fn visit(&mut self, offset: i64, x: &Fe) {
        let (bx, b2x_top) = endomorphisms(x);
        let hits = self.hits(x, &bx, b2x_top);
        if hits != 0 {
            self.push_hits(hits, offset, x, &bx);
        }
    }

    /// Both points' arithmetic and top-limb tests before the single branch:
    /// the loop body stays one basic block, which the compiler can schedule
    /// as a whole (the plus and minus chains are independent).
    #[inline(always)]
    fn visit_pair(&mut self, offset: i64, x_plus: &Fe, x_minus: &Fe) {
        let (bx_plus, b2x_top_plus) = endomorphisms(x_plus);
        let (bx_minus, b2x_top_minus) = endomorphisms(x_minus);
        let hits_plus = self.hits(x_plus, &bx_plus, b2x_top_plus);
        let hits_minus = self.hits(x_minus, &bx_minus, b2x_top_minus);
        if hits_plus | hits_minus != 0 {
            self.push_hits(hits_plus, offset, x_plus, &bx_plus);
            self.push_hits(hits_minus, -offset, x_minus, &bx_minus);
        }
    }
}

/// `β·x` and the top limb of `β²·x`. Since β² + β + 1 = 0,
/// β²·x = −(x + β·x): an add and a negation instead of a second
/// multiplication, and the hot loop only needs the top limb of the negation
/// (the full value is recomputed by [`endomorphism`] on a hit). `x + β·x` is
/// never zero: it would mean x = 0, which is not on the curve.
#[inline(always)]
fn endomorphisms(x: &Fe) -> (Fe, u64) {
    let bx = Fe::BETA.mul(x);
    (bx, bx.add(x).neg_top_limb())
}

/// `λ^endo` applied to `x` given `x` and `β·x`.
fn endomorphism(x: &Fe, bx: &Fe, endo: u8) -> Fe {
    match endo {
        0 => *x,
        1 => *bx,
        _ => bx.add(x).neg(),
    }
}

impl<'p, K: Keys> Searcher<'p, '_, K> {
    /// Top-limb test of the three candidates of one point, as a bit mask
    /// (bit `e` set when `λ^e·x` matched a key): branch-free.
    #[inline(always)]
    fn hits(&self, x: &Fe, bx: &Fe, b2x_top: u64) -> u8 {
        u8::from(self.keys.hit(x.top_limb()))
            | u8::from(self.keys.hit(bx.top_limb())) << 1
            | u8::from(self.keys.hit(b2x_top)) << 2
    }

    /// Full check of the candidates flagged in `hits`.
    #[cold]
    #[inline(never)]
    fn push_hits(&mut self, hits: u8, offset: i64, x: &Fe, bx: &Fe) {
        for endo in 0..3u8 {
            if hits >> endo & 1 != 0 {
                push_candidate(self.out, self.patterns, &self.k0, offset, endo, x, bx);
            }
        }
    }
}

/// Full check of a candidate whose top limb matched some pattern.
#[cold]
#[inline(never)]
fn push_candidate<'p>(
    out: &mut Vec<Candidate<'p>>,
    patterns: &'p PatternSet,
    k0: &Scalar,
    offset: i64,
    endo: u8,
    x: &Fe,
    bx: &Fe,
) {
    let x = endomorphism(x, bx, endo);
    if let Some(pattern) = patterns.find(&x) {
        out.push(Candidate {
            k0: Zeroizing::new(*k0),
            offset,
            endo,
            x,
            pattern,
        });
    }
}

/// How candidate scalars map to points: `k·G` (random start) or
/// `D + t·G` (split-key mode, `t` small).
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Random,
    Split { base: ProjectivePoint },
}

/// What a search produces: the scan secret key itself (random mode) or the
/// base key and the public tweak that turns it into the scan key (split-key
/// mode). No `Debug`: the secret must not reach logs or error text.
pub enum Key {
    Secret(Zeroizing<Scalar>),
    Tweak { base: ProjectivePoint, tweak: Tweak },
}

impl Key {
    /// The key of `−P` given the key of `P`.
    fn negated(self) -> Key {
        match self {
            Key::Secret(k) => Key::Secret(Zeroizing::new(k.negate())),
            Key::Tweak { base, tweak } => Key::Tweak {
                base,
                tweak: Tweak {
                    negate: !tweak.negate,
                    ..tweak
                },
            },
        }
    }
}

/// A verified vanity scan key (no `Debug`: it may carry the secret).
pub struct Found {
    pub key: Key,
    pub pubkey: [u8; 33],
}

/// Reconstructs the scalar of a candidate, checks its x against k256, and
/// fixes the y parity when the pattern requires one.
fn resolve(candidate: &Candidate, mode: &Mode) -> Result<Found, String> {
    let magnitude = Scalar::from(candidate.offset.unsigned_abs());
    let k = if candidate.offset >= 0 {
        candidate.k0.add(&magnitude)
    } else {
        candidate.k0.sub(&magnitude)
    };
    let (point, key) = match mode {
        Mode::Random => {
            let k = k.mul(&lambda_pow(candidate.endo));
            (
                ProjectivePoint::GENERATOR * k,
                Key::Secret(Zeroizing::new(k)),
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
            (tweak.apply_point(base), Key::Tweak { base: *base, tweak })
        }
    };
    if affine_xy(&point).0 != candidate.x {
        return Err("internal: reconstructed key does not reproduce the candidate x".to_string());
    }
    let pubkey = compressed(&point);
    // The walk never looks at y: negate when the pattern fixes the other parity.
    match candidate.pattern.parity {
        Some(want_odd) if (pubkey[0] == 0x03) != want_odd => Ok(Found {
            key: key.negated(),
            pubkey: compressed(&(-point)),
        }),
        _ => Ok(Found { key, pubkey }),
    }
}

/// The scalar as a `u64` if it fits.
fn scalar_to_u64(k: &Scalar) -> Option<u64> {
    let bytes: [u8; 32] = k.to_bytes().into();
    if bytes[..24].iter().any(|&b| b != 0) {
        return None;
    }
    let mut low = [0u8; 8];
    low.copy_from_slice(&bytes[24..]);
    Some(u64::from_be_bytes(low))
}

/// Start centre of split-mode range `range`: `range·2^RANGE_BITS + H`, so its
/// first batch visits `range·2^RANGE_BITS ..= range·2^RANGE_BITS + 2H`.
fn split_start(range: usize, half: usize) -> u64 {
    ((range as u64) << RANGE_BITS) + half as u64
}

/// Number of batches a split-mode walk may run before its highest visited
/// offset (`k0 + H`) would leave its range.
fn split_batches(half: usize) -> u64 {
    ((1u64 << RANGE_BITS) - 2 * half as u64) / (2 * half as u64 + 1)
}

/// x candidates a full split-mode search visits (all ranges, three per point).
pub fn split_coverage(half: usize) -> f64 {
    let per_range = split_batches(half) as f64 * (2 * half + 1) as f64;
    3.0 * per_range * SPLIT_RANGES as f64
}

/// A counter on its own cache line (128 bytes on Apple silicon, 64 elsewhere):
/// one per worker, written by that worker only, so the per-batch accounting
/// never writes a line another core is reading. Every other atomic the
/// workers touch is read-only in steady state (`stop`) or rare (`next_range`,
/// once per `2^44` offsets), so the batch loop has no cross-core traffic.
#[repr(align(128))]
#[derive(Default)]
pub struct Counter(AtomicU64);

impl Counter {
    #[inline]
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// State shared by the worker threads of one search.
pub struct Shared {
    pub stop: AtomicBool,
    /// x candidates tested so far, one slot per worker.
    tested: Vec<Counter>,
    /// Split mode: next range to hand out.
    next_range: AtomicUsize,
}

impl Shared {
    pub fn new(threads: usize) -> Shared {
        Shared {
            stop: AtomicBool::new(false),
            tested: (0..threads).map(|_| Counter::default()).collect(),
            next_range: AtomicUsize::new(0),
        }
    }

    /// x candidates tested so far by all workers.
    pub fn tested(&self) -> u64 {
        self.tested.iter().map(Counter::get).sum()
    }
}

/// Body of one worker thread. Every hit (or error) goes to `sender`; the
/// worker returns when `shared.stop` is set, the receiver is gone or (split
/// mode) every range has been taken.
///
/// Random mode walks from a random start and, after a hit, reports only that
/// hit and reseeds, so two keys from one run are never related by a small
/// public offset (which would let anyone prove common ownership and turn one
/// secret into the other). Split mode takes ranges `[i·2^44, (i+1)·2^44)` from
/// the shared queue; a range starts at `i·2^44 + H` so its first batch visits
/// `i·2^44 ..= i·2^44 + 2H`, and stops before a batch would cross the range end.
pub fn worker(
    index: usize,
    table: &Table,
    patterns: &PatternSet,
    mode: &Mode,
    shared: &Shared,
    sender: &mpsc::Sender<Result<Found, String>>,
) {
    let tested = &shared.tested[index];
    // Monomorphise the batch loop on the pattern count (see `Keys`).
    match patterns.keys.as_slice() {
        &[a] => run(table, patterns, [a], mode, shared, tested, sender),
        &[a, b] => run(table, patterns, [a, b], mode, shared, tested, sender),
        &[a, b, c] => run(table, patterns, [a, b, c], mode, shared, tested, sender),
        &[a, b, c, d] => run(table, patterns, [a, b, c, d], mode, shared, tested, sender),
        keys => run(table, patterns, keys, mode, shared, tested, sender),
    }
}

/// [`worker`] for one `Keys` form; `tested` is this worker's own counter.
fn run<K: Keys>(
    table: &Table,
    patterns: &PatternSet,
    keys: K,
    mode: &Mode,
    shared: &Shared,
    tested: &Counter,
    sender: &mpsc::Sender<Result<Found, String>>,
) {
    let mut hits = Vec::new();
    match mode {
        Mode::Random => {
            let mut walk = match Walk::random(table) {
                Ok(walk) => walk,
                Err(message) => {
                    let _ = sender.send(Err(message));
                    return;
                }
            };
            while !shared.stop.load(Ordering::Relaxed) {
                if walk.search_batch(keys, patterns, &mut hits) {
                    tested.add(table.candidates_per_batch());
                }
                if let Some(candidate) = hits.first() {
                    let resolved = resolve(candidate, mode);
                    hits.clear();
                    if sender.send(resolved).is_err() {
                        return;
                    }
                    if let Err(message) = walk.reseed() {
                        let _ = sender.send(Err(message));
                        return;
                    }
                }
            }
        }
        Mode::Split { base } => loop {
            let range = shared.next_range.fetch_add(1, Ordering::Relaxed);
            if range >= SPLIT_RANGES || shared.stop.load(Ordering::Relaxed) {
                return;
            }
            let k0 = Scalar::from(split_start(range, table.half));
            let mut walk = Walk::new(table, k0, *base);
            for _ in 0..split_batches(table.half) {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if walk.search_batch(keys, patterns, &mut hits) {
                    tested.add(table.candidates_per_batch());
                }
                for candidate in hits.drain(..) {
                    if sender.send(resolve(&candidate, mode)).is_err() {
                        return;
                    }
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::{self, Network};
    use std::thread;

    fn x_of(k: &Scalar) -> Fe {
        affine_xy(&(ProjectivePoint::GENERATOR * k)).0
    }

    fn g_table(half: usize) -> Table {
        Table::new(half, ProjectivePoint::GENERATOR)
    }

    fn g_walk(table: &Table, k0: u64) -> Walk<'_> {
        Walk::new(table, Scalar::from(k0), ProjectivePoint::IDENTITY)
    }

    fn secret_of(found: &Found) -> Scalar {
        match &found.key {
            Key::Secret(k) => **k,
            Key::Tweak { .. } => panic!("expected a secret"),
        }
    }

    /// Scalar of `k0 + offset`.
    fn shifted(k0: &Scalar, offset: i64) -> Scalar {
        let magnitude = Scalar::from(offset.unsigned_abs());
        if offset >= 0 {
            k0.add(&magnitude)
        } else {
            k0.sub(&magnitude)
        }
    }

    #[test]
    fn batch_invert_matches_individual_inverts() {
        let values: Vec<Fe> = (0..37).map(|_| x_of(&random_scalar().unwrap())).collect();
        let mut out = vec![Fe::ZERO; values.len()];
        assert!(batch_invert(&values, &mut out));
        for (v, inv) in values.iter().zip(&out) {
            assert_eq!(*inv, v.invert());
            assert_eq!(v.mul(inv), Fe::ONE);
        }
        let mut with_zero = values.clone();
        with_zero[5] = Fe::ZERO;
        assert!(!batch_invert(&with_zero, &mut out));
        assert!(batch_invert(&[], &mut []));
        // Lengths around the lane count, and a zero in every lane position.
        for n in 1..=2 * LANES + 1 {
            let values = &values[..n];
            let mut out = vec![Fe::ZERO; n];
            assert!(batch_invert(values, &mut out), "n = {n}");
            for (v, inv) in values.iter().zip(&out) {
                assert_eq!(v.mul(inv), Fe::ONE, "n = {n}");
            }
            for zero_at in 0..n {
                let mut with_zero = values.to_vec();
                with_zero[zero_at] = Fe::ZERO;
                assert!(!batch_invert(&with_zero, &mut out));
            }
        }
    }

    #[test]
    fn batch_x_coordinates_match_k256() {
        let table = g_table(256);
        let k0 = random_scalar().unwrap();
        let mut walk = Walk::new(&table, k0, ProjectivePoint::IDENTITY);
        let mut seen = vec![None; 2 * table.half + 1];
        assert!(walk.batch(|offset: i64, x: &Fe| {
            let idx = (offset + table.half as i64) as usize;
            assert!(seen[idx].is_none(), "offset {offset} visited twice");
            seen[idx] = Some(*x);
        }));
        for (idx, x) in seen.iter().enumerate() {
            let offset = idx as i64 - table.half as i64;
            assert_eq!(x.unwrap(), x_of(&shifted(&k0, offset)), "offset {offset}");
        }
        // After the batch the centre moved by 2H+1.
        let next = k0.add(&Scalar::from(table.step()));
        assert_eq!(*walk.k0, next);
        assert_eq!(
            affine_xy(&(ProjectivePoint::GENERATOR * next)),
            (walk.cx, walk.cy)
        );
        // And a second batch is consistent too.
        let mut count = 0;
        assert!(walk.batch(|offset: i64, x: &Fe| {
            count += 1;
            assert_eq!(*x, x_of(&shifted(&next, offset)));
        }));
        assert_eq!(count, 2 * table.half + 1);
    }

    /// Generic generator and base: the walk visits `base + (k0 + offset)·P`.
    #[test]
    fn batch_with_base_and_generator() {
        let m = random_scalar().unwrap();
        let generator = ProjectivePoint::GENERATOR * m;
        let table = Table::new(16, generator);
        let base = ProjectivePoint::GENERATOR * random_scalar().unwrap();
        let mut walk = Walk::new(&table, Scalar::from(1000u64), base);
        for batch in 0..3u64 {
            let k0 = 1000 + batch * table.step();
            let mut count = 0;
            assert!(walk.batch(|offset: i64, x: &Fe| {
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
        let mut walk = g_walk(&table, 8);
        assert!(!walk.batch(|_: i64, _: &Fe| panic!("visited a degenerate batch")));
        assert_eq!(*walk.k0, Scalar::from(8 + 17u64));
        assert!(walk.batch(|offset: i64, x: &Fe| {
            assert_eq!(*x, x_of(&Scalar::from((25 + offset) as u64)));
        }));
        // Centre at infinity: base = −k0·G.
        let base = -(ProjectivePoint::GENERATOR * Scalar::from(100u64));
        let mut walk = Walk::new(&table, Scalar::from(100u64), base);
        assert!(walk.degenerate);
        assert!(!walk.batch(|_: i64, _: &Fe| panic!("visited a degenerate batch")));
        assert!(!walk.degenerate);
        // The next centre is the jump point (2H+1)·G itself: skipped too.
        assert!(!walk.batch(|_: i64, _: &Fe| panic!("visited a degenerate batch")));
        assert!(walk.batch(|offset: i64, x: &Fe| {
            assert_eq!(*x, x_of(&Scalar::from((34 + offset) as u64)));
        }));
        // A visited point at infinity (offset ≠ 0): base = −(k0 + 3)·G.
        let base = -(ProjectivePoint::GENERATOR * Scalar::from(103u64));
        let mut walk = Walk::new(&table, Scalar::from(100u64), base);
        assert!(!walk.batch(|_: i64, _: &Fe| panic!("visited a degenerate batch")));
        assert!(walk.batch(|offset: i64, x: &Fe| {
            assert_eq!(*x, x_of(&Scalar::from((14 + offset) as u64)));
        }));
    }

    /// `reseed` moves the walk to an unrelated place: the new start is not
    /// within ±2^64 of the old one, and the walk is consistent from there.
    #[test]
    fn reseed_moves_far() {
        let table = g_table(8);
        let mut walk = g_walk(&table, 1000);
        assert!(walk.batch(|offset: i64, x: &Fe| {
            assert_eq!(*x, x_of(&Scalar::from((1000 + offset) as u64)));
        }));
        let before = *walk.k0;
        walk.reseed().unwrap();
        let after = *walk.k0;
        assert!(scalar_to_u64(&after.sub(&before)).is_none());
        assert!(scalar_to_u64(&before.sub(&after)).is_none());
        assert!(walk.batch(|offset: i64, x: &Fe| {
            assert_eq!(*x, x_of(&shifted(&after, offset)));
        }));
    }

    #[test]
    fn endomorphism_scalar_relation() {
        for _ in 0..8 {
            let k = random_scalar().unwrap();
            let x = x_of(&k);
            let lk = k.mul(&lambda_pow(1));
            let l2k = k.mul(&lambda_pow(2));
            assert_eq!(x_of(&lk), Fe::BETA.mul(&x));
            assert_eq!(x_of(&l2k), Fe::BETA2.mul(&x));
            assert_eq!(x_of(&l2k.mul(&lambda_pow(1))), x);
            let (bx, b2x_top) = endomorphisms(&x);
            assert_eq!(bx, Fe::BETA.mul(&x));
            assert_eq!(b2x_top, Fe::BETA2.mul(&x).top_limb());
            assert_eq!(endomorphism(&x, &bx, 0), x);
            assert_eq!(endomorphism(&x, &bx, 1), bx);
            assert_eq!(endomorphism(&x, &bx, 2), Fe::BETA2.mul(&x));
        }
    }

    #[test]
    fn small_table_and_step() {
        let table = g_table(4);
        assert_eq!(table.step(), 9);
        assert_eq!(table.candidates_per_batch(), 27);
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
        assert_eq!(split_start(0, 1024), 1024);
        assert_eq!(split_start(3, 1024), 3 * (1 << 44) + 1024);
        // The last batch of a range stays inside it, and no full batch is left out.
        for half in [1024usize, 64, 8, 1] {
            let batches = split_batches(half);
            let step = 2 * half as u64 + 1;
            let last_k0 = split_start(0, half) + (batches - 1) * step;
            assert!(last_k0 + (half as u64) < (1u64 << RANGE_BITS));
            assert!(last_k0 + (half as u64) + step >= (1u64 << RANGE_BITS) - 2 * half as u64);
        }
        assert_eq!(
            split_coverage(1024),
            3.0 * SPLIT_RANGES as f64 * (split_batches(1024) * 2049) as f64
        );
    }

    /// End-to-end: search a short pattern, then independently recompute the
    /// address from the secret key with k256 + bech32 and decode it.
    #[test]
    fn search_and_verify_independently() {
        let table = g_table(64);
        let spend = [2u8; 33];
        for (input, network) in [
            ("sp1qq?q", Network::Mainnet),
            ("sp1qqgp", Network::Mainnet),
            ("sp1qqvz", Network::Mainnet),
            ("tsp1qq?r", Network::Testnet),
            ("tsp1qqwy", Network::Signet),
            ("sprt1qq?s", Network::Regtest),
            ("sprt1qqf9", Network::Regtest),
        ] {
            let patterns = PatternSet::parse(&[input.to_string()], network).unwrap();
            let mut walk = Walk::random(&table).unwrap();
            let mut hits = Vec::new();
            let mut batches = 0;
            while hits.is_empty() {
                walk.search_batch(patterns.keys.as_slice(), &patterns, &mut hits);
                batches += 1;
                assert!(batches < 10_000, "no hit for {input}");
            }
            for candidate in &hits {
                let found = resolve(candidate, &Mode::Random).unwrap();
                // Independent path: scalar → point → compressed → bech32m.
                let point = ProjectivePoint::GENERATOR * secret_of(&found);
                let pubkey = compressed(&point);
                assert_eq!(pubkey, found.pubkey);
                let addr = address::encode(network.hrp(), &pubkey, &spend);
                assert!(
                    patterns.patterns[0].matches_address(&addr),
                    "{input}: {addr}"
                );
                let (got_network, scan, got_spend) = address::decode(&addr).unwrap();
                // Signet shares testnet's hrp and decodes as testnet.
                assert_eq!(got_network.hrp(), network.hrp());
                assert_eq!(scan, pubkey);
                assert_eq!(got_spend, spend);
            }
        }
    }

    /// Split-key mode with two walks on the first two ranges: every tweak maps
    /// the base secret to the found key and stays inside its walk's range.
    #[test]
    fn split_mode_tweaks_reproduce_keys() {
        let table = g_table(64);
        let d = random_scalar().unwrap();
        let base = ProjectivePoint::GENERATOR * d;
        let mode = Mode::Split { base };
        for input in ["sp1qq?q", "sp1qqgp", "sp1qqvz"] {
            let patterns = PatternSet::parse(&[input.to_string()], Network::Mainnet).unwrap();
            let mut walks: Vec<Walk> = (0..2)
                .map(|i| Walk::new(&table, Scalar::from(split_start(i, table.half)), base))
                .collect();
            let mut hits = Vec::new();
            let mut batches = 0;
            while hits.is_empty() {
                for walk in &mut walks {
                    walk.search_batch(patterns.keys.as_slice(), &patterns, &mut hits);
                }
                batches += 1;
                assert!(batches < 10_000, "no hit for {input}");
            }
            for candidate in &hits {
                let found = resolve(candidate, &mode).unwrap();
                let Key::Tweak {
                    base: found_base,
                    tweak,
                } = found.key
                else {
                    panic!("expected a tweak");
                };
                assert_eq!(found_base, base);
                assert!(tweak.t >> RANGE_BITS < 2, "{tweak}");
                let priv_key = tweak.apply(&d);
                assert_eq!(
                    compressed(&(ProjectivePoint::GENERATOR * priv_key)),
                    found.pubkey
                );
                assert_eq!(compressed(&tweak.apply_point(&base)), found.pubkey);
                let pattern = &patterns.patterns[0];
                if let Some(odd) = pattern.parity {
                    assert_eq!(found.pubkey[0] == 0x03, odd);
                }
                let addr = address::encode(Network::Mainnet.hrp(), &found.pubkey, &[2u8; 33]);
                assert!(pattern.matches_address(&addr), "{input}: {addr}");
                // Re-parse of the printed form gives the same tweak.
                let reparsed: Tweak = tweak.to_string().parse().unwrap();
                assert_eq!(reparsed, tweak);
            }
        }
    }

    #[test]
    fn resolve_rejects_wrong_x() {
        let table = g_table(4);
        let patterns = PatternSet::parse(&["sp1qq".to_string()], Network::Mainnet).unwrap();
        let walk = Walk::random(&table).unwrap();
        let candidate = Candidate {
            k0: walk.k0.clone(),
            offset: 1,
            endo: 0,
            x: Fe::ONE,
            pattern: &patterns.patterns[0],
        };
        assert!(resolve(&candidate, &Mode::Random).is_err());
        let base = ProjectivePoint::GENERATOR * random_scalar().unwrap();
        assert!(resolve(&candidate, &Mode::Split { base }).is_err());
        // Split mode rejects offsets that do not fit the tweak range.
        let big = Candidate {
            k0: Zeroizing::new(Scalar::from(1u64 << 52)),
            offset: 0,
            endo: 0,
            x: affine_xy(&(base + ProjectivePoint::GENERATOR * Scalar::from(1u64 << 52))).0,
            pattern: &patterns.patterns[0],
        };
        let Err(err) = resolve(&big, &Mode::Split { base }) else {
            panic!("expected an out-of-range error");
        };
        assert!(err.contains("out of range"), "{err}");
    }

    /// Split mode with every range already taken: the worker exits without
    /// reporting anything, which is what makes the main loop fail cleanly.
    #[test]
    fn split_worker_exits_when_ranges_are_exhausted() {
        let table = g_table(4);
        let patterns = PatternSet::parse(&["sp1qq".to_string()], Network::Mainnet).unwrap();
        let mode = Mode::Split {
            base: ProjectivePoint::GENERATOR,
        };
        let shared = Shared::new(1);
        shared.next_range.store(SPLIT_RANGES, Ordering::Relaxed);
        let (sender, receiver) = mpsc::channel();
        worker(0, &table, &patterns, &mode, &shared, &sender);
        drop(sender);
        assert!(receiver.recv().is_err());
    }

    /// Random mode reseeds after every hit: keys reported by one worker are
    /// not related by a small offset (in any of the six `±λ^e` variants), so
    /// an observer cannot link them and one secret does not yield another.
    #[test]
    fn random_hits_from_one_worker_are_unlinkable() {
        let table = g_table(64);
        let patterns = PatternSet::parse(&["sp1qq?q".to_string()], Network::Mainnet).unwrap();
        let shared = Shared::new(1);
        let (sender, receiver) = mpsc::channel();
        let mut secrets = Vec::new();
        thread::scope(|scope| {
            scope.spawn(|| worker(0, &table, &patterns, &Mode::Random, &shared, &sender));
            for _ in 0..4 {
                let Ok(found) = receiver.recv().unwrap() else {
                    panic!("worker reported an error");
                };
                let secret = secret_of(&found);
                assert_eq!(
                    compressed(&(ProjectivePoint::GENERATOR * secret)),
                    found.pubkey
                );
                secrets.push(secret);
            }
            shared.stop.store(true, Ordering::Relaxed);
        });
        let far = |d: Scalar| scalar_to_u64(&d).is_none_or(|t| t >= 1 << 40);
        for (i, a) in secrets.iter().enumerate() {
            for b in &secrets[i + 1..] {
                for variant in Tweak::variants(0) {
                    let related = variant.apply(a);
                    assert!(far(b.sub(&related)) && far(related.sub(b)), "linked keys");
                }
            }
        }
    }
}
