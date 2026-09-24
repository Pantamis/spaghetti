//! Apple Metal GPU search (feature `gpu`, macOS only): the batched walk of
//! `search.rs` as a compute kernel (`gpu/search.metal`), one walk per GPU
//! thread, driven by a host thread that plays the part of a `search::worker`.
//! The host seeds the walks, dispatches a few batches at a time, resolves the
//! hits the kernel reports through the same k256 reconstruction and check as a
//! CPU hit (`search::resolve_hit`, so every x the GPU claims is re-derived),
//! and reseeds. The shader is compiled by the Metal framework at start-up; no
//! Xcode is needed.
//!
//! Secrets stay on the host: the GPU buffers hold only public points (the walk
//! centres, the table `j·G`), and the start scalars `k0[t]` live in a
//! zeroizing vector here. A thread's centre after `b` batches is
//! `base + (k0[t] + b·(2H+1))·G`, so a hit record needs only the thread, the
//! batch index within the dispatch, the signed offset and the endomorphism
//! power (plus the claimed x, which the host checks).
//!
//! `unsafe` is confined to reading and writing the shared-memory Metal buffers
//! between dispatches, see [`Engine::slice`] and [`Engine::slice_mut`].

#[cfg(not(target_os = "macos"))]
compile_error!("the `gpu` feature needs macOS: the search kernel is Metal");

use std::ffi::c_void;
use std::mem::size_of;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use k256::elliptic_curve::Group;
use k256::{ProjectivePoint, Scalar};
use metal::objc::rc::autoreleasepool;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use zeroize::Zeroizing;

use crate::field::Fe;
use crate::pattern::PatternSet;
use crate::search::{self, Found, Mode, RANGE_BITS, SPLIT_RANGES, Shared, Table, affine_xy};

const SOURCE: &str = include_str!("gpu/search.metal");

/// Default number of walks (GPU threads).
pub const DEFAULT_THREADS: usize = 1 << 15;
/// Default half-batch `H` of a GPU walk (2H points per inversion). Measured
/// on an M3 Pro: 32768 walks × H = 1024 is the plateau (H = 512 costs 2-3%,
/// 16384 walks 6%, 65536 walks gain nothing), about 1.1 GB of scratch.
pub const DEFAULT_HALF: usize = 1024;
/// Walk count bounds: a power of two so split-mode ranges divide evenly.
pub const MIN_THREADS: usize = 1 << 5;
pub const MAX_THREADS: usize = 1 << 20;
/// Half-batch bound: the table `(H+1)·64` bytes lives in the constant
/// address space.
pub const MAX_HALF: usize = 1 << 11;
/// Hit records the kernel can append per dispatch.
const HIT_CAPACITY: usize = 1 << 16;
/// Dispatch length the driver aims for: long enough to amortise the launch,
/// short enough to notice `stop` and hits promptly.
const TARGET_DISPATCH: Duration = Duration::from_millis(150);
const MAX_BATCHES_PER_DISPATCH: u64 = 256;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Walks, i.e. GPU threads; a power of two in `MIN_THREADS..=MAX_THREADS`.
    pub threads: usize,
    /// Half-batch `H` of every walk.
    pub half: usize,
}

impl Config {
    pub fn check(&self) -> Result<(), String> {
        if !self.threads.is_power_of_two() || !(MIN_THREADS..=MAX_THREADS).contains(&self.threads) {
            return Err(format!(
                "--gpu-threads: a power of two between {MIN_THREADS} and {MAX_THREADS}, got {}",
                self.threads
            ));
        }
        if !(1..=MAX_HALF).contains(&self.half) {
            return Err(format!(
                "--gpu-batch: H must be between 1 and {MAX_HALF}, got {}",
                self.half
            ));
        }
        // Split mode: every walk gets 2^44 / threads offsets and needs room
        // for at least one batch there.
        if (1u64 << RANGE_BITS) / self.threads as u64 <= 4 * self.half as u64 + 1 {
            return Err("--gpu-threads × --gpu-batch too large for split-key mode".to_string());
        }
        Ok(())
    }

    /// Bytes of GPU memory the scratch buffer takes.
    pub fn scratch_bytes(&self) -> u64 {
        self.threads as u64 * (self.half as u64 + 1) * 32
    }

    /// Offsets of one split-mode range handled by one walk.
    fn split_span(&self) -> u64 {
        (1u64 << RANGE_BITS) / self.threads as u64
    }

    /// Batches a walk may run inside its split-mode span before its highest
    /// visited offset (`k0 + H`) would leave it.
    fn split_batches(&self) -> u64 {
        (self.split_span() - 2 * self.half as u64) / (2 * self.half as u64 + 1)
    }
}

/// x candidates a full split-mode GPU search visits (every range, three per
/// point).
pub fn split_coverage(cfg: &Config) -> f64 {
    let per_walk = cfg.split_batches() as f64 * (2 * cfg.half + 1) as f64;
    3.0 * per_walk * cfg.threads as f64 * SPLIT_RANGES as f64
}

/// Name of the default Metal device, for the start-up banner.
pub fn device_name() -> Result<String, String> {
    Device::system_default()
        .map(|d| d.name().to_string())
        .ok_or_else(|| "no Metal device".to_string())
}

/// Mirrors `struct Params` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n: u32,
    half: u32,
    batches: u32,
    nkeys: u32,
    hit_cap: u32,
    first_only: u32,
}

/// Mirrors `struct Key` in the shader: top 64 bits of x as (limb 7, limb 6).
#[repr(C)]
#[derive(Clone, Copy)]
struct Key {
    mask_hi: u32,
    mask_lo: u32,
    value_hi: u32,
    value_lo: u32,
}

/// Mirrors `struct Hit` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Hit {
    walk: u32,
    batch: u32,
    offset: i32,
    endo: u32,
    x: [u32; 8],
}

/// A field element as the kernel stores it: 8 little-endian 32-bit limbs.
type Limbs = [u32; 8];

fn to_limbs(fe: &Fe) -> Limbs {
    let mut out = [0u32; 8];
    for (i, limb) in fe.limbs().iter().enumerate() {
        out[2 * i] = *limb as u32;
        out[2 * i + 1] = (*limb >> 32) as u32;
    }
    out
}

fn from_limbs(limbs: &Limbs) -> Option<Fe> {
    let mut out = [0u64; 4];
    for (i, limb) in out.iter_mut().enumerate() {
        *limb = u64::from(limbs[2 * i]) | u64::from(limbs[2 * i + 1]) << 32;
    }
    Fe::from_limbs(out)
}

/// The compiled kernel and its buffers.
struct Engine {
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    /// `n` entries of `(x, y)` limbs: the walk centres.
    centres: Buffer,
    /// `(H+1)` entries of `(x, y)` limbs: `j·G` for `j = 1..=H`, then the jump.
    table: Buffer,
    /// `(H+1)·n` field elements of prefix products.
    scratch: Buffer,
    keys: Buffer,
    hits: Buffer,
    /// `[hits appended, degenerate walks]`.
    counters: Buffer,
    /// Batches each walk may still run in the current dispatch.
    remaining: Buffer,
    /// 1 where a walk met a zero difference and needs its centre recomputed.
    flags: Buffer,
    n: usize,
    half: usize,
    nkeys: u32,
    threadgroup: u64,
}

impl Engine {
    fn new(cfg: &Config, table: &Table, patterns: &PatternSet) -> Result<Engine, String> {
        cfg.check()?;
        let device = Device::system_default().ok_or("no Metal device")?;
        let budget = device.recommended_max_working_set_size();
        if budget > 0 && cfg.scratch_bytes() > budget / 2 {
            return Err(format!(
                "GPU scratch would take {} MB, more than half of the device's {} MB; lower \
                 --gpu-threads or --gpu-batch",
                cfg.scratch_bytes() >> 20,
                budget >> 20
            ));
        }
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(|e| format!("Metal shader compilation failed: {e}"))?;
        let function = library
            .get_function("search", None)
            .map_err(|e| format!("Metal kernel not found: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Metal pipeline creation failed: {e}"))?;
        // Two SIMD groups per threadgroup measured best (32..256 tried).
        let width = pipeline.thread_execution_width().max(1);
        let threadgroup = (2 * width).min(pipeline.max_total_threads_per_threadgroup());

        let n = cfg.threads;
        let half = cfg.half;
        let shared = MTLResourceOptions::StorageModeShared;
        let mut points: Vec<Limbs> = Vec::with_capacity(2 * (half + 1));
        for j in 0..half {
            let (x, y) = table.point(j);
            points.push(to_limbs(&x));
            points.push(to_limbs(&y));
        }
        let (jx, jy) = table.jump();
        points.push(to_limbs(&jx));
        points.push(to_limbs(&jy));
        let keys: Vec<Key> = patterns
            .keys
            .iter()
            .map(|&(mask, value)| Key {
                mask_hi: (mask >> 32) as u32,
                mask_lo: mask as u32,
                value_hi: (value >> 32) as u32,
                value_lo: value as u32,
            })
            .collect();
        let engine = Engine {
            queue: device.new_command_queue(),
            pipeline,
            centres: device.new_buffer((n * size_of::<[Limbs; 2]>()) as u64, shared),
            table: buffer_with(&device, &points),
            scratch: device.new_buffer(cfg.scratch_bytes(), shared),
            keys: buffer_with(&device, &keys),
            hits: device.new_buffer((HIT_CAPACITY * size_of::<Hit>()) as u64, shared),
            counters: device.new_buffer((2 * size_of::<u32>()) as u64, shared),
            remaining: device.new_buffer((n * size_of::<u32>()) as u64, shared),
            flags: device.new_buffer((n * size_of::<u32>()) as u64, shared),
            n,
            half,
            nkeys: keys.len() as u32,
            threadgroup,
        };
        engine.slice_mut::<u32>(&engine.flags).fill(0);
        Ok(engine)
    }

    /// The contents of a shared buffer as a slice of `len` plain values.
    ///
    /// Sound because every buffer is `StorageModeShared` (host-visible),
    /// outlives the returned borrow (it is owned by `self`), holds at least
    /// `len` values of `T` by construction (the callers pass the length the
    /// buffer was created with) and is only accessed between dispatches, after
    /// `wait_until_completed`, so the GPU never writes it concurrently. `T` is
    /// a plain integer or `#[repr(C)]` aggregate of integers for which every
    /// bit pattern is valid, and Metal buffers are at least 16-byte aligned.
    fn slice<T: Copy>(&self, buffer: &Buffer) -> &[T] {
        let len = buffer.length() as usize / size_of::<T>();
        // SAFETY: see above.
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, len) }
    }

    #[allow(clippy::mut_from_ref)]
    fn slice_mut<T: Copy>(&self, buffer: &Buffer) -> &mut [T] {
        let len = buffer.length() as usize / size_of::<T>();
        // SAFETY: as for `slice`; callers never hold two slices of one buffer.
        unsafe { std::slice::from_raw_parts_mut(buffer.contents() as *mut T, len) }
    }

    /// Runs up to `batches` batches on every walk (bounded per walk by
    /// `remaining`), blocking until done. `first_only`: a walk stops after the
    /// batch of its first hit.
    fn dispatch(&self, batches: u32, first_only: bool) -> Result<(), String> {
        let params = Params {
            n: self.n as u32,
            half: self.half as u32,
            batches,
            nkeys: self.nkeys,
            hit_cap: HIT_CAPACITY as u32,
            first_only: u32::from(first_only),
        };
        self.slice_mut::<u32>(&self.counters).fill(0);
        autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&self.centres), 0);
            encoder.set_buffer(1, Some(&self.table), 0);
            encoder.set_buffer(2, Some(&self.scratch), 0);
            encoder.set_bytes(
                3,
                size_of::<Params>() as u64,
                (&params as *const Params).cast::<c_void>(),
            );
            encoder.set_buffer(4, Some(&self.keys), 0);
            encoder.set_buffer(5, Some(&self.hits), 0);
            encoder.set_buffer(6, Some(&self.counters), 0);
            encoder.set_buffer(7, Some(&self.remaining), 0);
            encoder.set_buffer(8, Some(&self.flags), 0);
            encoder.dispatch_threads(
                MTLSize::new(self.n as u64, 1, 1),
                MTLSize::new(self.threadgroup, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            match command.status() {
                MTLCommandBufferStatus::Completed => Ok(()),
                status => Err(format!("GPU command buffer failed: {status:?}")),
            }
        })
    }

    /// Hit records of the last dispatch (clamped to the buffer capacity) and
    /// the number of walks that flagged a zero difference.
    fn results(&self) -> (&[Hit], u32) {
        let counters = self.slice::<u32>(&self.counters);
        let count = (counters[0] as usize).min(HIT_CAPACITY);
        (&self.slice::<Hit>(&self.hits)[..count], counters[1])
    }

    fn set_centre(&self, t: usize, x: &Fe, y: &Fe) {
        let centres = self.slice_mut::<[Limbs; 2]>(&self.centres);
        centres[t] = [to_limbs(x), to_limbs(y)];
    }
}

fn buffer_with<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Affine centres `base + k·G` for many `k`, computed on all CPU cores; `None`
/// for the point at infinity.
fn centres_of(k0: &[Scalar], base: &ProjectivePoint) -> Vec<Option<(Fe, Fe)>> {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = k0.len().div_ceil(cores).max(1);
    let mut out = vec![None; k0.len()];
    thread::scope(|scope| {
        for (ks, outs) in k0.chunks(chunk).zip(out.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (k, o) in ks.iter().zip(outs.iter_mut()) {
                    let point = *base + ProjectivePoint::GENERATOR * k;
                    if !bool::from(point.is_identity()) {
                        *o = Some(affine_xy(&point));
                    }
                }
            });
        }
    });
    out
}

/// The GPU worker: the counterpart of `search::worker` for the GPU engine.
/// Every hit (or error) goes to `sender`; it returns when `shared.stop` is
/// set, the receiver is gone or (split mode) every range has been taken.
pub fn worker(
    index: usize,
    cfg: &Config,
    patterns: &PatternSet,
    mode: &Mode,
    shared: &Shared,
    sender: &mpsc::Sender<Result<Found, String>>,
) {
    if let Err(message) = run(index, cfg, patterns, mode, shared, sender) {
        let _ = sender.send(Err(message));
    }
}

/// Host-side state of the walks.
struct Walks {
    /// Start scalar of every walk's next batch (secret in random mode).
    k0: Zeroizing<Vec<Scalar>>,
    /// Batches every walk may still run (split mode: within its span).
    budget: Vec<u64>,
    step: u64,
}

impl Walks {
    /// Advances walk `t` by `batches` batches.
    fn advance(&mut self, t: usize, batches: u64) {
        self.k0[t] = self.k0[t].add(&Scalar::from(batches * self.step));
        self.budget[t] = self.budget[t].saturating_sub(batches);
    }
}

fn run(
    index: usize,
    cfg: &Config,
    patterns: &PatternSet,
    mode: &Mode,
    shared: &Shared,
    sender: &mpsc::Sender<Result<Found, String>>,
) -> Result<(), String> {
    let table = Table::new(cfg.half, ProjectivePoint::GENERATOR);
    let engine = Engine::new(cfg, &table, patterns)?;
    let tested = shared.counter(index);
    let n = cfg.threads;
    let base = match mode {
        Mode::Random => ProjectivePoint::IDENTITY,
        Mode::Split { base } => *base,
    };
    let mut walks = Walks {
        k0: Zeroizing::new(vec![Scalar::ZERO; n]),
        budget: vec![0; n],
        step: table.step(),
    };
    let mut batches: u64 = 4;
    let mut set = vec![0u32; n];
    let mut seen = vec![false; n];
    loop {
        if shared.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Seed, or reseed once every walk has exhausted its span.
        if walks.budget.iter().all(|&b| b == 0) {
            match mode {
                Mode::Random => {
                    for k in walks.k0.iter_mut() {
                        *k = search::random_scalar()?;
                    }
                    walks.budget.fill(u64::MAX);
                }
                Mode::Split { .. } => {
                    let range = shared.next_range();
                    if range >= SPLIT_RANGES {
                        return Ok(());
                    }
                    let start = (range as u64) << RANGE_BITS;
                    for (t, k) in walks.k0.iter_mut().enumerate() {
                        *k = Scalar::from(start + t as u64 * cfg.split_span() + cfg.half as u64);
                    }
                    walks.budget.fill(cfg.split_batches());
                }
            }
            seed_centres(&engine, &mut walks, &base, (0..n).collect(), mode)?;
        }
        {
            let remaining = engine.slice_mut::<u32>(&engine.remaining);
            for (t, r) in remaining.iter_mut().enumerate() {
                *r = walks.budget[t].min(batches) as u32;
                set[t] = *r;
            }
        }
        let started = Instant::now();
        engine.dispatch(batches as u32, matches!(mode, Mode::Random))?;
        let elapsed = started.elapsed();

        // Batches actually run, hits, degenerate walks.
        let ran: Vec<u64> = engine
            .slice::<u32>(&engine.remaining)
            .iter()
            .zip(&set)
            .map(|(&rem, &set)| u64::from(set - rem))
            .collect();
        let points = ran.iter().sum::<u64>() * table.step();
        tested.add(3 * points);
        let (hits, degenerate) = engine.results();
        let mut reseed = Vec::new();
        for hit in hits {
            let t = hit.walk as usize;
            if t >= n || u64::from(hit.batch) >= ran[t] {
                return Err("internal: GPU hit record out of range".to_string());
            }
            if matches!(mode, Mode::Random) {
                // One hit per walk per dispatch, like the CPU worker, then a
                // fresh random start for that walk.
                if seen[t] {
                    continue;
                }
                seen[t] = true;
                reseed.push(t);
            }
            let x = from_limbs(&hit.x).ok_or("internal: GPU hit x is not canonical")?;
            let k0 = walks.k0[t].add(&Scalar::from(u64::from(hit.batch) * table.step()));
            let resolved = search::resolve_hit(
                &k0,
                i64::from(hit.offset),
                hit.endo as u8,
                x,
                patterns,
                mode,
            );
            if let Some(resolved) = resolved
                && sender.send(resolved).is_err()
            {
                return Ok(());
            }
        }
        for (t, &b) in ran.iter().enumerate() {
            walks.advance(t, b);
        }
        for &t in &reseed {
            walks.k0[t] = search::random_scalar()?;
            seen[t] = false;
        }
        if degenerate > 0 {
            let flags = engine.slice_mut::<u32>(&engine.flags);
            for (t, flag) in flags.iter_mut().enumerate() {
                if *flag != 0 {
                    *flag = 0;
                    // Like `Walk::skip_batch`: the batch is skipped, not redone.
                    walks.advance(t, 1);
                    reseed.push(t);
                }
            }
        }
        if !reseed.is_empty() {
            seed_centres(&engine, &mut walks, &base, reseed, mode)?;
        }
        // Aim for `TARGET_DISPATCH` per dispatch.
        if elapsed < TARGET_DISPATCH / 2 {
            batches = (batches * 2).min(MAX_BATCHES_PER_DISPATCH);
        } else if elapsed > TARGET_DISPATCH * 2 {
            batches = (batches / 2).max(1);
        }
    }
}

/// Computes and uploads the centres of the given walks; a centre at infinity
/// (the batch formulas do not apply) is skipped like `Walk::skip_batch`
/// (random mode: a fresh start instead).
fn seed_centres(
    engine: &Engine,
    walks: &mut Walks,
    base: &ProjectivePoint,
    mut which: Vec<usize>,
    mode: &Mode,
) -> Result<(), String> {
    while !which.is_empty() {
        let k0: Vec<Scalar> = which.iter().map(|&t| walks.k0[t]).collect();
        let centres = centres_of(&k0, base);
        let mut again = Vec::new();
        for (&t, centre) in which.iter().zip(centres) {
            match centre {
                Some((x, y)) => engine.set_centre(t, &x, &y),
                None => {
                    match mode {
                        Mode::Random => walks.k0[t] = search::random_scalar()?,
                        Mode::Split { .. } => walks.advance(t, 1),
                    }
                    again.push(t);
                }
            }
        }
        which = again;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::Network;
    use crate::search::random_scalar;
    use std::collections::HashSet;

    fn x_of(k: &Scalar) -> Fe {
        affine_xy(&(ProjectivePoint::GENERATOR * k)).0
    }

    fn all_patterns() -> PatternSet {
        PatternSet::parse(&["sp1qq?".to_string()], Network::Mainnet).unwrap()
    }

    /// Every field operation of the shader against the CPU implementation,
    /// on edge cases and random elements.
    /// CI runners may have no GPU: such tests pass vacuously and say so.
    fn device_or_skip() -> Option<Device> {
        let device = Device::system_default();
        if device.is_none() {
            eprintln!("no Metal device: GPU test skipped");
        }
        device
    }

    #[test]
    fn field_ops_match_cpu() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let library = device
            .new_library_with_source(SOURCE, &CompileOptions::new())
            .expect("shader compiles");
        let function = library.get_function("field_test", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let p_minus_1 = Fe::ZERO.sub(&Fe::ONE);
        let mut cases: Vec<Fe> = vec![
            Fe::ONE,
            Fe::from_limbs([2, 0, 0, 0]).unwrap(),
            p_minus_1,
            p_minus_1.sub(&Fe::ONE),
            Fe::from_limbs([0, 0, 0, 1 << 63]).unwrap(),
            Fe::from_limbs([u64::MAX, u64::MAX, u64::MAX, 0]).unwrap(),
            Fe::from_limbs([0xFFFF_FFFE_FFFF_FC2E, u64::MAX, u64::MAX, u64::MAX - 1]).unwrap(),
            Fe::from_limbs([0xFFFF_FFFF, 0, 0, 0]).unwrap(),
            Fe::from_limbs([1 << 32, 0, 0, 0]).unwrap(),
            Fe::BETA,
            Fe::BETA2,
        ];
        for _ in 0..200 {
            cases.push(x_of(&random_scalar().unwrap()));
        }
        let mut pairs: Vec<(Fe, Fe)> = Vec::new();
        for a in &cases {
            for b in cases.iter().step_by(7) {
                pairs.push((*a, *b));
            }
            pairs.push((*a, *a));
        }
        let input: Vec<[Limbs; 2]> = pairs
            .iter()
            .map(|(a, b)| [to_limbs(a), to_limbs(b)])
            .collect();
        let count = pairs.len() as u32;
        let inb = buffer_with(&device, &input);
        let outb = device.new_buffer(
            (pairs.len() * 6 * size_of::<Limbs>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let queue = device.new_command_queue();
        autoreleasepool(|| {
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&inb), 0);
            encoder.set_buffer(1, Some(&outb), 0);
            encoder.set_bytes(2, 4, (&count as *const u32).cast::<c_void>());
            encoder.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(32, 1, 1));
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        });
        // SAFETY: shared buffer, GPU finished, sized for `pairs.len()` results.
        let out: &[[Limbs; 6]] =
            unsafe { std::slice::from_raw_parts(outb.contents() as *const _, pairs.len()) };
        for ((a, b), got) in pairs.iter().zip(out) {
            let got: Vec<Fe> = got
                .iter()
                .map(|l| from_limbs(l).expect("canonical"))
                .collect();
            assert_eq!(got[0], a.mul(b), "mul {a:?} {b:?}");
            assert_eq!(got[1], a.square(), "sqr {a:?}");
            assert_eq!(got[2], a.add(b), "add {a:?} {b:?}");
            assert_eq!(got[3], a.sub(b), "sub {a:?} {b:?}");
            assert_eq!(got[4], a.neg(), "neg {a:?}");
            assert_eq!(got[5], a.invert(), "inv {a:?}");
        }
    }

    /// With a match-everything pattern the kernel reports every visited x;
    /// each one must be `x(λ^e·(k0 + offset)·G)` and every offset of every
    /// batch must appear exactly once per endomorphism power.
    #[test]
    fn walk_visits_every_point_and_matches_k256() {
        let cfg = Config {
            threads: MIN_THREADS,
            half: 8,
        };
        let patterns = all_patterns();
        let table = Table::new(cfg.half, ProjectivePoint::GENERATOR);
        if device_or_skip().is_none() {
            return;
        }
        let engine = Engine::new(&cfg, &table, &patterns).expect("engine");
        let n = cfg.threads;
        let k0: Vec<Scalar> = (0..n).map(|_| random_scalar().unwrap()).collect();
        for (t, centre) in centres_of(&k0, &ProjectivePoint::IDENTITY)
            .into_iter()
            .enumerate()
        {
            let (x, y) = centre.unwrap();
            engine.set_centre(t, &x, &y);
        }
        let batches = 3u32;
        engine.slice_mut::<u32>(&engine.remaining).fill(batches);
        engine.dispatch(batches, false).unwrap();
        assert!(
            engine
                .slice::<u32>(&engine.remaining)
                .iter()
                .all(|&r| r == 0)
        );
        let (hits, degenerate) = engine.results();
        assert_eq!(degenerate, 0);
        let per_batch = 3 * table.step() as usize;
        assert_eq!(hits.len(), n * batches as usize * per_batch);
        let mut seen = HashSet::new();
        for hit in hits {
            let t = hit.walk as usize;
            let k0 = k0[t].add(&Scalar::from(u64::from(hit.batch) * table.step()));
            let x = from_limbs(&hit.x).expect("canonical");
            let found = search::resolve_hit(
                &k0,
                i64::from(hit.offset),
                hit.endo as u8,
                x,
                &patterns,
                &Mode::Random,
            )
            .expect("match-all pattern")
            .expect("k256 agrees with the GPU");
            assert_eq!(found.pubkey[1..], x.to_bytes_be());
            assert!(
                seen.insert((t, hit.batch, hit.offset, hit.endo)),
                "duplicate {hit:?}"
            );
        }
        // The centres moved by `batches` steps.
        let centres = engine.slice::<[Limbs; 2]>(&engine.centres);
        for (t, centre) in centres.iter().enumerate() {
            let k = k0[t].add(&Scalar::from(u64::from(batches) * table.step()));
            assert_eq!(from_limbs(&centre[0]).unwrap(), x_of(&k), "walk {t}");
        }
    }

    /// A centre equal to a table point is flagged, not silently corrupted.
    #[test]
    fn degenerate_centre_is_flagged() {
        let cfg = Config {
            threads: MIN_THREADS,
            half: 8,
        };
        let patterns = all_patterns();
        let table = Table::new(cfg.half, ProjectivePoint::GENERATOR);
        if device_or_skip().is_none() {
            return;
        }
        let engine = Engine::new(&cfg, &table, &patterns).expect("engine");
        // Walk 0 starts at 5·G (a table point); walk 1 at 17·G (the jump).
        let k0: Vec<Scalar> = (0..cfg.threads)
            .map(|t| match t {
                0 => Scalar::from(5u64),
                1 => Scalar::from(17u64),
                _ => random_scalar().unwrap(),
            })
            .collect();
        for (t, centre) in centres_of(&k0, &ProjectivePoint::IDENTITY)
            .into_iter()
            .enumerate()
        {
            let (x, y) = centre.unwrap();
            engine.set_centre(t, &x, &y);
        }
        engine.slice_mut::<u32>(&engine.remaining).fill(2);
        engine.dispatch(2, false).unwrap();
        let (hits, degenerate) = engine.results();
        assert_eq!(degenerate, 2);
        let flags = engine.slice::<u32>(&engine.flags);
        assert_eq!(&flags[..3], &[1, 1, 0]);
        let remaining = engine.slice::<u32>(&engine.remaining);
        assert_eq!(&remaining[..3], &[2, 2, 0]);
        assert!(hits.iter().all(|h| h.walk >= 2));
        assert_eq!(
            hits.len(),
            (cfg.threads - 2) * 2 * 3 * table.step() as usize
        );
    }

    /// The worker end to end, in random mode: a real (easy) pattern, matches
    /// verified against k256 by the resolver.
    #[test]
    fn random_mode_worker_finds_verified_matches() {
        let cfg = Config {
            threads: MIN_THREADS,
            half: 16,
        };
        let patterns = PatternSet::parse(&["sp1qqgq".to_string()], Network::Mainnet).unwrap();
        if device_or_skip().is_none() {
            return;
        }
        let shared = Shared::new(1);
        let (sender, receiver) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| worker(0, &cfg, &patterns, &Mode::Random, &shared, &sender));
            for _ in 0..3 {
                let found = receiver.recv().unwrap().expect("no error");
                assert_eq!(found.pubkey[0], 0x02, "sp1qqg fixes an even y");
                let addr =
                    crate::address::encode(Network::Mainnet.hrp(), &found.pubkey, &found.pubkey);
                assert!(patterns.patterns[0].matches_address(&addr), "{addr}");
            }
            shared.stop.store(true, Ordering::Relaxed);
        });
        assert!(shared.tested() > 0);
    }

    /// Split mode: every reported tweak reproduces the address from the base
    /// key, offsets stay below 2^52, and the walks stop at their spans.
    #[test]
    fn split_mode_worker_reports_valid_tweaks() {
        let cfg = Config {
            threads: MIN_THREADS,
            half: 16,
        };
        let patterns = PatternSet::parse(&["sp1qq?q".to_string()], Network::Mainnet).unwrap();
        let base = ProjectivePoint::GENERATOR * random_scalar().unwrap();
        let mode = Mode::Split { base };
        if device_or_skip().is_none() {
            return;
        }
        let shared = Shared::new(1);
        let (sender, receiver) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| worker(0, &cfg, &patterns, &mode, &shared, &sender));
            for _ in 0..5 {
                let found = receiver.recv().unwrap().expect("no error");
                let search::Key::Tweak { base: b, tweak } = found.key else {
                    panic!("split mode reports tweaks");
                };
                assert_eq!(b, base);
                assert!(tweak.t < 1 << crate::tweak::MAX_TWEAK_BITS);
                assert_eq!(search::compressed(&tweak.apply_point(&base)), found.pubkey);
            }
            shared.stop.store(true, Ordering::Relaxed);
        });
    }

    #[test]
    fn config_limits() {
        assert!(
            Config {
                threads: 1000,
                half: 8
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                threads: MIN_THREADS,
                half: 0
            }
            .check()
            .is_err()
        );
        assert!(
            Config {
                threads: MAX_THREADS,
                half: MAX_HALF
            }
            .check()
            .is_ok()
        );
        let cfg = Config {
            threads: DEFAULT_THREADS,
            half: DEFAULT_HALF,
        };
        assert!(cfg.check().is_ok());
        // Every walk's last visited offset stays inside its span.
        let last = cfg.half as u64
            + (cfg.split_batches() - 1) * (2 * cfg.half as u64 + 1)
            + cfg.half as u64;
        assert!(last < cfg.split_span());
        assert!(split_coverage(&cfg) > 0.9 * 3.0 * 2f64.powi(52));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::search::random_scalar;

    fn x_of(k: &Scalar) -> Fe {
        affine_xy(&(ProjectivePoint::GENERATOR * k)).0
    }

    /// `cargo test --release --features gpu bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn report() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let library = device
            .new_library_with_source(SOURCE, &CompileOptions::new())
            .expect("shader compiles");
        for name in ["search", "field_test", "bench"] {
            let function = library.get_function(name, None).unwrap();
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap();
            eprintln!(
                "{name}: max threads/threadgroup {} (lower = more registers), width {}",
                pipeline.max_total_threads_per_threadgroup(),
                pipeline.thread_execution_width()
            );
        }
        let function = library.get_function("bench", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let n: usize = 1 << 17;
        let iters: u32 = 2000;
        let mut data: Vec<[Limbs; 2]> = Vec::with_capacity(n);
        let a = x_of(&random_scalar().unwrap());
        let b = x_of(&random_scalar().unwrap());
        for _ in 0..n {
            data.push([to_limbs(&a), to_limbs(&b)]);
        }
        let buffer = buffer_with(&device, &data);
        let queue = device.new_command_queue();
        for (mode, name) in [
            (0u32, "fe_mul"),
            (1, "fe_sqr"),
            (2, "fe_add"),
            (3, "point test"),
        ] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = Instant::now();
                autoreleasepool(|| {
                    let command = queue.new_command_buffer();
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_buffer(0, Some(&buffer), 0);
                    encoder.set_bytes(1, 4, (&iters as *const u32).cast::<c_void>());
                    encoder.set_bytes(2, 4, (&mode as *const u32).cast::<c_void>());
                    encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(64, 1, 1));
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                });
                best = best.min(started.elapsed().as_secs_f64());
            }
            let ops = n as f64 * iters as f64;
            eprintln!("{name}: {:.2} G ops/s", ops / best / 1e9);
        }
    }
}
