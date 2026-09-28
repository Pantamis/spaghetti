//! NVIDIA CUDA GPU search (feature `cuda`): the batched walk of `search.rs`
//! as a CUDA kernel (`cuda/search.cu`), one walk per GPU thread, driven by
//! one host thread per device that plays the part of a `search::worker`. The
//! host seeds the walks, launches a few batches at a time, resolves the hits
//! the kernel reports through the same k256 reconstruction and check as a CPU
//! hit (`search::resolve_hit`, so every x the GPU claims is re-derived), and
//! reseeds. The kernel is compiled by NVRTC at start-up for the device found,
//! with the half-batch and the pattern keys as compile-time constants; the
//! driver and NVRTC libraries are loaded at run time (no CUDA toolkit is
//! needed to run, only to build: see the README).
//!
//! This is the Metal engine (`gpu.rs`) with the same host protocol, plus two
//! things a large NVIDIA machine needs: several devices (`Config::devices`,
//! each takes split-mode ranges from the shared queue on its own), and a
//! `shift` kernel that carries every walk from one split-mode range to the
//! next with one point addition on the GPU instead of one scalar
//! multiplication per walk on the host.
//!
//! Secrets stay on the host: the GPU buffers hold only public points (the walk
//! centres, the table `j·G`), and the start scalars `k0[t]` live in a
//! zeroizing vector here. A thread's centre after `b` batches is
//! `base + (k0[t] + b·(2H+1))·G`, so a hit record needs only the thread, the
//! batch index within the launch, the signed offset and the endomorphism
//! power (plus the claimed x, which the host checks).
//!
//! `unsafe` is confined to the two kernel launches ([`Engine::launch_search`],
//! [`Engine::launch_shift`]) and the test kernels: a launch takes untyped
//! arguments, which must match the kernel signature.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{CompileError, CompileOptions, Ptx, compile_ptx_with_opts};
use k256::elliptic_curve::Group;
use k256::elliptic_curve::point::{AffineCoordinates, BatchNormalize};
use k256::{ProjectivePoint, Scalar};
use zeroize::Zeroizing;

use crate::field::Fe;
use crate::pattern::PatternSet;
use crate::search::{self, Found, Mode, RANGE_BITS, SPLIT_RANGES, Shared, Table, affine_xy};

const SOURCE: &str = include_str!("cuda/search.cu");

/// Default number of walks (GPU threads) per device.
pub const DEFAULT_THREADS: usize = 1 << 16;
/// Default half-batch `H` of a GPU walk (2H points per inversion). On a
/// Tesla T4, H = 2048 measured 5% faster than 512 and 2% faster than 1024;
/// 1024 keeps the default scratch at 2 GB for smaller cards. `bench-gpu`
/// picks the value for the machine.
pub const DEFAULT_HALF: usize = 1024;
/// Default threads per block.
pub const DEFAULT_BLOCK: usize = 128;
/// Default minimum resident blocks per SM asked of the compiler
/// (`__launch_bounds__`); 1 leaves the register allocation free.
pub const DEFAULT_MIN_BLOCKS: usize = 1;
/// Walk count bounds (a multiple of 32, a warp).
pub const MIN_THREADS: usize = 1 << 5;
pub const MAX_THREADS: usize = 1 << 21;
pub const MAX_HALF: usize = 1 << 12;
/// Hit records the kernel can append per launch.
const HIT_CAPACITY: usize = 1 << 16;
/// Words of one hit record (`HIT_WORDS` in the kernel).
const HIT_WORDS: usize = 12;
/// Launch length the driver aims for: long enough to amortise the launch and
/// the host round trip, short enough to notice `stop` and hits promptly.
const TARGET_DISPATCH: Duration = Duration::from_millis(200);
const MAX_BATCHES_PER_DISPATCH: u64 = 1024;
/// Above this many centres to upload, one full copy beats per-walk copies.
const FULL_UPLOAD: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    /// Walks per device, i.e. GPU threads; a multiple of 32 in
    /// `MIN_THREADS..=MAX_THREADS`. Best a multiple of what the device holds
    /// at once, so that no launch ends on a partial wave (`bench-gpu`).
    pub threads: usize,
    /// Half-batch `H` of every walk.
    pub half: usize,
    /// Threads per block, a multiple of 32.
    pub block: usize,
    /// `__launch_bounds__` minimum blocks per SM (caps the registers).
    pub min_blocks: usize,
    /// CUDA device ordinals, one host thread each.
    pub devices: Vec<usize>,
}

impl Config {
    pub fn check(&self) -> Result<(), String> {
        if !self.threads.is_multiple_of(32) || !(MIN_THREADS..=MAX_THREADS).contains(&self.threads)
        {
            return Err(format!(
                "--gpu-threads: a multiple of 32 between {MIN_THREADS} and {MAX_THREADS}, got {}",
                self.threads
            ));
        }
        if !(1..=MAX_HALF).contains(&self.half) {
            return Err(format!(
                "--gpu-batch: H must be between 1 and {MAX_HALF}, got {}",
                self.half
            ));
        }
        if !self.block.is_multiple_of(32) || !(32..=1024).contains(&self.block) {
            return Err(format!(
                "--gpu-block: a multiple of 32 between 32 and 1024, got {}",
                self.block
            ));
        }
        if !(1..=32).contains(&self.min_blocks) {
            return Err(format!(
                "--gpu-min-blocks: between 1 and 32, got {}",
                self.min_blocks
            ));
        }
        if self.devices.is_empty() {
            return Err("--gpu-devices: no device".to_string());
        }
        let mut seen = self.devices.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != self.devices.len() {
            return Err("--gpu-devices: a device is listed twice".to_string());
        }
        // Split mode: every walk gets 2^RANGE_BITS / threads offsets and needs room
        // for at least one batch there.
        if (1u64 << RANGE_BITS) / self.threads as u64 <= 4 * self.half as u64 + 1 {
            return Err("--gpu-threads × --gpu-batch too large for split-key mode".to_string());
        }
        Ok(())
    }

    /// Bytes of GPU memory the prefix products take on each device.
    pub fn scratch_bytes(&self) -> u64 {
        self.threads as u64 * self.half as u64 * 32
    }

    /// Offsets of one split-mode range handled by one walk (the remainder of
    /// `2^36 / walks`, fewer offsets than walks, is left at the range end).
    fn split_span(&self) -> u64 {
        (1u64 << RANGE_BITS) / self.threads as u64
    }

    /// Batches a walk may run inside its split-mode span before its highest
    /// visited offset (`k0 + H`) would leave it.
    fn split_batches(&self) -> u64 {
        let full = (self.split_span() - 2 * self.half as u64) / (2 * self.half as u64 + 1);
        #[cfg(test)]
        let full = full.min(tests::BATCH_LIMIT.get());
        full
    }
}

/// Number of host workers the configuration runs: one per device.
pub fn engines(cfg: &Config) -> usize {
    cfg.devices.len()
}

/// x candidates a full split-mode GPU search visits (every range, three per
/// point).
pub fn split_coverage(cfg: &Config) -> f64 {
    let per_walk = cfg.split_batches() as f64 * (2 * cfg.half + 1) as f64;
    3.0 * per_walk * cfg.threads as f64 * SPLIT_RANGES as f64
}

/// Fraction of the offsets of a split-mode range the walks visit (the tail of
/// every walk's span shorter than a batch is skipped).
pub fn split_fill(cfg: &Config) -> f64 {
    (cfg.split_batches() * (2 * cfg.half as u64 + 1) * cfg.threads as u64) as f64
        / (1u64 << RANGE_BITS) as f64
}

/// CUDA devices visible to the process.
pub fn device_count() -> Result<usize, String> {
    // The driver library is loaded on first use; a machine without it panics
    // inside cudarc, which is turned into an error here.
    let count = std::panic::catch_unwind(CudaContext::device_count)
        .map_err(|_| "the CUDA driver library (libcuda) could not be loaded".to_string())?
        .map_err(|e| format!("CUDA: {e}"))?;
    Ok(count.max(0) as usize)
}

/// Every visible device, for the default `--gpu-devices`.
pub fn all_devices() -> Result<Vec<usize>, String> {
    let count = device_count()?;
    if count == 0 {
        return Err("no CUDA device".to_string());
    }
    Ok((0..count).collect())
}

/// A `--gpu-devices` list such as `0,1,3`, checked against the machine.
pub fn parse_devices(list: &str) -> Result<Vec<usize>, String> {
    let count = device_count()?;
    let mut devices = Vec::new();
    for item in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let d: usize = item
            .parse()
            .map_err(|_| format!("--gpu-devices: '{item}' is not a device number"))?;
        if d >= count {
            return Err(format!(
                "--gpu-devices: device {d} does not exist ({count} CUDA device(s): 0..{})",
                count.saturating_sub(1)
            ));
        }
        devices.push(d);
    }
    if devices.is_empty() {
        return Err("--gpu-devices: empty list".to_string());
    }
    Ok(devices)
}

/// Name, compute capability, SM count and memory of a device.
pub struct DeviceInfo {
    pub name: String,
    pub capability: (i32, i32),
    pub sms: usize,
    pub threads_per_sm: usize,
    pub total_mem: usize,
    pub free_mem: usize,
}

pub fn device_info(ordinal: usize) -> Result<DeviceInfo, String> {
    let ctx = context(ordinal)?;
    let attr = |a| {
        ctx.attribute(a)
            .map(|v| v.max(0) as usize)
            .map_err(cuda_err)
    };
    Ok(DeviceInfo {
        name: ctx.name().map_err(cuda_err)?,
        capability: ctx.compute_capability().map_err(cuda_err)?,
        sms: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        threads_per_sm: attr(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        )?,
        total_mem: ctx.total_mem().map_err(cuda_err)?,
        free_mem: ctx.mem_get_info().map_err(cuda_err)?.0,
    })
}

/// Names of the configured devices, for the start-up banner.
pub fn device_name(cfg: &Config) -> Result<String, String> {
    let mut names: Vec<(String, usize)> = Vec::new();
    for &d in &cfg.devices {
        let name = device_info(d)?.name;
        match names.iter_mut().find(|(n, _)| *n == name) {
            Some((_, count)) => *count += 1,
            None => names.push((name, 1)),
        }
    }
    Ok(names
        .iter()
        .map(|(name, count)| {
            if *count > 1 {
                format!("{count}× {name}")
            } else {
                name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" + "))
}

fn cuda_err(e: impl std::fmt::Display) -> String {
    format!("CUDA: {e}")
}

fn context(ordinal: usize) -> Result<Arc<CudaContext>, String> {
    std::panic::catch_unwind(|| CudaContext::new(ordinal))
        .map_err(|_| "the CUDA driver library (libcuda) could not be loaded".to_string())?
        .map_err(|e| format!("CUDA device {ordinal}: {e}"))
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

fn from_limbs(limbs: &[u32]) -> Option<Fe> {
    let mut out = [0u64; 4];
    for (i, limb) in out.iter_mut().enumerate() {
        *limb = u64::from(limbs[2 * i]) | u64::from(limbs[2 * i + 1]) << 32;
    }
    Fe::from_limbs(out)
}

/// `(x, y)` as the 16 words of a centre or table entry.
fn point_words(x: &Fe, y: &Fe) -> [u32; 16] {
    let mut out = [0u32; 16];
    out[..8].copy_from_slice(&to_limbs(x));
    out[8..].copy_from_slice(&to_limbs(y));
    out
}

/// One hit record of the kernel.
#[derive(Clone, Copy, Debug)]
struct Hit {
    walk: u32,
    batch: u32,
    offset: i32,
    endo: u32,
    x: Limbs,
}

impl Hit {
    fn parse(words: &[u32; HIT_WORDS]) -> Hit {
        let mut x = [0u32; 8];
        x.copy_from_slice(&words[4..12]);
        Hit {
            walk: words[0],
            batch: words[1],
            offset: words[2] as i32,
            endo: words[3],
            x,
        }
    }
}

/// The `#define`s the kernel source expects (see the head of `search.cu`).
fn prelude(half: usize, block: usize, min_blocks: usize, patterns: Option<&PatternSet>) -> String {
    let test = match patterns {
        Some(set) if !set.keys.is_empty() => set
            .keys
            .iter()
            .map(|&(mask, value)| {
                format!(
                    "((((hi) & 0x{:08x}u) == 0x{:08x}u) & (((lo) & 0x{:08x}u) == 0x{:08x}u))",
                    (mask >> 32) as u32,
                    (value >> 32) as u32,
                    mask as u32,
                    value as u32
                )
            })
            .collect::<Vec<_>>()
            .join(" | "),
        _ => "false".to_string(),
    };
    format!(
        "#define HALF {half}\n#define BLOCK {block}\n#define MINB {min_blocks}\n\
         #define KEY_TEST(hi, lo) ({test})\n"
    )
}

/// Compiles the kernel source with `prelude` for the device of `ctx` and
/// loads it. NVRTC emits PTX for the device's virtual architecture, which
/// the driver compiles for the actual GPU; when this NVRTC predates the
/// device, the newest architecture it knows is used instead.
fn compile(ctx: &Arc<CudaContext>, prelude: &str) -> Result<Arc<CudaModule>, String> {
    let (major, minor) = ctx.compute_capability().map_err(cuda_err)?;
    let device = major * 10 + minor;
    const KNOWN: [i32; 12] = [120, 100, 90, 89, 87, 86, 80, 75, 72, 70, 61, 60];
    let mut archs = vec![device];
    archs.extend(KNOWN.iter().copied().filter(|&a| a < device));
    let source = format!("{prelude}\n{SOURCE}");
    // NVRTC takes seconds for this kernel: compile each variant once per
    // process (`bench-gpu` loads the same one several times; the driver
    // caches its own PTX-to-binary step on disk).
    static CACHE: Mutex<Vec<(i32, String, Ptx)>> = Mutex::new(Vec::new());
    let cached = CACHE.lock().ok().and_then(|cache| {
        cache
            .iter()
            .find(|(d, p, _)| *d == device && p == prelude)
            .map(|(_, _, ptx)| ptx.clone())
    });
    if let Some(ptx) = cached {
        return ctx.load_module(ptx).map_err(cuda_err);
    }
    let mut last = String::new();
    for arch in archs {
        let opts = CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{arch}")],
            ..Default::default()
        };
        let compiled = std::panic::catch_unwind(|| compile_ptx_with_opts(&source, opts))
            .map_err(|_| "the NVRTC library (libnvrtc) could not be loaded".to_string())?;
        match compiled {
            Ok(ptx) => {
                if let Ok(mut cache) = CACHE.lock() {
                    cache.push((device, prelude.to_string(), ptx.clone()));
                }
                return ctx.load_module(ptx).map_err(cuda_err);
            }
            Err(CompileError::CompileError { log, .. }) => {
                let log = log.to_string_lossy().into_owned();
                // An unknown architecture is an option error: try an older
                // one; anything else is a real compilation error.
                if !log.contains("gpu-architecture") && !log.contains("invalid value") {
                    return Err(format!("CUDA kernel compilation failed:\n{log}"));
                }
                last = log;
            }
            Err(e) => last = format!("{e:?}"),
        }
    }
    Err(format!("CUDA kernel compilation failed: {last}"))
}

fn launch_config(threads: usize, block: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((threads as u32).div_ceil(block as u32), 1, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The compiled kernels and the device buffers of one GPU.
struct Engine {
    stream: Arc<CudaStream>,
    search: CudaFunction,
    shift: CudaFunction,
    /// `n` entries of `(x, y)` limbs: the walk centres.
    centres: CudaSlice<u32>,
    /// `(H+1)` entries of `(x, y)` limbs: `j·G` for `j = 1..=H`, then the jump.
    table: CudaSlice<u32>,
    /// `H·n` field elements of prefix products.
    scratch: CudaSlice<u32>,
    hits: CudaSlice<u32>,
    /// `[hits appended, degenerate walks]`.
    counters: CudaSlice<u32>,
    /// Batches each walk may still run in the current launch.
    remaining: CudaSlice<u32>,
    /// 1 where a walk met a zero difference and needs its centre recomputed.
    flags: CudaSlice<u32>,
    /// The point `shift` adds.
    delta: CudaSlice<u32>,
    n: usize,
    block: usize,
}

/// What one launch of the search kernel did.
struct Launch {
    /// Batches left per walk (compare with what was granted).
    remaining: Vec<u32>,
    hits: Vec<Hit>,
    /// Walks that met a zero difference (their flag is cleared).
    degenerate: Vec<usize>,
}

impl Engine {
    fn new(
        ordinal: usize,
        cfg: &Config,
        table: &Table,
        patterns: &PatternSet,
    ) -> Result<Engine, String> {
        cfg.check()?;
        let ctx = context(ordinal)?;
        // Wait for the GPU without spinning a core: the CPU workers want it.
        let _ = ctx.set_blocking_synchronize();
        let (free, _) = ctx.mem_get_info().map_err(cuda_err)?;
        if cfg.scratch_bytes() > free as u64 / 10 * 9 {
            return Err(format!(
                "GPU scratch would take {} MB, more than the {} MB free on device {ordinal}; lower \
                 --gpu-threads or --gpu-batch",
                cfg.scratch_bytes() >> 20,
                free >> 20
            ));
        }
        let module = compile(
            &ctx,
            &prelude(cfg.half, cfg.block, cfg.min_blocks, Some(patterns)),
        )?;
        let load = |name: &str| module.load_function(name).map_err(cuda_err);
        let (search_fn, shift_fn) = (load("search")?, load("shift")?);
        let stream = ctx.default_stream();
        let n = cfg.threads;
        let half = cfg.half;
        let mut points: Vec<u32> = Vec::with_capacity(16 * (half + 1));
        for j in 0..half {
            let (x, y) = table.point(j);
            points.extend_from_slice(&point_words(&x, &y));
        }
        let (jx, jy) = table.jump();
        points.extend_from_slice(&point_words(&jx, &jy));
        let zeros = |len: usize| stream.alloc_zeros::<u32>(len.max(1)).map_err(cuda_err);
        Ok(Engine {
            centres: zeros(16 * n)?,
            table: stream.clone_htod(&points).map_err(cuda_err)?,
            scratch: zeros(8 * half * n)?,
            hits: zeros(HIT_WORDS * HIT_CAPACITY)?,
            counters: zeros(2)?,
            remaining: zeros(n)?,
            flags: zeros(n)?,
            delta: zeros(16)?,
            search: search_fn,
            shift: shift_fn,
            stream,
            n,
            block: cfg.block,
        })
    }

    /// Runs up to `batches` batches on every walk (bounded per walk by
    /// `remaining`), blocking until done. `first_only`: a walk stops after the
    /// batch of its first hit.
    fn dispatch(
        &mut self,
        remaining: &[u32],
        batches: u32,
        first_only: bool,
    ) -> Result<Launch, String> {
        self.stream
            .memcpy_htod(remaining, &mut self.remaining)
            .map_err(cuda_err)?;
        self.stream
            .memset_zeros(&mut self.counters)
            .map_err(cuda_err)?;
        self.launch_search(batches, first_only)?;
        let remaining = self.stream.clone_dtoh(&self.remaining).map_err(cuda_err)?;
        let counters = self.stream.clone_dtoh(&self.counters).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        let count = (counters[0] as usize).min(HIT_CAPACITY);
        let hits = if count > 0 {
            let words = self
                .stream
                .clone_dtoh(&self.hits.slice(..count * HIT_WORDS))
                .map_err(cuda_err)?;
            self.stream.synchronize().map_err(cuda_err)?;
            words
                .as_chunks::<HIT_WORDS>()
                .0
                .iter()
                .map(Hit::parse)
                .collect()
        } else {
            Vec::new()
        };
        let degenerate = if counters[1] > 0 {
            self.take_flags()?
        } else {
            Vec::new()
        };
        Ok(Launch {
            remaining,
            hits,
            degenerate,
        })
    }

    fn launch_search(&mut self, batches: u32, first_only: bool) -> Result<(), String> {
        let n = self.n as u32;
        let cap = HIT_CAPACITY as u32;
        let first = u32::from(first_only);
        let mut launch = self.stream.launch_builder(&self.search);
        launch
            .arg(&mut self.centres)
            .arg(&self.table)
            .arg(&mut self.scratch)
            .arg(&n)
            .arg(&batches)
            .arg(&cap)
            .arg(&first)
            .arg(&mut self.hits)
            .arg(&mut self.counters)
            .arg(&mut self.remaining)
            .arg(&mut self.flags);
        // SAFETY: the arguments match `search` in search.cu one for one
        // (buffers sized in `Engine::new` for `n` walks and `H`, u32 scalars),
        // and the grid covers exactly the threads the kernel bounds-checks.
        unsafe { launch.launch(launch_config(self.n, self.block)) }.map_err(cuda_err)?;
        Ok(())
    }

    /// Adds the point `(x, y)` to every centre on the GPU; returns the walks
    /// it could not move (centre `±Δ`), whose centres the caller recomputes.
    fn shift_all(&mut self, x: &Fe, y: &Fe) -> Result<Vec<usize>, String> {
        self.stream
            .memcpy_htod(&point_words(x, y), &mut self.delta)
            .map_err(cuda_err)?;
        self.stream
            .memset_zeros(&mut self.counters)
            .map_err(cuda_err)?;
        self.launch_shift()?;
        let counters = self.stream.clone_dtoh(&self.counters).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        if counters[1] > 0 {
            self.take_flags()
        } else {
            Ok(Vec::new())
        }
    }

    fn launch_shift(&mut self) -> Result<(), String> {
        let n = self.n as u32;
        let mut launch = self.stream.launch_builder(&self.shift);
        launch
            .arg(&mut self.centres)
            .arg(&self.delta)
            .arg(&n)
            .arg(&mut self.flags)
            .arg(&mut self.counters);
        // SAFETY: the arguments match `shift` in search.cu one for one; the
        // grid covers the `n` walks the kernel bounds-checks.
        unsafe { launch.launch(launch_config(self.n, self.block)) }.map_err(cuda_err)?;
        Ok(())
    }

    /// The walks whose flag is set, clearing every flag.
    fn take_flags(&mut self) -> Result<Vec<usize>, String> {
        let flags = self.stream.clone_dtoh(&self.flags).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        self.stream
            .memset_zeros(&mut self.flags)
            .map_err(cuda_err)?;
        Ok(flags
            .iter()
            .enumerate()
            .filter(|&(_, &f)| f != 0)
            .map(|(t, _)| t)
            .collect())
    }

    /// Uploads the centres of the given walks.
    fn set_centres(&mut self, centres: &[(usize, Fe, Fe)]) -> Result<(), String> {
        if centres.len() > FULL_UPLOAD {
            let mut all = self.centres()?;
            for (t, x, y) in centres {
                all[16 * t..16 * t + 16].copy_from_slice(&point_words(x, y));
            }
            self.stream
                .memcpy_htod(&all, &mut self.centres)
                .map_err(cuda_err)?;
        } else {
            for (t, x, y) in centres {
                let mut view = self.centres.slice_mut(16 * t..16 * t + 16);
                self.stream
                    .memcpy_htod(&point_words(x, y), &mut view)
                    .map_err(cuda_err)?;
            }
        }
        self.stream.synchronize().map_err(cuda_err)
    }

    /// Every centre, as the 16 words per walk the kernel stores.
    fn centres(&self) -> Result<Vec<u32>, String> {
        let all = self.stream.clone_dtoh(&self.centres).map_err(cuda_err)?;
        self.stream.synchronize().map_err(cuda_err)?;
        Ok(all)
    }
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

/// Affine centres `base + (first + t·span)·G` for `t < n`: the start of a
/// split-mode range. Each core does one scalar multiplication, then point
/// additions and one batched normalisation, which is what keeps seeding a
/// million walks at a second or so. `None` for the point at infinity.
fn split_centres(base: &ProjectivePoint, first: u64, span: u64, n: usize) -> Vec<Option<(Fe, Fe)>> {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = n.div_ceil(cores).max(1);
    let step = ProjectivePoint::GENERATOR * Scalar::from(span);
    let mut out = vec![None; n];
    thread::scope(|scope| {
        for (c, outs) in out.chunks_mut(chunk).enumerate() {
            scope.spawn(move || {
                let t0 = (c * chunk) as u64;
                let mut point =
                    *base + ProjectivePoint::GENERATOR * Scalar::from(first + t0 * span);
                let mut points = Vec::with_capacity(outs.len());
                for _ in 0..outs.len() {
                    points.push(point);
                    point += step;
                }
                let affine =
                    <ProjectivePoint as BatchNormalize<[ProjectivePoint]>>::batch_normalize_vartime(
                        &points,
                    );
                for ((o, p), a) in outs.iter_mut().zip(&points).zip(&affine) {
                    if !bool::from(p.is_identity()) {
                        let x: [u8; 32] = a.x().into();
                        let y: [u8; 32] = a.y().into();
                        *o = Some((Fe::from_bytes_be(&x), Fe::from_bytes_be(&y)));
                    }
                }
            });
        }
    });
    out
}

/// The GPU worker for device `cfg.devices[slot]`: the counterpart of
/// `search::worker`. Every hit (or error) goes to `sender`; it returns when
/// `shared.stop` is set, the receiver is gone or (split mode) every range
/// has been taken.
pub fn worker(
    index: usize,
    slot: usize,
    cfg: &Config,
    patterns: &PatternSet,
    mode: &Mode,
    shared: &Shared,
    sender: &mpsc::Sender<Result<Found, String>>,
) {
    let ordinal = cfg.devices[slot];
    if let Err(message) = run(index, ordinal, cfg, patterns, mode, shared, sender) {
        let _ = sender.send(Err(format!("GPU {ordinal}: {message}")));
    }
}

/// Host-side state of the walks. A walk's next batch starts at
/// `start + done·(2H+1)`: the per-launch bookkeeping is integer counters, and
/// the scalar is only formed for a hit, a reseed or a range switch (a scalar
/// addition per walk per launch measured as idle GPU time).
struct Walks {
    /// Start scalar of every walk (secret in random mode).
    start: Zeroizing<Vec<Scalar>>,
    /// Batches run (or skipped) since `start`.
    done: Vec<u64>,
    /// Batches every walk may still run (split mode: within its span).
    budget: Vec<u64>,
    step: u64,
}

impl Walks {
    /// Scalar of walk `t`'s next batch centre.
    fn k0(&self, t: usize) -> Scalar {
        self.start[t].add(&Scalar::from(self.done[t]).mul(&Scalar::from(self.step)))
    }

    /// Advances walk `t` by `batches` batches.
    fn advance(&mut self, t: usize, batches: u64) {
        self.done[t] += batches;
        self.budget[t] = self.budget[t].saturating_sub(batches);
    }

    /// Restarts walk `t` at `k`.
    fn set(&mut self, t: usize, k: Scalar) {
        self.start[t] = k;
        self.done[t] = 0;
    }
}

fn run(
    index: usize,
    ordinal: usize,
    cfg: &Config,
    patterns: &PatternSet,
    mode: &Mode,
    shared: &Shared,
    sender: &mpsc::Sender<Result<Found, String>>,
) -> Result<(), String> {
    let table = Table::new(cfg.half, ProjectivePoint::GENERATOR);
    let mut engine = Engine::new(ordinal, cfg, &table, patterns)?;
    let tested = shared.counter(index);
    let n = cfg.threads;
    let base = match mode {
        Mode::Random => ProjectivePoint::IDENTITY,
        Mode::Split { base } => *base,
    };
    let mut walks = Walks {
        start: Zeroizing::new(vec![Scalar::ZERO; n]),
        done: vec![0; n],
        budget: vec![0; n],
        step: table.step(),
    };
    let mut batches: u64 = 2;
    let mut granted = vec![0u32; n];
    let mut seen = vec![false; n];
    // Split mode: the range the walks are sweeping.
    let mut current: Option<usize> = None;
    loop {
        if shared.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Seed, or move on once every walk has exhausted its span.
        if walks.budget.iter().all(|&b| b == 0) {
            match mode {
                Mode::Random => {
                    for t in 0..n {
                        walks.set(t, search::random_scalar()?);
                    }
                    walks.budget.fill(u64::MAX);
                    seed_centres(&mut engine, &mut walks, &base, (0..n).collect(), mode)?;
                }
                Mode::Split { .. } => {
                    let previous = current.take();
                    if let Some(done) = previous {
                        shared.mark_done(done);
                    }
                    let range = shared.next_range();
                    if range >= SPLIT_RANGES {
                        return Ok(());
                    }
                    current = Some(range);
                    let start = (range as u64) << RANGE_BITS;
                    let next: Vec<Scalar> = (0..n)
                        .map(|t| {
                            Scalar::from(start + t as u64 * cfg.split_span() + cfg.half as u64)
                        })
                        .collect();
                    let fix = match previous {
                        // First range: every centre from scratch.
                        None => {
                            let first = start + cfg.half as u64;
                            let centres = split_centres(&base, first, cfg.split_span(), n);
                            let mut upload = Vec::with_capacity(n);
                            let mut fix = Vec::new();
                            for (t, centre) in centres.into_iter().enumerate() {
                                match centre {
                                    Some((x, y)) => upload.push((t, x, y)),
                                    None => fix.push(t),
                                }
                            }
                            engine.set_centres(&upload)?;
                            fix
                        }
                        // Every walk ran or skipped all its batches, and
                        // walk t of every range starts t·span after the range,
                        // so every centre moves by the same scalar: one point
                        // addition per walk on the GPU (a walk that somehow
                        // did a different number of batches is recomputed).
                        Some(_) => {
                            let delta = next[0].sub(&walks.k0(0));
                            let (dx, dy) = affine_xy(&(ProjectivePoint::GENERATOR * delta));
                            let mut fix = engine.shift_all(&dx, &dy)?;
                            fix.extend((0..n).filter(|&t| walks.done[t] != walks.done[0]));
                            fix.sort_unstable();
                            fix.dedup();
                            fix
                        }
                    };
                    for (t, k) in next.into_iter().enumerate() {
                        walks.set(t, k);
                    }
                    walks.budget.fill(cfg.split_batches());
                    if !fix.is_empty() {
                        seed_centres(&mut engine, &mut walks, &base, fix, mode)?;
                    }
                }
            }
        }
        for (t, g) in granted.iter_mut().enumerate() {
            *g = walks.budget[t].min(batches) as u32;
        }
        let started = Instant::now();
        let launch = engine.dispatch(&granted, batches as u32, matches!(mode, Mode::Random))?;
        let elapsed = started.elapsed();

        // Batches actually run, hits, degenerate walks.
        let ran: Vec<u64> = launch
            .remaining
            .iter()
            .zip(&granted)
            .map(|(&rem, &granted)| u64::from(granted - rem))
            .collect();
        let points = ran.iter().sum::<u64>() * table.step();
        tested.add(3 * points);
        let mut reseed = Vec::new();
        for hit in &launch.hits {
            let t = hit.walk as usize;
            if t >= n || u64::from(hit.batch) >= ran[t] {
                return Err("internal: GPU hit record out of range".to_string());
            }
            if matches!(mode, Mode::Random) {
                // One hit per walk per launch, like the CPU worker, then a
                // fresh random start for that walk.
                if seen[t] {
                    continue;
                }
                seen[t] = true;
                reseed.push(t);
            }
            let x = from_limbs(&hit.x).ok_or("internal: GPU hit x is not canonical")?;
            let k0 = walks
                .k0(t)
                .add(&Scalar::from(u64::from(hit.batch) * table.step()));
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
            walks.set(t, search::random_scalar()?);
            seen[t] = false;
        }
        for &t in &launch.degenerate {
            // Like `Walk::skip_batch`: the batch is skipped, not redone.
            walks.advance(t, 1);
            reseed.push(t);
        }
        if !reseed.is_empty() {
            seed_centres(&mut engine, &mut walks, &base, reseed, mode)?;
        }
        // Aim for `TARGET_DISPATCH` per launch.
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
    engine: &mut Engine,
    walks: &mut Walks,
    base: &ProjectivePoint,
    mut which: Vec<usize>,
    mode: &Mode,
) -> Result<(), String> {
    while !which.is_empty() {
        let k0: Vec<Scalar> = which.iter().map(|&t| walks.k0(t)).collect();
        let centres = centres_of(&k0, base);
        let mut again = Vec::new();
        let mut upload = Vec::with_capacity(which.len());
        for (&t, centre) in which.iter().zip(centres) {
            match centre {
                Some((x, y)) => upload.push((t, x, y)),
                None => {
                    match mode {
                        Mode::Random => walks.set(t, search::random_scalar()?),
                        Mode::Split { .. } => walks.advance(t, 1),
                    }
                    again.push(t);
                }
            }
        }
        engine.set_centres(&upload)?;
        which = again;
    }
    Ok(())
}

// ---- tuning file -----------------------------------------------------------

/// GPU parameters for `--gpu-params`, as `bench-gpu` writes them.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    pub threads: usize,
    pub half: usize,
    pub block: usize,
    pub min_blocks: usize,
    /// CPU threads to run alongside (`-c`), when the bench found them worth it.
    pub cores: Option<usize>,
    /// Device the parameters were measured on (informational).
    pub device: Option<String>,
    /// Measured x candidates per second, all devices (informational).
    pub rate: Option<f64>,
}

impl Tuning {
    pub fn load(path: &Path) -> Result<Tuning, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("--gpu-params {}: {e}", path.display()))?;
        Tuning::parse(&text).map_err(|e| format!("--gpu-params {}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Tuning, String> {
        let (mut threads, mut half, mut block, mut min_blocks) = (None, None, None, None);
        let (mut cores, mut device, mut rate) = (None, None, None);
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line.split_once(' ').unwrap_or((line, ""));
            let value = value.trim();
            let number = || {
                value
                    .parse::<usize>()
                    .map_err(|_| format!("'{line}': expected a number"))
            };
            match key {
                "version" if value == "1" => {}
                "version" => return Err(format!("unsupported version {value}")),
                "gpu_threads" => threads = Some(number()?),
                "gpu_batch" => half = Some(number()?),
                "gpu_block" => block = Some(number()?),
                "gpu_min_blocks" => min_blocks = Some(number()?),
                "cores" => cores = Some(number()?),
                "device" => device = Some(value.to_string()),
                "rate" => rate = value.parse().ok(),
                _ => return Err(format!("unknown entry '{line}'")),
            }
        }
        let missing = |what: &str| format!("missing '{what}'");
        Ok(Tuning {
            threads: threads.ok_or_else(|| missing("gpu_threads"))?,
            half: half.ok_or_else(|| missing("gpu_batch"))?,
            block: block.ok_or_else(|| missing("gpu_block"))?,
            min_blocks: min_blocks.ok_or_else(|| missing("gpu_min_blocks"))?,
            cores,
            device,
            rate,
        })
    }

    /// The file: `comments` as `#` lines, then the entries.
    pub fn to_text(&self, comments: &[String]) -> String {
        let mut out = String::new();
        for line in comments {
            out.push_str(&format!("# {line}\n"));
        }
        out.push_str("version 1\n");
        out.push_str(&format!("gpu_threads {}\n", self.threads));
        out.push_str(&format!("gpu_batch {}\n", self.half));
        out.push_str(&format!("gpu_block {}\n", self.block));
        out.push_str(&format!("gpu_min_blocks {}\n", self.min_blocks));
        if let Some(cores) = self.cores {
            out.push_str(&format!("cores {cores}\n"));
        }
        if let Some(device) = &self.device {
            out.push_str(&format!("device {device}\n"));
        }
        if let Some(rate) = self.rate {
            out.push_str(&format!("rate {rate:.4e}\n"));
        }
        out
    }

    /// The equivalent command-line flags.
    pub fn flags(&self) -> String {
        format!(
            "--gpu --gpu-threads {} --gpu-batch {} --gpu-block {} --gpu-min-blocks {} -c {}",
            self.threads,
            self.half,
            self.block,
            self.min_blocks,
            self.cores.unwrap_or(0)
        )
    }

    /// Parameters tuned on another GPU model still work, but may be slow.
    pub fn warn_if_other_device(&self, cfg: &Config) {
        let Some(tuned) = &self.device else {
            return;
        };
        for &d in &cfg.devices {
            if let Ok(info) = device_info(d)
                && info.name != *tuned
            {
                eprintln!(
                    "warning: --gpu-params was tuned on {tuned}, device {d} is {}; rerun \
                     `spaghetti bench-gpu` on this machine",
                    info.name
                );
                return;
            }
        }
    }
}

// ---- benchmark -------------------------------------------------------------

/// What `bench-gpu` found.
pub struct TuneReport {
    pub tuning: Tuning,
    /// Rate of one device with the chosen parameters.
    pub per_device: f64,
    /// Rate of every device together, GPUs alone.
    pub gpus_alone: f64,
    /// Every configuration timed, with its rate.
    pub samples: Vec<(Config, f64)>,
}

/// The measurements of one `autotune` run.
struct Tuner<'a> {
    device: usize,
    patterns: &'a PatternSet,
    warmup: Duration,
    window: Duration,
    /// Scratch memory a configuration may take.
    budget: u64,
    /// Streaming multiprocessors of the device.
    sms: usize,
    samples: Vec<(Config, f64)>,
}

impl Tuner<'_> {
    fn fits(&self, cfg: &Config) -> bool {
        cfg.check().is_ok() && cfg.scratch_bytes() <= self.budget && split_fill(cfg) >= 0.97
    }

    /// The rate of `cfg` (timed once, then remembered).
    fn time(&mut self, cfg: &Config, log: &mut dyn FnMut(&str)) -> Result<f64, String> {
        if let Some((_, rate)) = self.samples.iter().find(|(c, _)| c == cfg) {
            return Ok(*rate);
        }
        let (registers, spill, _) = kernel_resources(self.device, cfg, self.patterns)?;
        let rate = measure(cfg, 0, 1024, self.patterns, self.warmup, self.window)?;
        log(&format!(
            "  H {:>4} × {:>7} walks, block {:>3} × {} ({registers} regs{}): {}/s",
            cfg.half,
            cfg.threads,
            cfg.block,
            cfg.min_blocks,
            if spill > 0 {
                format!(", {spill} B spilled")
            } else {
                String::new()
            },
            rate_text(rate)
        ));
        self.samples.push((cfg.clone(), rate));
        Ok(rate)
    }

    fn best(&self) -> Config {
        self.samples
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(c, _)| c.clone())
            .expect("at least one configuration timed")
    }

    /// Walks the device runs at once with `cfg`'s kernel shape.
    fn resident(&self, cfg: &Config) -> Result<usize, String> {
        let (_, _, blocks) = kernel_resources(self.device, cfg, self.patterns)?;
        Ok((blocks * cfg.block * self.sms).max(32))
    }

    /// Times `cfg` with `multiples` × the resident walk count.
    fn sweep_walks(
        &mut self,
        cfg: &Config,
        multiples: &[usize],
        log: &mut dyn FnMut(&str),
    ) -> Result<(), String> {
        let resident = self.resident(cfg)?;
        for &k in multiples {
            let threads = k * resident;
            let c = Config {
                threads,
                ..cfg.clone()
            };
            if (MIN_THREADS..=MAX_THREADS).contains(&threads) && self.fits(&c) {
                self.time(&c, log)?;
            }
        }
        Ok(())
    }
}

/// Times kernel shapes, walk counts and batch sizes on the first device of
/// `devices` (each configuration for `window` after a warm-up, in split mode
/// with a random base key and the planned `patterns`), then the winner on
/// every device together, with and without CPU threads. Coordinate descent:
/// shape first (threads per block × minimum blocks per SM, which sets the
/// register budget), then the walk count, then `H`, then the walk count
/// again. `log` gets one line per measurement.
pub fn autotune(
    devices: &[usize],
    patterns: &PatternSet,
    window: Duration,
    quick: bool,
    log: &mut dyn FnMut(&str),
) -> Result<TuneReport, String> {
    let first = devices[0];
    let info = device_info(first)?;
    log(&format!(
        "device {first}: {} (sm_{}{}, {} SMs, {} threads/SM, {} MB, {} MB free)",
        info.name,
        info.capability.0,
        info.capability.1,
        info.sms,
        info.threads_per_sm,
        info.total_mem >> 20,
        info.free_mem >> 20
    ));
    let mut tuner = Tuner {
        device: first,
        patterns,
        warmup: Duration::from_secs(1),
        window,
        budget: info.free_mem as u64 / 10 * 7,
        sms: info.sms,
        samples: Vec::new(),
    };

    // Shapes compared at four full devices of walks.
    let full = (info.sms * info.threads_per_sm).max(1024);
    let mut cfg = Config {
        threads: (4 * full).clamp(MIN_THREADS, MAX_THREADS),
        half: DEFAULT_HALF,
        block: DEFAULT_BLOCK,
        min_blocks: DEFAULT_MIN_BLOCKS,
        devices: vec![first],
    };
    while !tuner.fits(&cfg) && cfg.threads > MIN_THREADS {
        cfg.threads = (cfg.threads / 2).next_multiple_of(32);
    }

    log("shape (threads per block × minimum blocks per SM):");
    let blocks: &[usize] = if quick {
        &[128, 256]
    } else {
        &[64, 128, 256, 512]
    };
    let mins: &[usize] = if quick {
        &[1, 2, 4]
    } else {
        &[1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
    };
    let mut kernels: Vec<(usize, i32)> = Vec::new();
    for &block in blocks {
        for &min_blocks in mins {
            if block * min_blocks > info.threads_per_sm {
                continue;
            }
            let shape = Config {
                block,
                min_blocks,
                ..cfg.clone()
            };
            // A register cap the compiler met anyway gives the same kernel.
            let (registers, _, _) = kernel_resources(first, &shape, patterns)?;
            if kernels.contains(&(block, registers)) {
                continue;
            }
            kernels.push((block, registers));
            tuner.time(&shape, log)?;
        }
    }
    cfg = tuner.best();

    log("walks per device (multiples of what the device runs at once):");
    let multiples: &[usize] = if quick {
        &[4, 8, 16]
    } else {
        &[2, 4, 6, 8, 12, 16, 24, 32]
    };
    tuner.sweep_walks(&cfg.clone(), multiples, log)?;
    cfg = tuner.best();

    log("half-batch H:");
    let halves: &[usize] = if quick {
        &[256, 512, 1024]
    } else {
        &[128, 256, 512, 1024, 2048, 4096]
    };
    let resident = tuner.resident(&cfg)?;
    for &half in halves {
        // As many walk batches in flight as the best so far (same scratch
        // memory), in whole multiples of the resident walks, where it fits.
        let mut k = (cfg.threads * cfg.half / half / resident).max(1);
        let mut c = Config {
            half,
            threads: k * resident,
            ..cfg.clone()
        };
        while !tuner.fits(&c) && k > 1 {
            k -= 1;
            c.threads = k * resident;
        }
        if tuner.fits(&c) {
            tuner.time(&c, log)?;
        }
    }
    cfg = tuner.best();
    if !quick {
        log("walks per device, again:");
        let k = cfg.threads / tuner.resident(&cfg)?;
        let around: Vec<usize> = [k / 2, k * 3 / 4, k, k * 3 / 2, k * 2]
            .into_iter()
            .filter(|&m| m >= 1)
            .collect();
        tuner.sweep_walks(&cfg.clone(), &around, log)?;
        cfg = tuner.best();
    }
    // Short windows are noisy (clocks follow power and temperature): the
    // best three again, twice as long, and the fastest of those wins.
    log("finalists, timed again for twice as long:");
    let mut ranked = tuner.samples.clone();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut per_device = 0.0;
    for (c, _) in ranked.iter().take(3) {
        let rate = measure(c, 0, 1024, patterns, tuner.warmup, window * 2)?;
        log(&format!(
            "  H {:>4} × {:>7} walks, block {:>3} × {}: {}/s",
            c.half,
            c.threads,
            c.block,
            c.min_blocks,
            rate_text(rate)
        ));
        if rate > per_device {
            per_device = rate;
            cfg = c.clone();
        }
    }
    log(&format!(
        "best on one device: H {} × {} walks, block {} × {}: {}/s",
        cfg.half,
        cfg.threads,
        cfg.block,
        cfg.min_blocks,
        rate_text(per_device)
    ));

    let all = Config {
        devices: devices.to_vec(),
        ..cfg.clone()
    };
    let gpus_alone = if devices.len() > 1 {
        let rate = measure(&all, 0, 1024, patterns, tuner.warmup, window * 2)?;
        log(&format!(
            "all {} devices: {}/s ({:.2}× one device)",
            devices.len(),
            rate_text(rate),
            rate / per_device
        ));
        rate
    } else {
        per_device
    };
    // CPU threads alongside: worth it only for a clear gain, since a CPU
    // thread needs hours per range and loses its range on an interruption.
    let cpus = thread::available_parallelism()
        .map_or(1, |n| n.get())
        .saturating_sub(1 + devices.len());
    let mut cores = 0;
    let mut rate = gpus_alone;
    if cpus > 0 {
        let with_cpu = measure(&all, cpus, 4096, patterns, tuner.warmup, window * 2)?;
        log(&format!(
            "GPUs + {cpus} CPU threads: {}/s ({:+.1}%)",
            rate_text(with_cpu),
            100.0 * (with_cpu / gpus_alone - 1.0)
        ));
        if with_cpu > gpus_alone * 1.05 {
            cores = cpus;
            rate = with_cpu;
        }
    }
    Ok(TuneReport {
        tuning: Tuning {
            threads: cfg.threads,
            half: cfg.half,
            block: cfg.block,
            min_blocks: cfg.min_blocks,
            cores: Some(cores),
            device: Some(info.name),
            rate: Some(rate),
        },
        per_device,
        gpus_alone,
        samples: tuner.samples,
    })
}

fn rate_text(rate: f64) -> String {
    const UNITS: [&str; 5] = ["", " K", " M", " G", " T"];
    let mut value = rate;
    let mut unit = 0;
    while value >= 1000.0 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.2}{}", UNITS[unit])
}

/// Registers, spill bytes and resident blocks per SM of the search kernel
/// compiled for `cfg` and `patterns`.
pub fn kernel_resources(
    ordinal: usize,
    cfg: &Config,
    patterns: &PatternSet,
) -> Result<(i32, i32, usize), String> {
    let ctx = context(ordinal)?;
    let module = compile(
        &ctx,
        &prelude(cfg.half, cfg.block, cfg.min_blocks, Some(patterns)),
    )?;
    let f = module.load_function("search").map_err(cuda_err)?;
    let blocks = f
        .occupancy_max_active_blocks_per_multiprocessor(cfg.block as u32, 0, None)
        .map_err(cuda_err)?;
    Ok((
        f.num_regs().map_err(cuda_err)?,
        f.local_size_bytes().map_err(cuda_err)?,
        blocks as usize,
    ))
}

/// Runs the real GPU workers (split mode, a random base key, `patterns`) of
/// `cfg` plus `cpu` CPU threads for `warmup + window` and returns the rate over
/// the window: seeding and compilation are left out, the range switches of a
/// long search are in.
pub fn measure(
    cfg: &Config,
    cpu: usize,
    cpu_half: usize,
    patterns: &PatternSet,
    warmup: Duration,
    window: Duration,
) -> Result<f64, String> {
    cfg.check()?;
    let base = ProjectivePoint::GENERATOR * search::random_scalar()?;
    let mode = Mode::Split { base };
    let gpus = engines(cfg);
    let shared = Shared::split(gpus + cpu, 0);
    let cpu_table = Table::new(cpu_half, ProjectivePoint::GENERATOR);
    let (sender, receiver) = mpsc::channel();
    thread::scope(|scope| {
        for slot in 0..gpus {
            let sender = sender.clone();
            let (shared, mode) = (&shared, &mode);
            scope.spawn(move || worker(slot, slot, cfg, patterns, mode, shared, &sender));
        }
        for thread in 0..cpu {
            let sender = sender.clone();
            let (shared, mode, table) = (&shared, &mode, &cpu_table);
            scope.spawn(move || {
                search::worker(gpus + thread, table, patterns, mode, shared, &sender)
            });
        }
        drop(sender);
        let result = (|| {
            // Wait until every GPU has run its first launch.
            let started = Instant::now();
            while (0..gpus).any(|g| shared.counter(g).get() == 0) {
                if let Ok(Err(message)) = receiver.try_recv() {
                    return Err(message);
                }
                if started.elapsed() > Duration::from_secs(300) {
                    return Err("GPU did not start within 5 minutes".to_string());
                }
                thread::sleep(Duration::from_millis(20));
            }
            thread::sleep(warmup);
            // The count moves in steps, once per launch (~0.2 s per
            // device): time between two steps, not over a fixed window,
            // or the rate is off by up to a launch per window.
            let mut last = shared.tested_here();
            let mut first: Option<(Instant, u64)> = None;
            let mut latest = None;
            let end = Instant::now() + window + Duration::from_secs(2);
            loop {
                if let Ok(Err(message)) = receiver.try_recv() {
                    return Err(message);
                }
                let now = Instant::now();
                let count = shared.tested_here();
                if count != last {
                    last = count;
                    match first {
                        None => first = Some((now, count)),
                        Some((t0, _)) => {
                            latest = Some((now, count));
                            if now - t0 >= window {
                                break;
                            }
                        }
                    }
                }
                if now > end {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
            match (first, latest) {
                (Some((t0, c0)), Some((t1, c1))) => Ok((c1 - c0) as f64 / (t1 - t0).as_secs_f64()),
                _ => Err("the GPU made no progress during the measurement".to_string()),
            }
        })();
        shared.stop.store(true, Ordering::Relaxed);
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::Network;
    use crate::search::random_scalar;
    use std::cell::Cell;
    use std::collections::HashSet;

    thread_local! {
        /// Caps the batches per split-mode range on this thread, so a test
        /// sweeps several ranges in seconds.
        pub static BATCH_LIMIT: Cell<u64> = const { Cell::new(u64::MAX) };
    }

    fn x_of(k: &Scalar) -> Fe {
        affine_xy(&(ProjectivePoint::GENERATOR * k)).0
    }

    fn all_patterns() -> PatternSet {
        PatternSet::parse(&["sp1qq?".to_string()], Network::Mainnet).unwrap()
    }

    /// CI runners may have no GPU: such tests pass vacuously and say so.
    fn gpu_or_skip() -> bool {
        match device_count() {
            Ok(n) if n > 0 => true,
            _ => {
                eprintln!("no CUDA device: GPU test skipped");
                false
            }
        }
    }

    fn test_cfg(half: usize) -> Config {
        Config {
            threads: MIN_THREADS,
            half,
            block: 32,
            min_blocks: 1,
            devices: vec![0],
        }
    }

    /// Every field operation of the kernel against the CPU implementation,
    /// on edge cases and random elements.
    #[test]
    fn field_ops_match_cpu() {
        if !gpu_or_skip() {
            return;
        }
        let ctx = context(0).unwrap();
        let module = compile(&ctx, &prelude(8, 128, 1, None)).expect("kernel compiles");
        let f = module.load_function("field_test").unwrap();
        let stream = ctx.default_stream();
        let p_minus_1 = Fe::ZERO.sub(&Fe::ONE);
        let mut cases: Vec<Fe> = vec![
            Fe::ZERO,
            Fe::ONE,
            Fe::from_limbs([2, 0, 0, 0]).unwrap(),
            p_minus_1,
            p_minus_1.sub(&Fe::ONE),
            Fe::from_limbs([0, 0, 0, 1 << 63]).unwrap(),
            Fe::from_limbs([u64::MAX, u64::MAX, u64::MAX, 0]).unwrap(),
            Fe::from_limbs([0xFFFF_FFFE_FFFF_FC2E, u64::MAX, u64::MAX, u64::MAX - 1]).unwrap(),
            Fe::from_limbs([0xFFFF_FFFF, 0, 0, 0]).unwrap(),
            Fe::from_limbs([1 << 32, 0, 0, 0]).unwrap(),
            Fe::from_limbs([0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 0, 0]).unwrap(),
            Fe::from_limbs([0, 0, u64::MAX, u64::MAX]).unwrap(),
            Fe::BETA,
            Fe::BETA2,
        ];
        // Elements near the reduction edges: 2^256 − 2^32 − 977 − k and
        // values whose square lands on the fold carries.
        for k in 1..20u64 {
            cases.push(p_minus_1.sub(&Fe::from_limbs([k * 977, 0, 0, 0]).unwrap()));
            cases.push(Fe::from_limbs([u64::MAX - k, u64::MAX, u64::MAX, u64::MAX >> 1]).unwrap());
        }
        for _ in 0..300 {
            cases.push(x_of(&random_scalar().unwrap()));
        }
        let mut pairs: Vec<(Fe, Fe)> = Vec::new();
        for a in &cases {
            for b in cases.iter().step_by(5) {
                pairs.push((*a, *b));
            }
            pairs.push((*a, *a));
        }
        let mut input: Vec<u32> = Vec::with_capacity(16 * pairs.len());
        for (a, b) in &pairs {
            input.extend_from_slice(&point_words(a, b));
        }
        let count = pairs.len() as u32;
        let inb = stream.clone_htod(&input).unwrap();
        let mut outb = stream.alloc_zeros::<u32>(pairs.len() * 48).unwrap();
        let mut launch = stream.launch_builder(&f);
        launch.arg(&inb).arg(&mut outb).arg(&count);
        // SAFETY: matches `field_test(const uint4*, uint4*, u32)`; buffers sized
        // for `count` pairs.
        unsafe { launch.launch(launch_config(pairs.len(), 128)) }.unwrap();
        let out = stream.clone_dtoh(&outb).unwrap();
        stream.synchronize().unwrap();
        for ((a, b), got) in pairs.iter().zip(out.as_chunks::<48>().0) {
            let got: Vec<Fe> = got
                .as_chunks::<8>()
                .0
                .iter()
                .map(|l| from_limbs(l).expect("canonical"))
                .collect();
            assert_eq!(got[0], a.mul(b), "mul {a:?} {b:?}");
            assert_eq!(got[1], a.square(), "sqr {a:?}");
            assert_eq!(got[2], a.add(b), "add {a:?} {b:?}");
            assert_eq!(got[3], a.sub(b), "sub {a:?} {b:?}");
            assert_eq!(got[4], a.neg(), "neg {a:?}");
            if !a.is_zero() {
                assert_eq!(got[5], a.invert(), "inv {a:?}");
            }
        }
    }

    fn engine_with(cfg: &Config, k0: &[Scalar]) -> Engine {
        let table = Table::new(cfg.half, ProjectivePoint::GENERATOR);
        let mut engine = Engine::new(0, cfg, &table, &all_patterns()).expect("engine");
        let centres: Vec<(usize, Fe, Fe)> = centres_of(k0, &ProjectivePoint::IDENTITY)
            .into_iter()
            .enumerate()
            .map(|(t, c)| {
                let (x, y) = c.unwrap();
                (t, x, y)
            })
            .collect();
        engine.set_centres(&centres).unwrap();
        engine
    }

    /// With a match-everything pattern the kernel reports every visited x;
    /// each one must be `x(λ^e·(k0 + offset)·G)` and every offset of every
    /// batch must appear exactly once per endomorphism power.
    #[test]
    fn walk_visits_every_point_and_matches_k256() {
        if !gpu_or_skip() {
            return;
        }
        let cfg = test_cfg(8);
        let patterns = all_patterns();
        let n = cfg.threads;
        let k0: Vec<Scalar> = (0..n).map(|_| random_scalar().unwrap()).collect();
        let mut engine = engine_with(&cfg, &k0);
        let batches = 3u32;
        let launch = engine.dispatch(&vec![batches; n], batches, false).unwrap();
        assert!(launch.remaining.iter().all(|&r| r == 0));
        assert!(launch.degenerate.is_empty());
        let step = 2 * cfg.half as u64 + 1;
        let per_batch = 3 * step as usize;
        assert_eq!(launch.hits.len(), n * batches as usize * per_batch);
        let mut seen = HashSet::new();
        for hit in &launch.hits {
            let t = hit.walk as usize;
            let k0 = k0[t].add(&Scalar::from(u64::from(hit.batch) * step));
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
        let centres = engine.centres().unwrap();
        for t in 0..n {
            let k = k0[t].add(&Scalar::from(u64::from(batches) * step));
            assert_eq!(
                from_limbs(&centres[16 * t..16 * t + 8]).unwrap(),
                x_of(&k),
                "walk {t}"
            );
        }
    }

    /// The shift kernel moves every centre by the same point and flags the
    /// one it cannot move.
    #[test]
    fn shift_moves_every_centre() {
        if !gpu_or_skip() {
            return;
        }
        let cfg = test_cfg(8);
        let n = cfg.threads;
        let delta = random_scalar().unwrap();
        let mut k0: Vec<Scalar> = (0..n).map(|_| random_scalar().unwrap()).collect();
        k0[3] = delta; // C = Δ: flagged
        k0[4] = delta.negate(); // C = −Δ: flagged
        let mut engine = engine_with(&cfg, &k0);
        let (dx, dy) = affine_xy(&(ProjectivePoint::GENERATOR * delta));
        let flagged = engine.shift_all(&dx, &dy).unwrap();
        assert_eq!(flagged, vec![3, 4]);
        let centres = engine.centres().unwrap();
        for t in (0..n).filter(|t| !flagged.contains(t)) {
            let (x, y) = affine_xy(&(ProjectivePoint::GENERATOR * k0[t].add(&delta)));
            assert_eq!(from_limbs(&centres[16 * t..16 * t + 8]).unwrap(), x);
            assert_eq!(from_limbs(&centres[16 * t + 8..16 * t + 16]).unwrap(), y);
        }
        // Flags were cleared.
        assert!(engine.shift_all(&dx, &dy).unwrap().len() <= 2);
    }

    /// A centre equal to a table point is flagged, not silently corrupted.
    #[test]
    fn degenerate_centre_is_flagged() {
        if !gpu_or_skip() {
            return;
        }
        let cfg = test_cfg(8);
        // Walk 0 starts at 5·G (a table point); walk 1 at 17·G (the jump).
        let k0: Vec<Scalar> = (0..cfg.threads)
            .map(|t| match t {
                0 => Scalar::from(5u64),
                1 => Scalar::from(17u64),
                _ => random_scalar().unwrap(),
            })
            .collect();
        let mut engine = engine_with(&cfg, &k0);
        let launch = engine.dispatch(&vec![2; cfg.threads], 2, false).unwrap();
        assert_eq!(launch.degenerate, vec![0, 1]);
        assert_eq!(&launch.remaining[..3], &[2, 2, 0]);
        assert!(launch.hits.iter().all(|h| h.walk >= 2));
        assert_eq!(
            launch.hits.len(),
            (cfg.threads - 2) * 2 * 3 * (2 * cfg.half + 1)
        );
    }

    /// The worker end to end, in random mode: a real (easy) pattern, matches
    /// verified against k256 by the resolver.
    #[test]
    fn random_mode_worker_finds_verified_matches() {
        if !gpu_or_skip() {
            return;
        }
        let cfg = test_cfg(16);
        let patterns = PatternSet::parse(&["sp1qqgq".to_string()], Network::Mainnet).unwrap();
        let shared = Shared::new(1);
        let (sender, receiver) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| worker(0, 0, &cfg, &patterns, &Mode::Random, &shared, &sender));
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

    /// Split mode across several ranges (batches capped per range so the
    /// shift path runs within seconds): every
    /// reported tweak reproduces the address from the base key, offsets stay
    /// below 2^MAX_TWEAK_BITS, and the matches come from more than one range.
    #[test]
    fn split_mode_worker_reports_valid_tweaks() {
        if !gpu_or_skip() {
            return;
        }
        let cfg = Config {
            threads: 1 << 16,
            half: 128,
            block: 128,
            min_blocks: 1,
            devices: vec![0],
        };
        let patterns = PatternSet::parse(&["sp1qq?qqqq".to_string()], Network::Mainnet).unwrap();
        let base = ProjectivePoint::GENERATOR * random_scalar().unwrap();
        let mode = Mode::Split { base };
        let shared = Shared::split(1, 7);
        let (sender, receiver) = mpsc::channel();
        let mut ranges = HashSet::new();
        thread::scope(|scope| {
            scope.spawn(|| {
                BATCH_LIMIT.set(4);
                worker(0, 0, &cfg, &patterns, &mode, &shared, &sender)
            });
            let started = Instant::now();
            while ranges.len() < 3 && started.elapsed() < Duration::from_secs(600) {
                let found = receiver.recv().unwrap().expect("no error");
                let search::Key::Tweak { base: b, tweak } = found.key else {
                    panic!("split mode reports tweaks");
                };
                assert_eq!(b, base);
                assert!(tweak.t < 1 << crate::tweak::MAX_TWEAK_BITS);
                assert!(tweak.t >> RANGE_BITS >= 7, "{tweak}");
                assert_eq!(search::compressed(&tweak.apply_point(&base)), found.pubkey);
                ranges.insert(tweak.t >> RANGE_BITS);
            }
            shared.stop.store(true, Ordering::Relaxed);
        });
        assert!(ranges.len() >= 3, "matches from {ranges:?}");
    }

    #[test]
    fn config_limits() {
        let ok = Config {
            threads: DEFAULT_THREADS,
            half: DEFAULT_HALF,
            block: DEFAULT_BLOCK,
            min_blocks: DEFAULT_MIN_BLOCKS,
            devices: vec![0],
        };
        assert!(ok.check().is_ok());
        for bad in [
            Config {
                threads: 1000,
                ..ok.clone()
            },
            Config {
                half: 0,
                ..ok.clone()
            },
            Config {
                block: 100,
                ..ok.clone()
            },
            Config {
                devices: vec![],
                ..ok.clone()
            },
            Config {
                devices: vec![1, 1],
                ..ok.clone()
            },
            Config {
                threads: 2 * MAX_THREADS,
                ..ok.clone()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
        // Every walk's last visited offset stays inside its span.
        let last =
            ok.half as u64 + (ok.split_batches() - 1) * (2 * ok.half as u64 + 1) + ok.half as u64;
        assert!(last < ok.split_span());
        assert!(split_fill(&ok) > 0.99);
        assert!(split_coverage(&ok) > 0.99 * 3.0 * 2f64.powi(58));
    }

    #[test]
    fn prelude_bakes_the_keys() {
        let patterns = PatternSet::parse(&["sp1qqgpasta".to_string()], Network::Mainnet).unwrap();
        let text = prelude(512, 128, 2, Some(&patterns));
        assert!(text.contains("#define HALF 512"));
        assert!(text.contains("#define MINB 2"));
        let (mask, value) = patterns.keys[0];
        assert!(text.contains(&format!("0x{:08x}u", (value >> 32) as u32)));
        assert!(text.contains(&format!("0x{:08x}u", (mask >> 32) as u32)));
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::search::random_scalar;

    fn x_of(k: &Scalar) -> Fe {
        affine_xy(&(ProjectivePoint::GENERATOR * k)).0
    }

    /// `cargo test --release --features cuda bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn report() {
        if device_count().unwrap_or(0) == 0 {
            return;
        }
        let ctx = context(0).unwrap();
        let module = compile(&ctx, &prelude(512, 128, 1, None)).unwrap();
        for name in ["search", "shift", "field_test", "bench"] {
            let f = module.load_function(name).unwrap();
            eprintln!(
                "{name}: {} registers, {} bytes local",
                f.num_regs().unwrap(),
                f.local_size_bytes().unwrap()
            );
        }
        let f = module.load_function("bench").unwrap();
        let stream = ctx.default_stream();
        let n: usize = 1 << 17;
        let iters: u32 = 2000;
        let a = x_of(&random_scalar().unwrap());
        let b = x_of(&random_scalar().unwrap());
        let mut data: Vec<u32> = Vec::with_capacity(16 * n);
        for _ in 0..n {
            data.extend_from_slice(&point_words(&a, &b));
        }
        let mut buffer = stream.clone_htod(&data).unwrap();
        for (mode, name) in [
            (0u32, "fe_mul"),
            (1, "fe_sqr"),
            (2, "fe_add"),
            (3, "point test"),
        ] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = Instant::now();
                let mut launch = stream.launch_builder(&f);
                launch.arg(&mut buffer).arg(&iters).arg(&mode);
                // SAFETY: matches `bench(uint4*, u32, u32)`; `n` threads, `n` pairs.
                unsafe { launch.launch(launch_config(n, 128)) }.unwrap();
                stream.synchronize().unwrap();
                best = best.min(started.elapsed().as_secs_f64());
            }
            let ops = n as f64 * iters as f64;
            eprintln!("{name}: {:.2} G ops/s", ops / best / 1e9);
        }
    }
}
