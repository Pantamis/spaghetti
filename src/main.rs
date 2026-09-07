//! `spaghetti`: find a BIP352 silent payment scan key whose addresses start
//! with a chosen prefix.

mod address;
mod bip32;
mod field;
mod pattern;
mod recover;
mod search;
mod tweak;

use std::io::Write;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand, ValueEnum};
use k256::elliptic_curve::{Generate, PrimeField};
use k256::{NonZeroScalar, ProjectivePoint, PublicKey, Scalar};

use pattern::{Network, Pattern, PatternSet};
use search::{Found, Key, Mode, Table, Worker};
use tweak::{MAX_TWEAK_BITS, RANGE_BITS, SPLIT_RANGES, Tweak};

/// Upper bound on `-c` (OS threads).
const MAX_THREADS: usize = 1024;
/// Upper bound on the hidden `--batch` half size `H` (`2H <= 2^20`).
const MAX_BATCH_HALF: usize = 1 << 19;

#[derive(Clone, Copy, ValueEnum)]
enum NetworkArg {
    /// hrp `sp`
    Mainnet,
    /// hrp `tsp` (BIP352: testnet and signet)
    Testnet,
}

impl From<NetworkArg> for Network {
    fn from(value: NetworkArg) -> Network {
        match value {
            NetworkArg::Mainnet => Network::Mainnet,
            NetworkArg::Testnet => Network::Testnet,
        }
    }
}

/// Vanity BIP352 silent payment address generator: finds a scan key whose addresses start with a chosen prefix.
#[derive(Parser)]
#[command(
    version,
    about,
    long_about = None,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// address prefix, full (`sp1qq?pasta`) or bare (`pasta` = `sp1qq?pasta`); `?` = any char
    #[arg(required = true, value_name = "PATTERN")]
    patterns: Vec<String>,

    /// mainnet (hrp `sp`) | testnet (hrp `tsp`, the BIP352 hrp for testnet and signet;
    /// regtest has no standard hrp and is not supported)
    #[arg(short, long, value_enum, default_value_t = NetworkArg::Mainnet)]
    network: NetworkArg,

    /// OS threads, at most 1024 [default: available_parallelism]
    #[arg(short, long, value_name = "N")]
    cores: Option<usize>,

    /// stop after N matches
    #[arg(short = 'k', long, value_name = "N", default_value_t = 1)]
    count: u64,

    /// spend public key used to render the example address (random throwaway one if omitted)
    #[arg(short, long, value_name = "HEX33")]
    spend_pubkey: Option<String>,

    #[command(flatten)]
    base: BaseKeyArgs,

    /// no progress output
    #[arg(short, long)]
    quiet: bool,

    /// half-batch size H (2H points per inversion)
    #[arg(long, value_name = "N", default_value_t = 1024, hide = true)]
    batch: usize,
}

/// Where the base scan public key `D` of split-key mode comes from.
#[derive(Args)]
struct BaseKeyArgs {
    /// split-key mode: search offsets from this compressed scan pubkey D
    #[arg(short = 'b', long, value_name = "HEX33", conflicts_with = "xpub")]
    base_pubkey: Option<String>,

    /// split-key mode: D = child 0 (non-hardened) of this extended pubkey; give the node
    /// m/352'/0'/0'/1' (testnet: m/352'/1'/0'/1', tpub). Mutually exclusive with -b.
    #[arg(long, value_name = "XPUB")]
    xpub: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Recover the split-key tweak of a published address from the base scan pubkey
    Recover(RecoverArgs),
    /// Apply a tweak to the BIP32-derived scan secret key
    Apply(ApplyArgs),
}

#[derive(Args)]
struct RecoverArgs {
    /// silent payment address made with split-key mode
    #[arg(long, value_name = "SP_ADDRESS")]
    address: String,

    #[command(flatten)]
    base: BaseKeyArgs,

    /// OS threads, at most 1024 [default: available_parallelism]
    #[arg(short, long, value_name = "N")]
    cores: Option<usize>,

    /// baby-step table size 2^K (≈ 15 bytes × 2^K: 22 → 60 MB; each +1 halves the giant-step work)
    #[arg(long, value_name = "K", default_value_t = 22)]
    baby_bits: u32,

    /// half-batch size H (2H points per inversion)
    #[arg(long, value_name = "N", default_value_t = 1024, hide = true)]
    batch: usize,

    /// search offsets below 2^M (testing only)
    #[arg(long, value_name = "M", default_value_t = MAX_TWEAK_BITS, hide = true)]
    max_bits: u32,
}

#[derive(Args)]
struct ApplyArgs {
    /// BIP32-derived scan secret key d (m/352'/coin'/account'/1'/0)
    #[arg(long, value_name = "HEX32")]
    scan_priv: String,

    /// tweak string `<t>/<e>/<s>` printed by the search or by `recover`
    #[arg(long, value_name = "t/e/s")]
    tweak: Tweak,

    /// address to check against the resulting scan pubkey
    #[arg(long, value_name = "SP_ADDRESS")]
    address: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome = match cli.command {
        Some(Command::Recover(args)) => run_recover(args),
        Some(Command::Apply(args)) => run_apply(&args),
        None => run(cli),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn parse_spend_pubkey(hex_key: Option<&str>) -> Result<([u8; 33], bool), String> {
    match hex_key {
        Some(text) => {
            let point = parse_pubkey(text, "--spend-pubkey")?;
            Ok((search::compressed(&point), true))
        }
        None => {
            let scalar = NonZeroScalar::try_generate().map_err(|e| format!("rng failure: {e}"))?;
            let key = PublicKey::from_secret_scalar(&scalar);
            Ok((search::compressed(&key.to_projective()), false))
        }
    }
}

fn parse_pubkey(text: &str, what: &str) -> Result<ProjectivePoint, String> {
    let bytes = hex::decode(text.trim()).map_err(|e| format!("{what}: {e}"))?;
    if bytes.len() != 33 {
        return Err(format!(
            "{what}: expected a 33-byte compressed public key, got {} bytes",
            bytes.len()
        ));
    }
    PublicKey::from_sec1_bytes(&bytes)
        .map(|key| key.to_projective())
        .map_err(|_| format!("{what}: not a valid secp256k1 public key"))
}

/// Base scan pubkey `D` from `-b` or `--xpub`, checked against `network`.
fn parse_base_key(args: &BaseKeyArgs, network: Network) -> Result<Option<ProjectivePoint>, String> {
    match (&args.base_pubkey, &args.xpub) {
        (Some(hex_key), None) => parse_pubkey(hex_key, "--base-pubkey").map(Some),
        (None, Some(xpub)) => {
            let (key, key_network) = bip32::scan_account_pubkey(xpub)?;
            if key_network != network {
                return Err(format!(
                    "--xpub is a {} key but the search/address network is {}",
                    describe(key_network),
                    describe(network)
                ));
            }
            parse_pubkey(&hex::encode(key), "--xpub child").map(Some)
        }
        (None, None) => Ok(None),
        (Some(_), Some(_)) => Err("give either --base-pubkey or --xpub, not both".to_string()),
    }
}

fn describe(network: Network) -> &'static str {
    match network {
        Network::Mainnet => "mainnet (xpub / sp1…)",
        Network::Testnet => "testnet (tpub / tsp1…)",
    }
}

fn thread_count(cores: Option<usize>) -> Result<usize, String> {
    let n = match cores {
        Some(0) | None => thread::available_parallelism().map_or(1, |n| n.get()),
        Some(n) => n,
    };
    if n > MAX_THREADS {
        return Err(format!("--cores: at most {MAX_THREADS} threads, got {n}"));
    }
    Ok(n)
}

fn check_batch(half: usize) -> Result<(), String> {
    if !(1..=MAX_BATCH_HALF).contains(&half) {
        return Err(format!(
            "--batch: H must be between 1 and {MAX_BATCH_HALF} (2H points per inversion, at most \
             2^20), got {half}"
        ));
    }
    Ok(())
}

fn run(cli: Cli) -> Result<(), String> {
    let network: Network = cli.network.into();
    let patterns = PatternSet::parse(&cli.patterns, network)?;
    let (spend, spend_provided) = parse_spend_pubkey(cli.spend_pubkey.as_deref())?;
    let hrp = match network {
        Network::Mainnet => address::HRP_MAINNET,
        Network::Testnet => address::HRP_TESTNET,
    };
    let cores = thread_count(cli.cores)?;
    check_batch(cli.batch)?;
    if cli.count == 0 {
        return Err("--count must be at least 1".to_string());
    }
    let mode = match parse_base_key(&cli.base, network)? {
        Some(base) => Mode::Split { base },
        None => Mode::Random,
    };
    // Split mode hands out SPLIT_RANGES offset ranges from a queue: more OS
    // threads than ranges would have nothing to do.
    let threads = match mode {
        Mode::Random => cores,
        Mode::Split { .. } => cores.min(SPLIT_RANGES),
    };
    let table = Table::new(cli.batch, ProjectivePoint::GENERATOR);
    let expected = patterns.expected_candidates();
    let quiet = cli.quiet;

    if !quiet {
        let names: Vec<&str> = patterns.patterns.iter().map(|p| p.text.as_str()).collect();
        eprintln!(
            "searching {} | difficulty 2^{} ≈ {} candidates | {threads} threads, batch {}{}",
            names.join(" | "),
            patterns.patterns.iter().map(|p| p.bits).min().unwrap_or(0),
            human_count(expected),
            2 * table.half,
            if matches!(mode, Mode::Split { .. }) {
                format!(" | split-key mode ({SPLIT_RANGES} ranges of 2^{RANGE_BITS} offsets)")
            } else {
                String::new()
            }
        );
        warn_split_coverage(&mode, expected, table.half);
    }

    let stop = AtomicBool::new(false);
    let tested = AtomicU64::new(0);
    let ranges = AtomicUsize::new(0);
    let (sender, receiver) = mpsc::channel::<Result<Found, String>>();
    let started = Instant::now();
    let mut found_count = 0u64;
    let mut result = Ok(());

    thread::scope(|scope| {
        let mut spawn_error = None;
        for thread in 0..threads {
            let sender = sender.clone();
            let (table, patterns, mode) = (&table, &patterns, &mode);
            let (ranges, stop, tested) = (&ranges, &stop, &tested);
            let spawned = thread::Builder::new().spawn_scoped(scope, move || {
                worker_loop(table, patterns, mode, ranges, stop, tested, &sender)
            });
            if let Err(e) = spawned {
                spawn_error = Some(format!(
                    "could not spawn worker thread {} of {threads}: {e}; lower -c",
                    thread + 1
                ));
                break;
            }
        }
        drop(sender);
        if let Some(message) = spawn_error {
            stop.store(true, Ordering::Relaxed);
            result = Err(message);
            return;
        }

        let mut last_progress = Instant::now();
        let mut printed_progress = false;
        loop {
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(found)) => {
                    let elapsed = started.elapsed();
                    let n = tested.load(Ordering::Relaxed);
                    if printed_progress {
                        eprint!("\r\x1b[K");
                    }
                    let addr = address::encode(hrp, &found.pubkey, &spend);
                    let pattern = &patterns.patterns[found.pattern];
                    if let Err(message) = final_check(&addr, &found, pattern, &mode) {
                        result = Err(message);
                        break;
                    }
                    print_match(&found, &mode, &spend, spend_provided, &addr, n, elapsed);
                    found_count += 1;
                    if found_count >= cli.count {
                        break;
                    }
                }
                Ok(Err(message)) => {
                    result = Err(message);
                    break;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    result = Err(match mode {
                        Mode::Random => "all workers exited".to_string(),
                        Mode::Split { .. } => format!(
                            "split-key mode: all {SPLIT_RANGES} ranges of 2^{RANGE_BITS} offsets \
                             (every offset below 2^{MAX_TWEAK_BITS}) exhausted without a match; \
                             use a shorter pattern"
                        ),
                    });
                    break;
                }
            }
            if !quiet && last_progress.elapsed() >= Duration::from_secs(1) {
                last_progress = Instant::now();
                printed_progress = true;
                print_progress(tested.load(Ordering::Relaxed), expected, started.elapsed());
            }
        }
        stop.store(true, Ordering::Relaxed);
        if printed_progress {
            eprint!("\r\x1b[K");
        }
    });
    result
}

/// Split-key mode covers all `2^52` offsets, i.e. about `3·2^52` x candidates
/// whatever the thread count: warn when the pattern is expected to need more
/// than a quarter of that.
fn warn_split_coverage(mode: &Mode, expected: f64, half: usize) {
    if !matches!(mode, Mode::Split { .. }) {
        return;
    }
    let per_range = search::split_batches(RANGE_BITS, half) as f64 * (2 * half + 1) as f64;
    let coverage = 3.0 * per_range * SPLIT_RANGES as f64;
    if expected * 4.0 > coverage {
        eprintln!(
            "warning: split-key mode covers at most ≈ {} candidates (every offset below \
             2^{MAX_TWEAK_BITS}) and the pattern needs ≈ {} on average: the search will likely \
             exhaust all offsets without a match; use a shorter pattern",
            human_count(coverage),
            human_count(expected)
        );
    }
}

/// Independent re-check of a match: the rendered address must carry the
/// requested prefix and decode (bech32m) back to the scan key, and the printed
/// key material, re-parsed from its text form, must reproduce the scan key
/// (random mode: `G·secret`; split-key mode: the tweak applied to the base key).
fn final_check(addr: &str, found: &Found, pattern: &Pattern, mode: &Mode) -> Result<(), String> {
    if !pattern.matches_address(addr) {
        return Err(format!(
            "internal: address {addr} does not match {}",
            pattern.text
        ));
    }
    let (_, scan, _) = address::decode(addr).map_err(|e| format!("internal: {e}"))?;
    if scan != found.pubkey {
        return Err(format!(
            "internal: address {addr} does not decode to the scan key"
        ));
    }
    match (mode, &found.key) {
        (Mode::Random, Key::Secret(secret)) => {
            let reparsed = parse_secret(&hex::encode(secret), "internal: printed secret key")?;
            if search::compressed(&(ProjectivePoint::GENERATOR * reparsed)) != found.pubkey {
                return Err("internal: secret key does not reproduce the scan key".to_string());
            }
            Ok(())
        }
        (Mode::Split { base }, Key::Tweak(tweak)) => {
            let reparsed: Tweak = tweak.to_string().parse()?;
            if search::compressed(&reparsed.apply_point(base)) != found.pubkey {
                return Err(format!(
                    "internal: tweak {tweak} does not reproduce the scan key from the base key"
                ));
            }
            Ok(())
        }
        _ => Err("internal: search mode and result kind disagree".to_string()),
    }
}

/// One worker thread. Random mode: walk from a random start; after a hit,
/// report only that hit and move to a fresh random start, so two keys from
/// one run are never related by a small public offset (which would let anyone
/// prove common ownership and turn one secret into the other). Split mode:
/// take offset ranges `[i·2^44, (i+1)·2^44)` from the shared queue until all
/// `SPLIT_RANGES` are done; a range starts at `i·2^44 + H` so the first batch
/// visits `i·2^44 ..= i·2^44 + 2H`, and stops before a batch would cross the
/// range end.
fn worker_loop(
    table: &Table,
    patterns: &PatternSet,
    mode: &Mode,
    ranges: &AtomicUsize,
    stop: &AtomicBool,
    tested: &AtomicU64,
    sender: &mpsc::Sender<Result<Found, String>>,
) {
    let mut hits = Vec::new();
    match mode {
        Mode::Random => {
            let mut worker = match Worker::new(table) {
                Ok(worker) => worker,
                Err(message) => {
                    let _ = sender.send(Err(message));
                    return;
                }
            };
            while !stop.load(Ordering::Relaxed) {
                if worker.search_batch(patterns, &mut hits) {
                    tested.fetch_add(table.candidates_per_batch(), Ordering::Relaxed);
                }
                if let Some(candidate) = hits.first() {
                    let resolved = search::resolve(candidate, patterns, mode);
                    hits.clear();
                    if sender.send(resolved).is_err() {
                        return;
                    }
                    if let Err(message) = worker.reseed() {
                        let _ = sender.send(Err(message));
                        return;
                    }
                }
            }
        }
        Mode::Split { base } => loop {
            let range = ranges.fetch_add(1, Ordering::Relaxed);
            if range >= SPLIT_RANGES || stop.load(Ordering::Relaxed) {
                return;
            }
            let k0 = search::split_start(range, RANGE_BITS, table.half);
            let mut worker = Worker::with_base(table, Scalar::from(k0), *base);
            for _ in 0..search::split_batches(RANGE_BITS, table.half) {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                if worker.search_batch(patterns, &mut hits) {
                    tested.fetch_add(table.candidates_per_batch(), Ordering::Relaxed);
                }
                for candidate in hits.drain(..) {
                    if sender
                        .send(search::resolve(&candidate, patterns, mode))
                        .is_err()
                    {
                        return;
                    }
                }
            }
        },
    }
}

fn print_match(
    found: &Found,
    mode: &Mode,
    spend: &[u8; 33],
    spend_provided: bool,
    addr: &str,
    tested: u64,
    elapsed: Duration,
) {
    let spend_note = if spend_provided {
        "(provided)"
    } else {
        "(example, random)"
    };
    let rate = tested as f64 / elapsed.as_secs_f64().max(1e-9);
    let mut out = std::io::stdout().lock();
    match (mode, &found.key) {
        (Mode::Split { base }, Key::Tweak(tweak)) => {
            let _ = writeln!(
                out,
                "base scan pubkey  : {}",
                hex::encode(search::compressed(base))
            );
            let _ = writeln!(out, "tweak             : {tweak}");
            let _ = writeln!(out, "vanity scan pubkey: {}", hex::encode(found.pubkey));
            let _ = writeln!(
                out,
                "spend public key  : {} {spend_note}",
                hex::encode(spend)
            );
            let _ = writeln!(out, "address           : {addr}");
            let _ = writeln!(
                out,
                "scan_priv = {}   → run: spaghetti apply --scan-priv <d hex> --tweak {tweak}",
                tweak.formula()
            );
        }
        (_, Key::Secret(secret)) => {
            let _ = writeln!(out, "scan secret key : {}", hex::encode(secret));
            let _ = writeln!(out, "scan public key : {}", hex::encode(found.pubkey));
            let _ = writeln!(out, "spend public key: {} {spend_note}", hex::encode(spend));
            let _ = writeln!(out, "address         : {addr}");
        }
        (Mode::Random, Key::Tweak(tweak)) => {
            let _ = writeln!(out, "internal: unexpected tweak {tweak}");
        }
    }
    let _ = writeln!(
        out,
        "found after {} candidates in {} ({}/s)",
        human_count(tested as f64),
        human_duration(elapsed.as_secs_f64()),
        human_count(rate)
    );
    let _ = writeln!(out);
    let _ = out.flush();
}

/// Decodes a `--address` v0 silent payment address into its network and scan pubkey.
fn parse_address(text: &str) -> Result<(Network, ProjectivePoint), String> {
    let (hrp, scan, _) = address::decode(text.trim()).map_err(|e| format!("--address: {e}"))?;
    let network = if hrp == address::HRP_MAINNET {
        Network::Mainnet
    } else if hrp == address::HRP_TESTNET {
        Network::Testnet
    } else {
        return Err(format!(
            "--address: hrp '{hrp}' is neither sp (mainnet) nor tsp (testnet/signet)"
        ));
    };
    let key = parse_pubkey(&hex::encode(scan), "--address: scan key")?;
    Ok((network, key))
}

fn run_recover(args: RecoverArgs) -> Result<(), String> {
    let (network, target) = parse_address(&args.address)?;
    let base = parse_base_key(&args.base, network)?.ok_or_else(|| {
        "recover needs the base scan pubkey: -b <HEX33> or --xpub <XPUB>".to_string()
    })?;
    check_batch(args.batch)?;
    let params = recover::Params {
        baby_bits: args.baby_bits,
        max_bits: args.max_bits,
        threads: thread_count(args.cores)?,
        half: args.batch,
    };
    if params.max_bits < params.baby_bits || params.max_bits > MAX_TWEAK_BITS {
        return Err(format!(
            "--max-bits must be between --baby-bits and {MAX_TWEAK_BITS}"
        ));
    }
    let started = Instant::now();
    let baby = recover::BabyTable::build(params.baby_bits, params.half)?;
    eprintln!(
        "baby table: 2^{} points in {} | giant steps: up to 2^{} per variant, {} threads",
        params.baby_bits,
        human_duration(started.elapsed().as_secs_f64()),
        params.max_bits - params.baby_bits,
        params.threads
    );
    let mut printed = false;
    let mut report = |p: recover::Progress| {
        let secs = started.elapsed().as_secs_f64();
        printed = true;
        eprint!(
            "\r\x1b[Kvariant {}/6 (e={}, s={}) | {} / {} giant steps ({:.1}%) | elapsed {}",
            p.variant + 1,
            p.endo,
            if p.negate { '-' } else { '+' },
            human_count(p.done as f64),
            human_count(p.total as f64),
            100.0 * p.done as f64 / p.total.max(1) as f64,
            human_duration(secs)
        );
        let _ = std::io::stderr().flush();
    };
    let found = recover::recover(&base, &target, &params, &baby, &mut report)?;
    if printed {
        eprint!("\r\x1b[K");
    }
    let tweak = found.ok_or_else(|| {
        format!(
            "no tweak below 2^{} reproduces this address from this base key (wrong base key / \
             xpub, or the address was not made by spaghetti split-key mode)",
            params.max_bits
        )
    })?;
    let pubkey = search::compressed(&tweak.apply_point(&base));
    if pubkey != search::compressed(&target) {
        return Err("internal: recovered tweak does not reproduce the address".to_string());
    }
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        "base scan pubkey  : {}",
        hex::encode(search::compressed(&base))
    );
    let _ = writeln!(out, "tweak             : {tweak}");
    let _ = writeln!(out, "vanity scan pubkey: {}", hex::encode(pubkey));
    let _ = writeln!(
        out,
        "scan_priv = {}   → run: spaghetti apply --scan-priv <d hex> --tweak {tweak}",
        tweak.formula()
    );
    let _ = writeln!(
        out,
        "recovered in {}",
        human_duration(started.elapsed().as_secs_f64())
    );
    let _ = out.flush();
    Ok(())
}

/// A 32-byte hex secret key as a non-zero scalar; `what` prefixes errors.
fn parse_secret(text: &str, what: &str) -> Result<Scalar, String> {
    let bytes = hex::decode(text.trim()).map_err(|e| format!("{what}: {e}"))?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| format!("{what}: expected 32 bytes of hex"))?;
    let scalar = Scalar::from_repr_vartime(bytes.into())
        .ok_or_else(|| format!("{what}: value is not below the curve order n"))?;
    if scalar == Scalar::ZERO {
        return Err(format!("{what}: the zero key is not a valid secret key"));
    }
    Ok(scalar)
}

fn run_apply(args: &ApplyArgs) -> Result<(), String> {
    let d = parse_secret(&args.scan_priv, "--scan-priv")?;
    let secret = args.tweak.apply(&d);
    if secret == Scalar::ZERO {
        return Err("the tweak maps this key to zero; it cannot be a valid scan key".to_string());
    }
    let pubkey = search::compressed(&(ProjectivePoint::GENERATOR * secret));
    if let Some(addr) = &args.address {
        let (_, scan) = parse_address(addr)?;
        if search::compressed(&scan) != pubkey {
            return Err(format!(
                "the address's scan key {} is not the tweaked key {} (wrong tweak or wrong scan_priv)",
                hex::encode(search::compressed(&scan)),
                hex::encode(pubkey)
            ));
        }
    }
    let secret_bytes: [u8; 32] = secret.to_bytes().into();
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "vanity scan secret key: {}", hex::encode(secret_bytes));
    let _ = writeln!(out, "vanity scan public key: {}", hex::encode(pubkey));
    if let Some(addr) = &args.address {
        let _ = writeln!(
            out,
            "address               : {} (scan key matches)",
            addr.trim()
        );
    }
    let _ = out.flush();
    Ok(())
}

fn print_progress(tested: u64, expected: f64, elapsed: Duration) {
    let secs = elapsed.as_secs_f64().max(1e-9);
    let rate = tested as f64 / secs;
    let remaining = (expected - tested as f64).max(0.0);
    let eta = if rate > 0.0 {
        human_duration(remaining / rate)
    } else {
        "∞".to_string()
    };
    eprint!(
        "\r\x1b[K{}/s | tested {} | expected {} | ETA {eta} | elapsed {}",
        human_count(rate),
        human_count(tested as f64),
        human_count(expected),
        human_duration(secs)
    );
    let _ = std::io::stderr().flush();
}

fn human_count(value: f64) -> String {
    const UNITS: [&str; 7] = ["", " K", " M", " G", " T", " P", " E"];
    if !value.is_finite() {
        return "∞".to_string();
    }
    let mut v = value;
    let mut unit = 0;
    while v >= 1000.0 && unit < UNITS.len() - 1 {
        v /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}{}", UNITS[unit])
    }
}

fn human_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "∞".to_string();
    }
    if seconds < 1.0 {
        return format!("{:.0} ms", seconds * 1000.0);
    }
    if seconds < 60.0 {
        return format!("{seconds:.1}s");
    }
    let total = seconds.round() as u64;
    let (d, h, m, s) = (
        total / 86_400,
        total % 86_400 / 3_600,
        total % 3_600 / 60,
        total % 60,
    );
    if d >= 365 * 1000 {
        format!("{:.1} years", seconds / (365.25 * 86_400.0))
    } else if d > 0 {
        format!("{d}d {h}h {m}m")
    } else if h > 0 {
        format!("{h}h {m}m {s}s")
    } else {
        format!("{m}m {s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bech32::{ByteIterExt, Fe32IterExt};

    #[test]
    fn human_formats() {
        assert_eq!(human_count(0.0), "0");
        assert_eq!(human_count(999.0), "999");
        assert_eq!(human_count(32_768.0), "32.77 K");
        assert_eq!(human_count(1.5e9), "1.50 G");
        assert_eq!(human_duration(0.5), "500 ms");
        assert_eq!(human_duration(1.5), "1.5s");
        assert_eq!(human_duration(65.0), "1m 5s");
        assert_eq!(human_duration(3_661.0), "1h 1m 1s");
        assert_eq!(human_duration(90_000.0), "1d 1h 0m");
        assert!(human_duration(1e12).ends_with("years"));
    }

    #[test]
    fn spend_pubkey_parsing() {
        let (random, provided) = parse_spend_pubkey(None).unwrap();
        assert!(!provided);
        assert!(random[0] == 0x02 || random[0] == 0x03);
        let (same, provided) = parse_spend_pubkey(Some(&hex::encode(random))).unwrap();
        assert!(provided);
        assert_eq!(same, random);
        assert!(parse_spend_pubkey(Some("zz")).is_err());
        assert!(parse_spend_pubkey(Some(&"00".repeat(33))).is_err());
        assert!(parse_pubkey(&"02".repeat(32), "x").is_err());
    }

    /// BIP32 test vector 1 node m/0H/1/2H (depth 3, child 2').
    const VECTOR_XPUB: &str = "xpub6D4BDPcP2GT577Vvch3R8wDkScZWzQzMMUm3PWbmWvVJrZwQY4VUNgqFJPMM3No2dFDFGTsxxpG5uJh7n7epu4trkrX7x7DogT5Uv6fcLW5";
    /// The same key and chain code re-serialised with depth 4 and child 1'
    /// (the header of an `m/352'/coin'/account'/1'` node).
    const ACCOUNT_XPUB: &str = "xpub6EwK5B8QEa84vLJR7ik6SXv8J5uvxFG5UqdZGqfwQWNqhQfKEd1enZhemimbo7gZw3GJMvfAJsqMYBDsBZHpmBr5j5sECGixfcyhTb4B9jY";

    #[test]
    fn base_key_sources() {
        let none = BaseKeyArgs {
            base_pubkey: None,
            xpub: None,
        };
        assert!(parse_base_key(&none, Network::Mainnet).unwrap().is_none());
        let wrong_node = BaseKeyArgs {
            base_pubkey: None,
            xpub: Some(VECTOR_XPUB.to_string()),
        };
        let err = parse_base_key(&wrong_node, Network::Mainnet).unwrap_err();
        assert!(err.contains("depth 3"), "{err}");
        let from_xpub = BaseKeyArgs {
            base_pubkey: None,
            xpub: Some(ACCOUNT_XPUB.to_string()),
        };
        let point = parse_base_key(&from_xpub, Network::Mainnet)
            .unwrap()
            .unwrap();
        assert_eq!(
            search::compressed(&point),
            bip32::derive_child(VECTOR_XPUB, 0).unwrap().0
        );
        let err = parse_base_key(&from_xpub, Network::Testnet).unwrap_err();
        assert!(err.contains("mainnet"), "{err}");
        let from_hex = BaseKeyArgs {
            base_pubkey: Some(hex::encode(search::compressed(&point))),
            xpub: None,
        };
        let same = parse_base_key(&from_hex, Network::Testnet)
            .unwrap()
            .unwrap();
        assert_eq!(same, point);
    }

    #[test]
    fn address_parsing() {
        const VECTOR: &str = "sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwv";
        let (network, scan) = parse_address(VECTOR).unwrap();
        assert_eq!(network, Network::Mainnet);
        assert!(hex::encode(search::compressed(&scan)).starts_with("02"));
        let err = parse_address("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4").unwrap_err();
        assert!(err.starts_with("--address:"), "{err}");
        let err = parse_address("sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwx").unwrap_err();
        assert!(err.starts_with("--address:"), "{err}");
        // A scan-key-only payload (33 bytes) is not an address.
        let short: String = [2u8; 33]
            .iter()
            .copied()
            .bytes_to_fes()
            .with_checksum::<bech32::Bech32m>(&address::HRP_MAINNET)
            .with_witness_version(bech32::Fe32::Q)
            .chars()
            .collect();
        let err = parse_address(&short).unwrap_err();
        assert!(
            err.starts_with("--address:") && err.contains("33 bytes"),
            "{err}"
        );
    }

    #[test]
    fn secret_parsing() {
        assert!(parse_secret(&"00".repeat(32), "x").is_err());
        assert!(parse_secret(&"ff".repeat(32), "x").is_err());
        let err = parse_secret("01", "--scan-priv").unwrap_err();
        assert!(err.starts_with("--scan-priv:"), "{err}");
        assert_eq!(
            parse_secret(&format!("{}01", "00".repeat(31)), "x").unwrap(),
            Scalar::ONE
        );
    }

    #[test]
    fn thread_and_batch_limits() {
        assert_eq!(thread_count(Some(MAX_THREADS)).unwrap(), MAX_THREADS);
        assert!(thread_count(Some(0)).unwrap() >= 1);
        let err = thread_count(Some(MAX_THREADS + 1)).unwrap_err();
        assert!(err.contains("1024"), "{err}");
        assert!(check_batch(1).is_ok());
        assert!(check_batch(MAX_BATCH_HALF).is_ok());
        assert!(check_batch(0).unwrap_err().contains("--batch"));
        assert!(check_batch(MAX_BATCH_HALF + 1).is_err());
    }

    /// Split mode with every range already taken: the worker exits without
    /// reporting anything, which is what makes the main loop fail cleanly.
    #[test]
    fn split_worker_exits_when_ranges_are_exhausted() {
        let table = Table::new(4, ProjectivePoint::GENERATOR);
        let patterns = PatternSet::parse(&["sp1qq".to_string()], Network::Mainnet).unwrap();
        let mode = Mode::Split {
            base: ProjectivePoint::GENERATOR,
        };
        let ranges = AtomicUsize::new(SPLIT_RANGES);
        let (sender, receiver) = mpsc::channel();
        worker_loop(
            &table,
            &patterns,
            &mode,
            &ranges,
            &AtomicBool::new(false),
            &AtomicU64::new(0),
            &sender,
        );
        drop(sender);
        assert!(receiver.recv().is_err());
    }

    /// Random mode reseeds after every hit: keys reported by one worker are
    /// not related by a small offset (in any of the six `±λ^e` variants), so
    /// an observer cannot link them and one secret does not yield another.
    #[test]
    fn random_hits_from_one_worker_are_unlinkable() {
        let table = Table::new(64, ProjectivePoint::GENERATOR);
        let patterns = PatternSet::parse(&["sp1qq?q".to_string()], Network::Mainnet).unwrap();
        let stop = AtomicBool::new(false);
        let tested = AtomicU64::new(0);
        let ranges = AtomicUsize::new(0);
        let (sender, receiver) = mpsc::channel();
        let mut secrets = Vec::new();
        thread::scope(|scope| {
            scope.spawn(|| {
                worker_loop(
                    &table,
                    &patterns,
                    &Mode::Random,
                    &ranges,
                    &stop,
                    &tested,
                    &sender,
                )
            });
            for _ in 0..4 {
                let found = receiver.recv().unwrap().unwrap();
                let Key::Secret(bytes) = found.key else {
                    panic!("expected a secret");
                };
                let secret = Scalar::from_repr_vartime(bytes.into()).unwrap();
                assert_eq!(
                    search::compressed(&(ProjectivePoint::GENERATOR * secret)),
                    found.pubkey
                );
                secrets.push(secret);
            }
            stop.store(true, Ordering::Relaxed);
        });
        let far = |d: Scalar| search::scalar_to_u64(&d).is_none_or(|t| t >= 1 << 40);
        for (i, a) in secrets.iter().enumerate() {
            for b in &secrets[i + 1..] {
                for endo in 0..3u8 {
                    for negate in [false, true] {
                        let variant = Tweak { t: 0, endo, negate }.apply(a);
                        assert!(far(b.sub(&variant)) && far(variant.sub(b)), "linked keys");
                    }
                }
            }
        }
    }

    /// `apply` on a search result reproduces the found key, and the final
    /// check accepts exactly that pubkey.
    #[test]
    fn split_final_check_and_apply() {
        let d = parse_secret(&format!("{}2a", "00".repeat(31)), "x").unwrap();
        let base = ProjectivePoint::GENERATOR * d;
        let tweak: Tweak = "12345/2/-".parse().unwrap();
        let pubkey = search::compressed(&tweak.apply_point(&base));
        let found = Found {
            key: Key::Tweak(tweak),
            pubkey,
            pattern: 0,
        };
        let pattern = Pattern::parse("sp1qq", Network::Mainnet).unwrap();
        let addr = address::encode(address::HRP_MAINNET, &pubkey, &[2u8; 33]);
        let mode = Mode::Split { base };
        final_check(&addr, &found, &pattern, &mode).unwrap();
        let other = Mode::Split {
            base: ProjectivePoint::GENERATOR,
        };
        assert!(final_check(&addr, &found, &pattern, &other).is_err());
        assert!(final_check(&addr, &found, &pattern, &Mode::Random).is_err());
        let secret = tweak.apply(&d);
        assert_eq!(
            search::compressed(&(ProjectivePoint::GENERATOR * secret)),
            pubkey
        );
    }

    /// Random mode: the final check re-derives the pubkey from the printed
    /// secret and rejects a secret that does not produce the address's key.
    #[test]
    fn random_final_check_rederives_pubkey() {
        let k = parse_secret(&format!("{}2a", "00".repeat(31)), "x").unwrap();
        let pubkey = search::compressed(&(ProjectivePoint::GENERATOR * k));
        let pattern = Pattern::parse("sp1qq", Network::Mainnet).unwrap();
        let addr = address::encode(address::HRP_MAINNET, &pubkey, &[2u8; 33]);
        let found = Found {
            key: Key::Secret(k.to_bytes().into()),
            pubkey,
            pattern: 0,
        };
        final_check(&addr, &found, &pattern, &Mode::Random).unwrap();
        let wrong = Found {
            key: Key::Secret(k.add(&Scalar::ONE).to_bytes().into()),
            pubkey,
            pattern: 0,
        };
        let err = final_check(&addr, &wrong, &pattern, &Mode::Random).unwrap_err();
        assert!(err.contains("secret key"), "{err}");
        let zero = Found {
            key: Key::Secret([0u8; 32]),
            pubkey,
            pattern: 0,
        };
        assert!(final_check(&addr, &zero, &pattern, &Mode::Random).is_err());
    }

    #[test]
    fn cli_shape() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        let cli = Cli::try_parse_from(["spaghetti", "pasta"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.patterns, vec!["pasta"]);
        let cli = Cli::try_parse_from(["spaghetti", "-b", "02aa", "pasta"]).unwrap();
        assert_eq!(cli.base.base_pubkey.as_deref(), Some("02aa"));
        assert!(Cli::try_parse_from(["spaghetti", "-b", "02aa", "--xpub", "x", "pasta"]).is_err());
        assert!(Cli::try_parse_from(["spaghetti"]).is_err());
        let cli =
            Cli::try_parse_from(["spaghetti", "recover", "--address", "sp1", "-b", "02"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Recover(_))));
        let cli = Cli::try_parse_from([
            "spaghetti",
            "apply",
            "--scan-priv",
            "aa",
            "--tweak",
            "1/0/+",
        ])
        .unwrap();
        assert!(matches!(cli.command, Some(Command::Apply(_))));
        assert!(
            Cli::try_parse_from([
                "spaghetti",
                "apply",
                "--scan-priv",
                "aa",
                "--tweak",
                "1/9/+"
            ])
            .is_err()
        );
    }
}
