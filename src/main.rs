//! `spaghetti`: find a BIP352 silent payment scan key whose addresses start
//! with a chosen prefix.
//!
//! Command-line parsing and orchestration only; the search lives in
//! `search`, tweak recovery in `recover`.

mod address;
mod bip32;
mod field;
mod pattern;
mod recover;
mod search;
mod tweak;

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use k256::elliptic_curve::PrimeField;
use k256::{ProjectivePoint, PublicKey, Scalar};
use zeroize::Zeroizing;

use address::Network;
use pattern::PatternSet;
use search::{Found, Key, Mode, RANGE_BITS, SPLIT_RANGES, Shared, Table, compressed};
use tweak::{MAX_TWEAK_BITS, Tweak};

/// Upper bound on `-c` (OS threads).
const MAX_THREADS: usize = 1024;
/// Upper bound on the hidden `--batch` half size `H` (`2H <= 2^20`).
const MAX_BATCH_HALF: usize = 1 << 19;

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

    /// mainnet (hrp `sp`) | testnet, signet (hrp `tsp`, the BIP352 hrp for both) |
    /// regtest (hrp `sprt`)
    #[arg(short, long, value_enum, default_value_t = Network::Mainnet)]
    network: Network,

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

    /// write each match, secret included, to this new file (mode 0600, never overwritten)
    /// and print it without the secret line
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// no progress output
    #[arg(short, long)]
    quiet: bool,

    /// half-batch size H (2H points per inversion)
    #[arg(long, value_name = "N", default_value_t = 4096, hide = true)]
    batch: usize,
}

/// Where the base scan public key `D` of split-key mode comes from.
#[derive(Args)]
struct BaseKeyArgs {
    /// split-key mode: search offsets from this compressed scan pubkey D
    #[arg(short = 'b', long, value_name = "HEX33", conflicts_with = "xpub")]
    base_pubkey: Option<String>,

    /// split-key mode: D = child 0 (non-hardened) of this extended pubkey; give the node
    /// m/352'/0'/0'/1' (testnet, signet, regtest: m/352'/1'/0'/1', tpub). Mutually exclusive with -b.
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
    /// file holding the hex BIP32-derived scan secret key d (m/352'/coin'/account'/1'/0); `-` = stdin
    #[arg(long, value_name = "PATH")]
    scan_priv_file: PathBuf,

    /// tweak string `<t>/<e>/<s>` printed by the search or by `recover`
    #[arg(long, value_name = "t/e/s")]
    tweak: Tweak,

    /// address to check against the resulting scan pubkey
    #[arg(long, value_name = "SP_ADDRESS")]
    address: Option<String>,

    /// write the result, secret included, to this new file (mode 0600, never overwritten)
    /// and print it without the secret line
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome = match cli.command {
        Some(Command::Recover(args)) => run_recover(&args),
        Some(Command::Apply(args)) => run_apply(&args),
        None => Search::from_args(cli).and_then(|search| search.run()),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// A validated key search (default or split-key mode).
struct Search {
    patterns: PatternSet,
    network: Network,
    mode: Mode,
    table: Table,
    threads: usize,
    count: u64,
    spend: [u8; 33],
    spend_provided: bool,
    output: Option<PathBuf>,
    quiet: bool,
}

impl Search {
    fn from_args(cli: Cli) -> Result<Search, String> {
        let network = cli.network;
        let patterns = PatternSet::parse(&cli.patterns, network)?;
        let spend = match &cli.spend_pubkey {
            Some(text) => parse_pubkey(text, "--spend-pubkey")?,
            None => ProjectivePoint::GENERATOR * search::random_scalar()?,
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
        if cli.output.is_some() && matches!(mode, Mode::Split { .. }) {
            return Err(
                "--output: split-key mode prints no secret (the tweak is public); nothing to write"
                    .to_string(),
            );
        }
        Ok(Search {
            patterns,
            network,
            mode,
            table: Table::new(cli.batch, ProjectivePoint::GENERATOR),
            threads,
            count: cli.count,
            spend: compressed(&spend),
            spend_provided: cli.spend_pubkey.is_some(),
            output: cli.output,
            quiet: cli.quiet,
        })
    }

    fn expected(&self) -> f64 {
        self.patterns.expected_candidates()
    }

    fn announce(&self) {
        let names: Vec<&str> = self
            .patterns
            .patterns
            .iter()
            .map(|p| p.text.as_str())
            .collect();
        eprintln!(
            "searching {} | difficulty 2^{} ≈ {} candidates | {} threads, batch {}{}",
            names.join(" | "),
            self.patterns
                .patterns
                .iter()
                .map(|p| p.bits)
                .min()
                .unwrap_or(0),
            human_count(self.expected()),
            self.threads,
            2 * self.table.half,
            if matches!(self.mode, Mode::Split { .. }) {
                format!(" | split-key mode ({SPLIT_RANGES} ranges of 2^{RANGE_BITS} offsets)")
            } else {
                String::new()
            }
        );
        // Split-key mode covers every offset below 2^52 whatever the thread
        // count: warn when the pattern is expected to need more than a quarter.
        let coverage = search::split_coverage(self.table.half);
        if matches!(self.mode, Mode::Split { .. }) && self.expected() * 4.0 > coverage {
            eprintln!(
                "warning: split-key mode covers at most ≈ {} candidates (every offset below \
                 2^{MAX_TWEAK_BITS}) and the pattern needs ≈ {} on average: the search will likely \
                 exhaust all offsets without a match; use a shorter pattern",
                human_count(coverage),
                human_count(self.expected())
            );
        }
    }

    /// Spawns the workers and prints matches until `count` of them are in.
    fn run(&self) -> Result<(), String> {
        let mut output = match &self.output {
            Some(path) => Some((path.as_path(), create_secret_file(path)?)),
            None => None,
        };
        if !self.quiet {
            self.announce();
        }
        let shared = Shared::new(self.threads);
        let (sender, receiver) = mpsc::channel();
        let started = Instant::now();
        let mut result = Ok(());
        thread::scope(|scope| {
            for thread in 0..self.threads {
                let sender = sender.clone();
                let (table, patterns, mode, shared) =
                    (&self.table, &self.patterns, &self.mode, &shared);
                let spawned = thread::Builder::new().spawn_scoped(scope, move || {
                    search::worker(thread, table, patterns, mode, shared, &sender)
                });
                if let Err(e) = spawned {
                    result = Err(format!(
                        "could not spawn worker thread {} of {}: {e}; lower -c",
                        thread + 1,
                        self.threads
                    ));
                    break;
                }
            }
            drop(sender);
            if result.is_ok() {
                result = self.collect(&receiver, &shared, started, &mut output);
            }
            shared.stop.store(true, Ordering::Relaxed);
        });
        result
    }

    /// Receives, verifies and prints matches; shows progress once a second.
    fn collect(
        &self,
        receiver: &mpsc::Receiver<Result<Found, String>>,
        shared: &Shared,
        started: Instant,
        output: &mut Option<(&Path, File)>,
    ) -> Result<(), String> {
        let mut status = StatusLine::default();
        let mut last_progress = Instant::now();
        let mut found_count = 0;
        let result = loop {
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(found)) => {
                    let elapsed = started.elapsed();
                    let tested = shared.tested();
                    status.clear();
                    let addr = address::encode(self.network.hrp(), &found.pubkey, &self.spend);
                    if let Err(message) = final_check(&addr, &found, &self.patterns) {
                        break Err(message);
                    }
                    let matched = Match {
                        found: &found,
                        spend: &self.spend,
                        spend_provided: self.spend_provided,
                        addr: &addr,
                        tested,
                        elapsed,
                    };
                    let mut out = io::stdout().lock();
                    match output {
                        Some((path, file)) => {
                            matched
                                .write(file, SecretLine::Show)
                                .and_then(|()| file.flush())
                                .map_err(|e| format!("--output: {}: {e}", path.display()))?;
                            let _ = matched.write(&mut out, SecretLine::WrittenTo(path));
                        }
                        None => {
                            let _ = matched.write(&mut out, SecretLine::Show);
                        }
                    }
                    found_count += 1;
                    if found_count >= self.count {
                        break Ok(());
                    }
                }
                Ok(Err(message)) => break Err(message),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(match self.mode {
                        Mode::Random => "all workers exited".to_string(),
                        Mode::Split { .. } => format!(
                            "split-key mode: all {SPLIT_RANGES} ranges of 2^{RANGE_BITS} offsets \
                             (every offset below 2^{MAX_TWEAK_BITS}) exhausted without a match; \
                             use a shorter pattern"
                        ),
                    });
                }
            }
            if !self.quiet && last_progress.elapsed() >= Duration::from_secs(1) {
                last_progress = Instant::now();
                status.show(&progress_text(
                    shared.tested(),
                    self.expected(),
                    started.elapsed(),
                ));
            }
        };
        status.clear();
        result
    }
}

/// Independent re-check of a match: the rendered address must carry a
/// requested prefix and decode (bech32m) back to the scan key, and the printed
/// key material, re-parsed from its text form, must reproduce the scan key
/// (random mode: `secret·G`; split-key mode: the tweak applied to the base key).
fn final_check(addr: &str, found: &Found, patterns: &PatternSet) -> Result<(), String> {
    if !patterns.patterns.iter().any(|p| p.matches_address(addr)) {
        return Err(format!(
            "internal: address {addr} does not match the pattern"
        ));
    }
    let (_, scan, _) = address::decode(addr).map_err(|e| format!("internal: {e}"))?;
    if scan != found.pubkey {
        return Err(format!(
            "internal: address {addr} does not decode to the scan key"
        ));
    }
    let (what, reproduced) = match &found.key {
        Key::Secret(secret) => {
            let secret = parse_secret(&secret_hex(secret), "internal: printed secret key")?;
            (
                "secret key".to_string(),
                ProjectivePoint::GENERATOR * *secret,
            )
        }
        Key::Tweak { base, tweak } => {
            let base = parse_pubkey(&hex::encode(compressed(base)), "internal: printed base key")?;
            let tweak: Tweak = tweak.to_string().parse()?;
            (format!("tweak {tweak}"), tweak.apply_point(&base))
        }
    };
    if compressed(&reproduced) != found.pubkey {
        return Err(format!(
            "internal: printed {what} does not reproduce the scan key"
        ));
    }
    Ok(())
}

/// Where the secret line of a match goes.
enum SecretLine<'a> {
    /// Printed in place.
    Show,
    /// Replaced by a pointer to the `--output` file that holds it.
    WrittenTo(&'a Path),
}

impl SecretLine<'_> {
    /// The text after the label: the hex secret, or where it went.
    fn render(&self, secret: &Scalar) -> Zeroizing<String> {
        match self {
            SecretLine::Show => secret_hex(secret),
            SecretLine::WrittenTo(path) => Zeroizing::new(format!("written to {}", path.display())),
        }
    }
}

/// Hex of a secret scalar, wiped on drop.
fn secret_hex(secret: &Scalar) -> Zeroizing<String> {
    let bytes = Zeroizing::new(secret.to_bytes());
    Zeroizing::new(hex::encode(bytes.as_slice()))
}

/// A verified match and the run statistics printed with it.
struct Match<'a> {
    found: &'a Found,
    spend: &'a [u8; 33],
    spend_provided: bool,
    addr: &'a str,
    tested: u64,
    elapsed: Duration,
}

impl Match<'_> {
    /// The match block: key lines, address, then the run statistics.
    fn write(&self, out: &mut dyn Write, secret: SecretLine) -> io::Result<()> {
        let spend_note = if self.spend_provided {
            "(provided)"
        } else {
            "(example, random)"
        };
        let spend = hex::encode(self.spend);
        match &self.found.key {
            Key::Tweak { base, tweak } => {
                writeln!(out, "base scan pubkey  : {}", hex::encode(compressed(base)))?;
                writeln!(out, "tweak             : {tweak}")?;
                writeln!(
                    out,
                    "vanity scan pubkey: {}",
                    hex::encode(self.found.pubkey)
                )?;
                writeln!(out, "spend public key  : {spend} {spend_note}")?;
                writeln!(out, "address           : {}", self.addr)?;
                writeln!(
                    out,
                    "scan_priv = {}   → run: spaghetti apply --scan-priv-file <d hex file> --tweak {tweak}",
                    tweak.formula()
                )?;
            }
            Key::Secret(key) => {
                writeln!(out, "scan secret key : {}", *secret.render(key))?;
                writeln!(out, "scan public key : {}", hex::encode(self.found.pubkey))?;
                writeln!(out, "spend public key: {spend} {spend_note}")?;
                writeln!(out, "address         : {}", self.addr)?;
            }
        }
        let rate = self.tested as f64 / self.elapsed.as_secs_f64().max(1e-9);
        writeln!(
            out,
            "found after {} candidates in {} ({}/s)",
            human_count(self.tested as f64),
            human_duration(self.elapsed.as_secs_f64()),
            human_count(rate)
        )?;
        writeln!(out)?;
        out.flush()
    }
}

/// Creates the `--output` file: new (never overwritten), owner read/write only.
fn create_secret_file(path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => format!(
            "--output: {} already exists; refusing to overwrite it",
            path.display()
        ),
        _ => format!("--output: {}: {e}", path.display()),
    })
}

fn run_recover(args: &RecoverArgs) -> Result<(), String> {
    let (network, target) = parse_address(&args.address)?;
    let base = parse_base_key(&args.base, network)?.ok_or_else(|| {
        "recover needs the base scan pubkey: -b <HEX33> or --xpub <XPUB>".to_string()
    })?;
    check_batch(args.batch)?;
    let threads = thread_count(args.cores)?;
    if args.max_bits < args.baby_bits || args.max_bits > MAX_TWEAK_BITS {
        return Err(format!(
            "--max-bits must be between --baby-bits and {MAX_TWEAK_BITS}"
        ));
    }
    let started = Instant::now();
    let baby = recover::BabyTable::build(args.baby_bits, args.batch)?;
    eprintln!(
        "baby table: 2^{} points in {} | giant steps: up to 2^{} per variant, {threads} threads",
        args.baby_bits,
        human_duration(started.elapsed().as_secs_f64()),
        args.max_bits - args.baby_bits,
    );
    let params = recover::Params {
        max_bits: args.max_bits,
        threads,
        half: args.batch,
    };
    let mut status = StatusLine::default();
    let found = recover::recover(&base, &target, &params, &baby, &mut |p| {
        status.show(&format!(
            "variant {}/6 (e={}, s={}) | {} / {} giant steps ({:.1}%) | elapsed {}",
            p.variant + 1,
            p.endo,
            if p.negate { '-' } else { '+' },
            human_count(p.done as f64),
            human_count(p.total as f64),
            100.0 * p.done as f64 / p.total.max(1) as f64,
            human_duration(started.elapsed().as_secs_f64())
        ));
    });
    status.clear();
    let tweak = found.ok_or_else(|| {
        format!(
            "no tweak below 2^{} reproduces this address from this base key (wrong base key / \
             xpub, or the address was not made by spaghetti split-key mode)",
            args.max_bits
        )
    })?;
    let pubkey = compressed(&tweak.apply_point(&base));
    if pubkey != compressed(&target) {
        return Err("internal: recovered tweak does not reproduce the address".to_string());
    }
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        "base scan pubkey  : {}",
        hex::encode(compressed(&base))
    );
    let _ = writeln!(out, "tweak             : {tweak}");
    let _ = writeln!(out, "vanity scan pubkey: {}", hex::encode(pubkey));
    let _ = writeln!(
        out,
        "scan_priv = {}   → run: spaghetti apply --scan-priv-file <d hex file> --tweak {tweak}",
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

fn run_apply(args: &ApplyArgs) -> Result<(), String> {
    let d = read_secret_file(&args.scan_priv_file)?;
    let secret = Zeroizing::new(args.tweak.apply(&d));
    if *secret == Scalar::ZERO {
        return Err("the tweak maps this key to zero; it cannot be a valid scan key".to_string());
    }
    let pubkey = compressed(&(ProjectivePoint::GENERATOR * *secret));
    if let Some(addr) = &args.address {
        let (_, scan) = parse_address(addr)?;
        if compressed(&scan) != pubkey {
            return Err(format!(
                "the address's scan key {} is not the tweaked key {} (wrong tweak or wrong scan_priv)",
                hex::encode(compressed(&scan)),
                hex::encode(pubkey)
            ));
        }
    }
    let write = |out: &mut dyn Write, secret_line: SecretLine| -> io::Result<()> {
        writeln!(
            out,
            "vanity scan secret key: {}",
            *secret_line.render(&secret)
        )?;
        writeln!(out, "vanity scan public key: {}", hex::encode(pubkey))?;
        if let Some(addr) = &args.address {
            writeln!(
                out,
                "address               : {} (scan key matches)",
                addr.trim()
            )?;
        }
        out.flush()
    };
    let mut out = io::stdout().lock();
    match &args.output {
        Some(path) => {
            let mut file = create_secret_file(path)?;
            write(&mut file, SecretLine::Show)
                .map_err(|e| format!("--output: {}: {e}", path.display()))?;
            let _ = write(&mut out, SecretLine::WrittenTo(path));
        }
        None => {
            let _ = write(&mut out, SecretLine::Show);
        }
    }
    Ok(())
}

/// The hex secret key in `--scan-priv-file` (`-` = stdin), trimmed.
fn read_secret_file(path: &Path) -> Result<Zeroizing<Scalar>, String> {
    let what = "--scan-priv-file";
    // One allocation for a 64-char hex line plus whitespace: no reallocation
    // leaves an unwiped copy behind.
    let mut bytes = Zeroizing::new(Vec::with_capacity(256));
    let read = if path == Path::new("-") {
        io::stdin().lock().read_to_end(&mut bytes)
    } else {
        File::open(path).and_then(|mut file| file.read_to_end(&mut bytes))
    };
    read.map_err(|e| format!("{what}: {}: {e}", path.display()))?;
    let text = std::str::from_utf8(&bytes).map_err(|_| format!("{what}: not UTF-8 text"))?;
    parse_secret(text, what)
}

// ---- argument parsing -------------------------------------------------------

/// A 33-byte hex compressed public key; `what` prefixes errors.
fn parse_pubkey(text: &str, what: &str) -> Result<ProjectivePoint, String> {
    let bytes = hex::decode(text.trim()).map_err(|e| format!("{what}: {e}"))?;
    let bytes: [u8; 33] = bytes.try_into().map_err(|bytes: Vec<u8>| {
        format!(
            "{what}: expected a 33-byte compressed public key, got {} bytes",
            bytes.len()
        )
    })?;
    point_from_sec1(&bytes, what)
}

fn point_from_sec1(bytes: &[u8; 33], what: &str) -> Result<ProjectivePoint, String> {
    PublicKey::from_sec1_bytes(bytes)
        .map(|key| key.to_projective())
        .map_err(|_| format!("{what}: not a valid secp256k1 public key"))
}

/// Base scan pubkey `D` from `-b` or `--xpub` (clap rejects both), checked
/// against `network`.
fn parse_base_key(args: &BaseKeyArgs, network: Network) -> Result<Option<ProjectivePoint>, String> {
    if let Some(hex_key) = &args.base_pubkey {
        return parse_pubkey(hex_key, "--base-pubkey").map(Some);
    }
    let Some(xpub) = &args.xpub else {
        return Ok(None);
    };
    let (key, version) = bip32::scan_account_pubkey(xpub)?;
    if version != bip32::Version::of(network) {
        return Err(format!(
            "--xpub is {} but the search/address network is {}",
            version.describe(),
            network.describe()
        ));
    }
    Ok(Some(key))
}

/// A `--address` v0 silent payment address as its network and scan pubkey.
fn parse_address(text: &str) -> Result<(Network, ProjectivePoint), String> {
    let (network, scan, _) = address::decode(text.trim()).map_err(|e| format!("--address: {e}"))?;
    Ok((network, point_from_sec1(&scan, "--address: scan key")?))
}

/// A 32-byte hex secret key as a non-zero scalar; `what` prefixes errors.
/// Every intermediate copy of the key is wiped on drop.
fn parse_secret(text: &str, what: &str) -> Result<Zeroizing<Scalar>, String> {
    let bytes = Zeroizing::new(hex::decode(text.trim()).map_err(|e| format!("{what}: {e}"))?);
    let bytes: Zeroizing<[u8; 32]> = bytes
        .as_slice()
        .try_into()
        .map(Zeroizing::new)
        .map_err(|_| format!("{what}: expected 32 bytes of hex"))?;
    let scalar = Scalar::from_repr_vartime((*bytes).into())
        .map(Zeroizing::new)
        .ok_or_else(|| format!("{what}: value is not below the curve order n"))?;
    if *scalar == Scalar::ZERO {
        return Err(format!("{what}: the zero key is not a valid secret key"));
    }
    Ok(scalar)
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

// ---- progress and formatting ------------------------------------------------

/// One stderr line that is overwritten in place.
#[derive(Default)]
struct StatusLine {
    shown: bool,
}

impl StatusLine {
    fn show(&mut self, text: &str) {
        eprint!("\r\x1b[K{text}");
        let _ = std::io::stderr().flush();
        self.shown = true;
    }

    fn clear(&mut self) {
        if self.shown {
            eprint!("\r\x1b[K");
            self.shown = false;
        }
    }
}

fn progress_text(tested: u64, expected: f64, elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64().max(1e-9);
    let rate = tested as f64 / secs;
    let remaining = (expected - tested as f64).max(0.0);
    let eta = if rate > 0.0 {
        human_duration(remaining / rate)
    } else {
        "∞".to_string()
    };
    format!(
        "{}/s | tested {} | expected {} | ETA {eta} | elapsed {}",
        human_count(rate),
        human_count(tested as f64),
        human_count(expected),
        human_duration(secs)
    )
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
    fn pubkey_parsing() {
        let point = ProjectivePoint::GENERATOR * search::random_scalar().unwrap();
        let hex_key = hex::encode(compressed(&point));
        assert_eq!(parse_pubkey(&hex_key, "x").unwrap(), point);
        assert_eq!(parse_pubkey(&format!(" {hex_key}\n"), "x").unwrap(), point);
        let err = parse_pubkey("zz", "--spend-pubkey").unwrap_err();
        assert!(err.starts_with("--spend-pubkey:"), "{err}");
        let err = parse_pubkey(&"02".repeat(32), "x").unwrap_err();
        assert!(err.contains("33-byte") && err.contains("32 bytes"), "{err}");
        let err = parse_pubkey(&"00".repeat(33), "x").unwrap_err();
        assert!(err.contains("not a valid"), "{err}");
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
        assert_eq!(point, bip32::scan_account_pubkey(ACCOUNT_XPUB).unwrap().0);
        for network in [Network::Testnet, Network::Signet, Network::Regtest] {
            let err = parse_base_key(&from_xpub, network).unwrap_err();
            assert!(err.contains("an xpub (mainnet)"), "{err}");
            assert!(err.contains(network.describe()), "{err}");
        }
        let from_hex = BaseKeyArgs {
            base_pubkey: Some(hex::encode(compressed(&point))),
            xpub: None,
        };
        let same = parse_base_key(&from_hex, Network::Regtest)
            .unwrap()
            .unwrap();
        assert_eq!(same, point);
    }

    #[test]
    fn address_parsing() {
        const VECTOR: &str = "sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwv";
        let (network, scan) = parse_address(VECTOR).unwrap();
        assert_eq!(network, Network::Mainnet);
        assert!(hex::encode(compressed(&scan)).starts_with("02"));
        let err = parse_address("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4").unwrap_err();
        assert!(err.starts_with("--address:"), "{err}");
        let err = parse_address("sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwx").unwrap_err();
        assert!(err.starts_with("--address:"), "{err}");
        // A scan-key-only payload (33 bytes) is not an address.
        let short: String = [2u8; 33]
            .iter()
            .copied()
            .bytes_to_fes()
            .with_checksum::<bech32::Bech32m>(&Network::Mainnet.hrp())
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
        let err = parse_secret("01", "--scan-priv-file").unwrap_err();
        assert!(err.starts_with("--scan-priv-file:"), "{err}");
        assert_eq!(
            *parse_secret(&format!("{}01", "00".repeat(31)), "x").unwrap(),
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

    fn pattern_set(text: &str) -> PatternSet {
        PatternSet::parse(&[text.to_string()], Network::Mainnet).unwrap()
    }

    /// Split mode: the final check accepts exactly the printed base and tweak,
    /// and `apply` on the base secret reproduces the found key.
    #[test]
    fn split_final_check_and_apply() {
        let d = parse_secret(&format!("{}2a", "00".repeat(31)), "x").unwrap();
        let base = ProjectivePoint::GENERATOR * *d;
        let tweak: Tweak = "12345/2/-".parse().unwrap();
        let pubkey = compressed(&tweak.apply_point(&base));
        let found = Found {
            key: Key::Tweak { base, tweak },
            pubkey,
        };
        let patterns = pattern_set("sp1qq");
        let addr = address::encode(Network::Mainnet.hrp(), &pubkey, &[2u8; 33]);
        final_check(&addr, &found, &patterns).unwrap();
        let other_base = Found {
            key: Key::Tweak {
                base: ProjectivePoint::GENERATOR,
                tweak,
            },
            pubkey,
        };
        let err = final_check(&addr, &other_base, &patterns).unwrap_err();
        assert!(err.contains("tweak 12345/2/-"), "{err}");
        let other_tweak = Found {
            key: Key::Tweak {
                base,
                tweak: Tweak { t: 12346, ..tweak },
            },
            pubkey,
        };
        assert!(final_check(&addr, &other_tweak, &patterns).is_err());
        let err = final_check(&addr, &found, &pattern_set("sp1qqgq")).unwrap_err();
        assert!(err.contains("does not match"), "{err}");
        let secret = tweak.apply(&d);
        assert_eq!(compressed(&(ProjectivePoint::GENERATOR * secret)), pubkey);
    }

    /// Random mode: the final check re-derives the pubkey from the printed
    /// secret and rejects a secret that does not produce the address's key.
    #[test]
    fn random_final_check_rederives_pubkey() {
        let k = *parse_secret(&format!("{}2a", "00".repeat(31)), "x").unwrap();
        let pubkey = compressed(&(ProjectivePoint::GENERATOR * k));
        let patterns = pattern_set("sp1qq");
        let addr = address::encode(Network::Mainnet.hrp(), &pubkey, &[2u8; 33]);
        let found = Found {
            key: Key::Secret(Zeroizing::new(k)),
            pubkey,
        };
        final_check(&addr, &found, &patterns).unwrap();
        let wrong = Found {
            key: Key::Secret(Zeroizing::new(k.add(&Scalar::ONE))),
            pubkey,
        };
        let err = final_check(&addr, &wrong, &patterns).unwrap_err();
        assert!(err.contains("secret key"), "{err}");
        let zero = Found {
            key: Key::Secret(Zeroizing::new(Scalar::ZERO)),
            pubkey,
        };
        assert!(final_check(&addr, &zero, &patterns).is_err());
        let other_key = Found {
            key: Key::Secret(Zeroizing::new(k)),
            pubkey: compressed(&ProjectivePoint::GENERATOR),
        };
        let err = final_check(&addr, &other_key, &patterns).unwrap_err();
        assert!(err.contains("does not decode"), "{err}");
    }

    #[test]
    fn cli_shape() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        let cli = Cli::try_parse_from(["spaghetti", "pasta"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.patterns, vec!["pasta"]);
        assert_eq!(cli.network, Network::Mainnet);
        let cli =
            Cli::try_parse_from(["spaghetti", "-n", "testnet", "-b", "02aa", "pasta"]).unwrap();
        assert_eq!(cli.network, Network::Testnet);
        assert_eq!(cli.base.base_pubkey.as_deref(), Some("02aa"));
        for (name, network) in [("signet", Network::Signet), ("regtest", Network::Regtest)] {
            let cli = Cli::try_parse_from(["spaghetti", "-n", name, "pasta"]).unwrap();
            assert_eq!(cli.network, network);
        }
        assert!(Cli::try_parse_from(["spaghetti", "-n", "testnet4", "pasta"]).is_err());
        assert!(Cli::try_parse_from(["spaghetti", "-b", "02aa", "--xpub", "x", "pasta"]).is_err());
        assert!(Cli::try_parse_from(["spaghetti"]).is_err());
        let cli =
            Cli::try_parse_from(["spaghetti", "recover", "--address", "sp1", "-b", "02"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Recover(_))));
        let cli = Cli::try_parse_from([
            "spaghetti",
            "apply",
            "--scan-priv-file",
            "-",
            "--tweak",
            "1/0/+",
        ])
        .unwrap();
        assert!(matches!(cli.command, Some(Command::Apply(_))));
        assert!(
            Cli::try_parse_from([
                "spaghetti",
                "apply",
                "--scan-priv-file",
                "-",
                "--tweak",
                "1/9/+"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "spaghetti",
                "apply",
                "--scan-priv",
                "aa",
                "--tweak",
                "1/0/+"
            ])
            .is_err()
        );
    }
}
