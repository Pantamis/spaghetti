//! `spaghetti`: find a BIP352 silent payment scan key whose addresses start
//! with a chosen prefix.

mod address;
mod field;
mod pattern;
mod search;

use std::io::Write;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use k256::elliptic_curve::Generate;
use k256::{NonZeroScalar, PublicKey};

use pattern::{Network, Pattern, PatternSet};
use search::{Found, Table, Worker};

#[derive(Clone, Copy, ValueEnum)]
enum NetworkArg {
    /// hrp `sp`
    Mainnet,
    /// hrp `tsp` (also used for signet and regtest)
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
#[command(version, about, long_about = None)]
struct Cli {
    /// address prefix, full (`sp1qq?pasta`) or bare (`pasta` = `sp1qq?pasta`); `?` = any char
    #[arg(required = true, value_name = "PATTERN")]
    patterns: Vec<String>,

    /// mainnet | testnet (testnet hrp `tsp` is also used for signet/regtest)
    #[arg(short, long, value_enum, default_value_t = NetworkArg::Mainnet)]
    network: NetworkArg,

    /// threads [default: available_parallelism]
    #[arg(short, long, value_name = "N")]
    cores: Option<usize>,

    /// stop after N matches
    #[arg(short = 'k', long, value_name = "N", default_value_t = 1)]
    count: u64,

    /// spend public key used to render the example address (random throwaway one if omitted)
    #[arg(short, long, value_name = "HEX33")]
    spend_pubkey: Option<String>,

    /// no progress output
    #[arg(short, long)]
    quiet: bool,

    /// half-batch size H (2H points per inversion)
    #[arg(long, value_name = "N", default_value_t = 1024, hide = true)]
    batch: usize,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
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
            let bytes = hex::decode(text.trim()).map_err(|e| format!("--spend-pubkey: {e}"))?;
            let key = PublicKey::from_sec1_bytes(&bytes)
                .map_err(|_| "--spend-pubkey: not a valid secp256k1 public key".to_string())?;
            Ok((search::compressed(&key.to_projective()), true))
        }
        None => {
            let scalar = NonZeroScalar::try_generate().map_err(|e| format!("rng failure: {e}"))?;
            let key = PublicKey::from_secret_scalar(&scalar);
            Ok((search::compressed(&key.to_projective()), false))
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let network: Network = cli.network.into();
    let patterns = PatternSet::parse(&cli.patterns, network)?;
    let (spend, spend_provided) = parse_spend_pubkey(cli.spend_pubkey.as_deref())?;
    let hrp = match network {
        Network::Mainnet => address::HRP_MAINNET,
        Network::Testnet => address::HRP_TESTNET,
    };
    let cores = match cli.cores {
        Some(0) | None => thread::available_parallelism().map_or(1, |n| n.get()),
        Some(n) => n,
    };
    if cli.count == 0 {
        return Err("--count must be at least 1".to_string());
    }
    let table = Table::new(cli.batch);
    let expected = patterns.expected_candidates();
    let quiet = cli.quiet;

    if !quiet {
        let names: Vec<&str> = patterns.patterns.iter().map(|p| p.text.as_str()).collect();
        eprintln!(
            "searching {} | difficulty 2^{} ≈ {} candidates | {cores} threads, batch {}",
            names.join(" | "),
            patterns.patterns.iter().map(|p| p.bits).min().unwrap_or(0),
            human_count(expected),
            2 * table.half
        );
    }

    let stop = AtomicBool::new(false);
    let tested = AtomicU64::new(0);
    let (sender, receiver) = mpsc::channel::<Result<Found, String>>();
    let started = Instant::now();
    let mut found_count = 0u64;
    let mut result = Ok(());

    thread::scope(|scope| {
        for _ in 0..cores {
            let sender = sender.clone();
            let (table, patterns, stop, tested) = (&table, &patterns, &stop, &tested);
            scope.spawn(move || worker_loop(table, patterns, stop, tested, &sender));
        }
        drop(sender);

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
                    if let Err(message) = final_check(&addr, &found.pubkey, pattern) {
                        result = Err(message);
                        break;
                    }
                    print_match(&found, &spend, spend_provided, &addr, n, elapsed);
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
                    result = Err("all workers exited".to_string());
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

/// Independent re-check of a match: the rendered address must carry the
/// requested prefix and decode (bech32m) back to the scan key.
fn final_check(addr: &str, pubkey: &[u8; 33], pattern: &Pattern) -> Result<(), String> {
    if !pattern.matches_address(addr) {
        return Err(format!(
            "internal: address {addr} does not match {}",
            pattern.text
        ));
    }
    let (_, version, payload) = address::decode(addr)?;
    if version != bech32::Fe32::Q || payload.get(..33) != Some(&pubkey[..]) {
        return Err(format!(
            "internal: address {addr} does not decode to the scan key"
        ));
    }
    Ok(())
}

fn worker_loop(
    table: &Table,
    patterns: &PatternSet,
    stop: &AtomicBool,
    tested: &AtomicU64,
    sender: &mpsc::Sender<Result<Found, String>>,
) {
    let mut worker = match Worker::new(table) {
        Ok(worker) => worker,
        Err(message) => {
            let _ = sender.send(Err(message));
            return;
        }
    };
    let mut hits = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        if worker.search_batch(patterns, &mut hits) {
            tested.fetch_add(table.candidates_per_batch(), Ordering::Relaxed);
        }
        for candidate in hits.drain(..) {
            if sender.send(search::resolve(&candidate, patterns)).is_err() {
                return;
            }
        }
    }
}

fn print_match(
    found: &Found,
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
    let _ = writeln!(out, "scan secret key : {}", hex::encode(found.secret));
    let _ = writeln!(out, "scan public key : {}", hex::encode(found.pubkey));
    let _ = writeln!(out, "spend public key: {} {spend_note}", hex::encode(spend));
    let _ = writeln!(out, "address         : {addr}");
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
    }
}
