//! End-to-end split-key flow through the binary: search from a base key,
//! `apply` the tweak to the base secret, `recover` the tweak from the address.
//! Also the secret-handling surface: `--scan-priv-file` and `--output`.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

const D_HEX: &str = "4f3c0a1b5e6d7c8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5";

fn spaghetti(args: &[&str]) -> (bool, String, String) {
    spaghetti_stdin(args, "")
}

/// Runs the binary with `input` on stdin.
fn spaghetti_stdin(args: &[&str], input: &str) -> (bool, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_spaghetti"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run spaghetti");
    let mut stdin = child.stdin.take().expect("piped stdin");
    stdin.write_all(input.as_bytes()).expect("write stdin");
    drop(stdin);
    let output = child.wait_with_output().expect("wait for spaghetti");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// A fresh path in the temp dir; the file is removed when the guard drops.
struct TempPath(PathBuf);

impl TempPath {
    fn new(tag: &str) -> TempPath {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        TempPath(std::env::temp_dir().join(format!("spaghetti-{}-{}-{tag}", std::process::id(), n)))
    }

    fn with_content(tag: &str, content: &str) -> TempPath {
        let path = TempPath::new(tag);
        std::fs::write(&path.0, content).expect("write temp file");
        path
    }

    fn as_str(&self) -> &str {
        self.0.to_str().expect("utf-8 temp path")
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `apply` with the secret on stdin.
fn apply(secret: &str, rest: &[&str]) -> (bool, String, String) {
    let mut args = vec!["apply", "--scan-priv-file", "-"];
    args.extend_from_slice(rest);
    spaghetti_stdin(&args, secret)
}

/// `label: value` lines of a stdout block.
fn fields(stdout: &str) -> HashMap<String, String> {
    stdout
        .lines()
        .filter_map(|line| {
            let (label, value) = line.split_once(':')?;
            Some((label.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

#[test]
fn search_apply_recover_roundtrip() {
    // Base pubkey D = d·G via `apply` with the neutral tweak, secret from a file.
    let secret_file = TempPath::with_content("d.hex", &format!("  {D_HEX}\n"));
    let (ok, out, err) = spaghetti(&[
        "apply",
        "--scan-priv-file",
        secret_file.as_str(),
        "--tweak",
        "0/0/+",
    ]);
    assert!(ok, "{err}");
    let base = fields(&out)["vanity scan public key"].clone();
    assert_eq!(base.len(), 66);
    // The same secret on stdin gives the same key.
    let (ok, out, err) = apply(D_HEX, &["--tweak", "0/0/+"]);
    assert!(ok, "{err}");
    assert_eq!(fields(&out)["vanity scan public key"], base);

    // Split-key search; one thread keeps t tiny so the debug-build recover is fast.
    let (ok, out, err) = spaghetti(&["-q", "-c", "1", "--batch", "64", "-b", &base, "sp1qqgq"]);
    assert!(ok, "{err}");
    let found = fields(&out);
    assert_eq!(found["base scan pubkey"], base);
    let tweak = found["tweak"].clone();
    let pubkey = found["vanity scan pubkey"].clone();
    let address = found["address"].clone();
    assert!(address.starts_with("sp1qqgq"), "{address}");
    assert!(pubkey.starts_with("02"), "{pubkey}");
    assert!(out.contains(&format!("--tweak {tweak}")), "{out}");

    // apply: the tweaked secret gives the vanity pubkey and matches the address.
    let (ok, out, err) = apply(D_HEX, &["--tweak", &tweak, "--address", &address]);
    assert!(ok, "{err}");
    let applied = fields(&out);
    assert_eq!(applied["vanity scan public key"], pubkey);
    assert_ne!(applied["vanity scan secret key"], D_HEX);

    // apply refuses a mismatching address.
    let other = "sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwv";
    let (ok, _, err) = apply(D_HEX, &["--tweak", &tweak, "--address", other]);
    assert!(!ok);
    assert!(err.contains("not the tweaked key"), "{err}");

    // recover: the tweak comes back from the address and the base key alone.
    let (ok, out, err) = spaghetti(&[
        "recover",
        "--address",
        &address,
        "-b",
        &base,
        "-c",
        "2",
        "--baby-bits",
        "12",
    ]);
    assert!(ok, "{err}");
    let recovered = fields(&out);
    assert_eq!(recovered["vanity scan pubkey"], pubkey);
    let (ok, out, err) = apply(D_HEX, &["--tweak", &recovered["tweak"]]);
    assert!(ok, "{err}");
    assert_eq!(fields(&out)["vanity scan public key"], pubkey);

    // recover with the wrong base key fails within the reduced range.
    let (ok, _, err) = spaghetti(&[
        "recover",
        "--address",
        &address,
        "-b",
        "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "--baby-bits",
        "8",
        "--max-bits",
        "12",
    ]);
    assert!(!ok);
    assert!(err.contains("no tweak below 2^12"), "{err}");
}

#[test]
fn xpub_base_key_and_network_check() {
    // BIP32 test vector 1 node m/0H/1/2H: its key and chain code re-serialised
    // with depth 4 and child 1', the header of an m/352'/coin'/account'/1' node.
    const XPUB: &str = "xpub6EwK5B8QEa84vLJR7ik6SXv8J5uvxFG5UqdZGqfwQWNqhQfKEd1enZhemimbo7gZw3GJMvfAJsqMYBDsBZHpmBr5j5sECGixfcyhTb4B9jY";
    // The vector as published (depth 3, child 2') is not a scan account node.
    const VECTOR_XPUB: &str = "xpub6D4BDPcP2GT577Vvch3R8wDkScZWzQzMMUm3PWbmWvVJrZwQY4VUNgqFJPMM3No2dFDFGTsxxpG5uJh7n7epu4trkrX7x7DogT5Uv6fcLW5";
    let (ok, _, err) = spaghetti(&["-q", "--xpub", VECTOR_XPUB, "pa"]);
    assert!(!ok);
    assert!(
        err.contains("depth 3") && err.contains("0x80000002"),
        "{err}"
    );
    let (ok, out, err) = spaghetti(&["-q", "-c", "1", "--batch", "64", "--xpub", XPUB, "pa"]);
    assert!(ok, "{err}");
    let found = fields(&out);
    // Base = child 0 of the re-serialised node (deterministic).
    let base = &found["base scan pubkey"];
    assert_eq!(base.len(), 66);
    assert!(base.starts_with("02") || base.starts_with("03"), "{base}");
    let address = &found["address"];
    assert!(
        address.starts_with("sp1qq") && address[6..].starts_with("pa"),
        "{address}"
    );
    // recover through the same xpub reproduces the vanity key.
    let (ok, out, err) = spaghetti(&[
        "recover",
        "--address",
        &found["address"],
        "--xpub",
        XPUB,
        "--baby-bits",
        "12",
    ]);
    assert!(ok, "{err}");
    let recovered = fields(&out);
    assert_eq!(&recovered["base scan pubkey"], base);
    assert_eq!(recovered["vanity scan pubkey"], found["vanity scan pubkey"]);
    assert_eq!(recovered["tweak"], found["tweak"]);
    let (ok, _, err) = spaghetti(&["-q", "-n", "testnet", "--xpub", XPUB, "pa"]);
    assert!(!ok);
    assert!(err.contains("mainnet"), "{err}");
    let (ok, _, err) = spaghetti(&["-b", "02aa", "--xpub", XPUB, "pa"]);
    assert!(!ok, "{err}");
}

#[test]
fn regtest_and_signet_with_tpub() {
    // The xpub of xpub_base_key_and_network_check re-serialised with the tpub
    // version bytes: the node m/352'/1'/0'/1' of a testnet/signet/regtest wallet.
    const TPUB: &str = "tpubDFJwbQx5wr5XC8VGZujBeTFVdD1awdm92UDrdPj5kR8jShL2Jqrjw1NV1krPKcdoiwoCyWB2qyfQEMiUaR94edJNngcXsoPCFaa7DckCpRJ";
    const XPUB: &str = "xpub6EwK5B8QEa84vLJR7ik6SXv8J5uvxFG5UqdZGqfwQWNqhQfKEd1enZhemimbo7gZw3GJMvfAJsqMYBDsBZHpmBr5j5sECGixfcyhTb4B9jY";
    let search = |network: &str| {
        let (ok, out, err) = spaghetti(&[
            "-q", "-n", network, "-c", "1", "--batch", "64", "--xpub", TPUB, "pa",
        ]);
        assert!(ok, "{err}");
        fields(&out)
    };
    let regtest = search("regtest");
    let address = &regtest["address"];
    assert!(
        address.starts_with("sprt1qq") && address[8..].starts_with("pa"),
        "{address}"
    );
    // Same tpub and deterministic single-thread walk: the key is the same on
    // every test network, only the hrp differs (the spend key is random).
    let signet = search("signet");
    assert!(
        signet["address"].starts_with("tsp1qq"),
        "{}",
        signet["address"]
    );
    for other in [&signet, &search("testnet")] {
        assert_eq!(other["base scan pubkey"], regtest["base scan pubkey"]);
        assert_eq!(other["vanity scan pubkey"], regtest["vanity scan pubkey"]);
        assert_eq!(other["tweak"], regtest["tweak"]);
    }

    // recover takes the network from the sprt1 address.
    let (ok, out, err) = spaghetti(&[
        "recover",
        "--address",
        address,
        "--xpub",
        TPUB,
        "--baby-bits",
        "12",
    ]);
    assert!(ok, "{err}");
    let recovered = fields(&out);
    assert_eq!(
        recovered["vanity scan pubkey"],
        regtest["vanity scan pubkey"]
    );
    assert_eq!(recovered["tweak"], regtest["tweak"]);

    // A mainnet xpub does not fit a regtest search or address, nor a tpub mainnet.
    let (ok, _, err) = spaghetti(&["-q", "-n", "regtest", "--xpub", XPUB, "pa"]);
    assert!(!ok);
    assert!(
        err.contains("xpub (mainnet)") && err.contains("regtest"),
        "{err}"
    );
    let (ok, _, err) = spaghetti(&["recover", "--address", address, "--xpub", XPUB]);
    assert!(!ok);
    assert!(err.contains("regtest (sprt1"), "{err}");
    let (ok, _, err) = spaghetti(&["-q", "--xpub", TPUB, "pa"]);
    assert!(!ok);
    assert!(err.contains("tpub") && err.contains("mainnet"), "{err}");

    // apply checks a regtest address like any other.
    let (ok, out, err) = spaghetti(&["-q", "-n", "regtest", "--batch", "64", "sprt1qq?pa"]);
    assert!(ok, "{err}");
    let random = fields(&out);
    let (ok, out, err) = apply(
        &random["scan secret key"],
        &["--tweak", "0/0/+", "--address", &random["address"]],
    );
    assert!(ok, "{err}");
    assert_eq!(
        fields(&out)["vanity scan public key"],
        random["scan public key"]
    );
    // A wrong-network pattern is rejected with a hint.
    let (ok, _, err) = spaghetti(&["-q", "-n", "regtest", "tsp1qq?pa"]);
    assert!(!ok);
    assert!(err.contains("-n signet"), "{err}");
}

#[test]
fn cli_limits_and_error_prefixes() {
    let (ok, _, err) = spaghetti(&["-q", "-c", "1025", "pa"]);
    assert!(!ok);
    assert!(err.contains("--cores") && err.contains("1024"), "{err}");
    let (ok, _, err) = spaghetti(&["-q", "--batch", "0", "pa"]);
    assert!(!ok);
    assert!(err.contains("--batch"), "{err}");
    let (ok, _, err) = spaghetti(&["-q", "--batch", "524289", "pa"]);
    assert!(!ok);
    assert!(err.contains("--batch"), "{err}");
    let (ok, _, err) = spaghetti(&["recover", "--address", "sp1qqnotanaddress", "-b", "02aa"]);
    assert!(!ok);
    assert!(err.contains("--address:"), "{err}");
    let (ok, _, err) = spaghetti(&["-q", "pastá"]);
    assert!(!ok);
    assert!(err.contains("'á'"), "{err}");
    // The secret never comes from the command line.
    let (ok, _, err) = spaghetti(&["apply", "--scan-priv", D_HEX, "--tweak", "0/0/+"]);
    assert!(!ok);
    assert!(err.contains("--scan-priv"), "{err}");
}

#[test]
fn scan_priv_file_errors_are_prefixed() {
    let missing = TempPath::new("missing");
    let (ok, _, err) = spaghetti(&[
        "apply",
        "--scan-priv-file",
        missing.as_str(),
        "--tweak",
        "0/0/+",
    ]);
    assert!(!ok);
    assert!(err.starts_with("error: --scan-priv-file:"), "{err}");
    for bad in ["zz", "01", &"00".repeat(32), &"ff".repeat(32)] {
        let (ok, _, err) = apply(bad, &["--tweak", "0/0/+"]);
        assert!(!ok, "{bad}");
        assert!(err.starts_with("error: --scan-priv-file:"), "{err}");
    }
}

#[cfg(unix)]
fn mode(path: &TempPath) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(&path.0)
        .expect("metadata")
        .permissions()
        .mode()
        & 0o777
}

/// `--output`: the secret goes to a new 0600 file only, stdout points at it,
/// an existing file is never overwritten, and split-key mode refuses the flag.
#[test]
fn output_file_holds_the_secret() {
    let out_file = TempPath::new("apply.out");
    let (ok, out, err) = apply(D_HEX, &["--tweak", "0/0/+", "--output", out_file.as_str()]);
    assert!(ok, "{err}");
    let shown = fields(&out);
    assert_eq!(
        shown["vanity scan secret key"],
        format!("written to {}", out_file.as_str())
    );
    assert!(!out.contains(D_HEX), "{out}");
    let written = std::fs::read_to_string(&out_file.0).expect("read output");
    let saved = fields(&written);
    assert_eq!(saved["vanity scan secret key"], D_HEX);
    assert_eq!(
        saved["vanity scan public key"],
        shown["vanity scan public key"]
    );
    #[cfg(unix)]
    assert_eq!(mode(&out_file), 0o600);
    // Never overwritten.
    let (ok, _, err) = apply(D_HEX, &["--tweak", "0/0/+", "--output", out_file.as_str()]);
    assert!(!ok);
    assert!(
        err.contains("--output:") && err.contains("already exists"),
        "{err}"
    );
    assert_eq!(
        std::fs::read_to_string(&out_file.0).expect("read output"),
        written
    );

    // Random-mode search: two matches appended to one file, no secret on stdout.
    let search_file = TempPath::new("search.out");
    let (ok, out, err) = spaghetti(&[
        "-q",
        "-c",
        "1",
        "--batch",
        "64",
        "-k",
        "2",
        "--output",
        search_file.as_str(),
        "sp1qq?q",
    ]);
    assert!(ok, "{err}");
    let pointer = format!("scan secret key : written to {}", search_file.as_str());
    assert_eq!(out.matches(&pointer).count(), 2, "{out}");
    assert_eq!(out.matches("found after").count(), 2, "{out}");
    let written = std::fs::read_to_string(&search_file.0).expect("read output");
    let secrets: Vec<&str> = written
        .lines()
        .filter_map(|line| line.strip_prefix("scan secret key : "))
        .collect();
    assert_eq!(secrets.len(), 2, "{written}");
    for secret in &secrets {
        assert_eq!(secret.len(), 64, "{secret}");
        assert!(!out.contains(secret), "{out}");
    }
    let public: Vec<&str> = out
        .lines()
        .filter(|line| line.starts_with("scan public key : "))
        .collect();
    for line in public {
        assert!(written.contains(line), "{written}");
    }
    assert_eq!(written.matches("found after").count(), 2, "{written}");
    #[cfg(unix)]
    assert_eq!(mode(&search_file), 0o600);

    // Split-key mode has no secret to write.
    let split_file = TempPath::new("split.out");
    let (ok, _, err) = spaghetti(&[
        "-q",
        "-b",
        "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "--output",
        split_file.as_str(),
        "sp1qq?q",
    ]);
    assert!(!ok);
    assert!(
        err.contains("--output: split-key mode prints no secret"),
        "{err}"
    );
    assert!(!split_file.0.exists());
}
