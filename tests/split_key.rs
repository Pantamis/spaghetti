//! End-to-end split-key flow through the binary: search from a base key,
//! `apply` the tweak to the base secret, `recover` the tweak from the address.

use std::collections::HashMap;
use std::process::Command;

const D_HEX: &str = "4f3c0a1b5e6d7c8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5";

fn spaghetti(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_spaghetti"))
        .args(args)
        .output()
        .expect("run spaghetti");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
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
    // Base pubkey D = d·G via `apply` with the neutral tweak.
    let (ok, out, err) = spaghetti(&["apply", "--scan-priv", D_HEX, "--tweak", "0/0/+"]);
    assert!(ok, "{err}");
    let base = fields(&out)["vanity scan public key"].clone();
    assert_eq!(base.len(), 66);

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
    let (ok, out, err) = spaghetti(&[
        "apply",
        "--scan-priv",
        D_HEX,
        "--tweak",
        &tweak,
        "--address",
        &address,
    ]);
    assert!(ok, "{err}");
    let applied = fields(&out);
    assert_eq!(applied["vanity scan public key"], pubkey);
    assert_ne!(applied["vanity scan secret key"], D_HEX);

    // apply refuses a mismatching address.
    let other = "sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwv";
    let (ok, _, err) = spaghetti(&[
        "apply",
        "--scan-priv",
        D_HEX,
        "--tweak",
        &tweak,
        "--address",
        other,
    ]);
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
    let (ok, out, err) = spaghetti(&[
        "apply",
        "--scan-priv",
        D_HEX,
        "--tweak",
        &recovered["tweak"],
    ]);
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
    const XPUB: &str = "xpub6D4BDPcP2GT577Vvch3R8wDkScZWzQzMMUm3PWbmWvVJrZwQY4VUNgqFJPMM3No2dFDFGTsxxpG5uJh7n7epu4trkrX7x7DogT5Uv6fcLW5";
    let (ok, out, err) = spaghetti(&["-q", "-c", "1", "--batch", "64", "--xpub", XPUB, "pa"]);
    assert!(ok, "{err}");
    let found = fields(&out);
    // Base = child 0 of the BIP32 vector-1 node m/0H/1/2H (deterministic).
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
