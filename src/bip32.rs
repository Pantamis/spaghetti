//! Minimal BIP32: parse a serialised extended *public* key and derive one
//! non-hardened child. Enough to turn the xpub of `m/352'/coin'/account'/1'`
//! into the scan public key at `…/1'/0`; [`scan_account_pubkey`] also checks
//! that the xpub sits at that node (depth 4, child `1'`).
//!
//! And from a wallet's root extended *private* key (`xprv`/`tprv`, depth 0),
//! the BIP352 keys of an account ([`bip352_from_root`]): the scan secret at
//! `m/352'/coin'/account'/1'/0` and the spend public key at
//! `m/352'/coin'/account'/0'/0`, `coin` being 0 for an `xprv` and 1 for a
//! `tprv` (testnet, signet and regtest). Secret intermediates (keys, chain
//! codes, HMAC outputs) are held in zeroizing wrappers.

use hmac::{Hmac, KeyInit, Mac};
use k256::elliptic_curve::Group;
use k256::elliptic_curve::PrimeField;
use k256::elliptic_curve::sec1::ToSec1Point;
use k256::{ProjectivePoint, PublicKey, Scalar};
use sha2::{Digest, Sha256, Sha512};
use zeroize::Zeroizing;

use crate::address::Network;

const VERSION_XPUB: [u8; 4] = [0x04, 0x88, 0xB2, 0x1E];
const VERSION_TPUB: [u8; 4] = [0x04, 0x35, 0x87, 0xCF];
const VERSION_XPRV: [u8; 4] = [0x04, 0x88, 0xAD, 0xE4];
const VERSION_TPRV: [u8; 4] = [0x04, 0x35, 0x83, 0x94];

/// The version of an extended public key: `xpub` on mainnet, `tpub` on every
/// test network (testnet, signet and regtest share it and coin type `1'`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Version {
    Xpub,
    Tpub,
}

impl Version {
    /// The version wallets of `network` export.
    pub fn of(network: Network) -> Version {
        if network.is_test() {
            Version::Tpub
        } else {
            Version::Xpub
        }
    }

    /// For messages.
    pub fn describe(self) -> &'static str {
        match self {
            Version::Xpub => "an xpub (mainnet)",
            Version::Tpub => "a tpub (testnet, signet or regtest)",
        }
    }
}

/// Depth of `m/352'/coin'/account'/1'`.
const SCAN_ACCOUNT_DEPTH: u8 = 4;
/// Child number of the last step, `1'`.
const SCAN_ACCOUNT_CHILD: u32 = 0x8000_0001;

/// The parts of a serialised extended public key that derivation needs.
#[derive(Debug)]
struct Xpub {
    version: Version,
    depth: u8,
    child: u32,
    chain_code: [u8; 32],
    key: ProjectivePoint,
}

/// Base58check-decodes and parses an `xpub`/`tpub`.
fn parse(text: &str) -> Result<Xpub, String> {
    let raw = bs58::decode(text.trim())
        .into_vec()
        .map_err(|e| format!("xpub: not base58: {e}"))?;
    if raw.len() != 82 {
        return Err(format!(
            "xpub: expected 82 base58check bytes, got {}",
            raw.len()
        ));
    }
    let (payload, checksum) = raw.split_at(78);
    let digest = Sha256::digest(Sha256::digest(payload));
    if digest[..4] != *checksum {
        return Err("xpub: bad base58check checksum".to_string());
    }
    let version: [u8; 4] = payload[..4]
        .try_into()
        .map_err(|_| "xpub: internal: version slice")?;
    let version = match version {
        VERSION_XPUB => Version::Xpub,
        VERSION_TPUB => Version::Tpub,
        VERSION_XPRV | VERSION_TPRV => {
            return Err(
                "xpub: this is an extended *private* key; export the xpub instead".to_string(),
            );
        }
        other => {
            return Err(format!(
                "xpub: unknown version bytes {}",
                hex::encode(other)
            ));
        }
    };
    let depth = payload[4];
    let child = u32::from_be_bytes(
        payload[9..13]
            .try_into()
            .map_err(|_| "xpub: internal: child number slice")?,
    );
    let chain_code: [u8; 32] = payload[13..45]
        .try_into()
        .map_err(|_| "xpub: internal: chain code slice")?;
    let key = PublicKey::from_sec1_bytes(&payload[45..78])
        .map_err(|_| "xpub: key bytes are not a valid compressed secp256k1 point".to_string())?
        .to_projective();
    Ok(Xpub {
        version,
        depth,
        child,
        chain_code,
        key,
    })
}

/// An extended private key; key and chain code are wiped on drop.
struct Xprv {
    /// `tprv` (test networks) rather than `xprv`.
    test: bool,
    depth: u8,
    child: u32,
    parent_fingerprint: [u8; 4],
    chain_code: Zeroizing<[u8; 32]>,
    key: Zeroizing<Scalar>,
}

/// Base58check-decodes and parses an `xprv`/`tprv`.
fn parse_xprv(text: &str) -> Result<Xprv, String> {
    let what = "root key";
    let raw = Zeroizing::new(
        bs58::decode(text.trim())
            .into_vec()
            .map_err(|_| format!("{what}: not base58"))?,
    );
    if raw.len() != 82 {
        return Err(format!(
            "{what}: expected an extended private key (82 base58check bytes), got {} bytes",
            raw.len()
        ));
    }
    let (payload, checksum) = raw.split_at(78);
    let digest = Sha256::digest(Sha256::digest(payload));
    if digest[..4] != *checksum {
        return Err(format!("{what}: bad base58check checksum"));
    }
    let test = match payload[..4] {
        [0x04, 0x88, 0xAD, 0xE4] => false,
        [0x04, 0x35, 0x83, 0x94] => true,
        [0x04, 0x88, 0xB2, 0x1E] | [0x04, 0x35, 0x87, 0xCF] => {
            return Err(format!(
                "{what}: this is an extended *public* key; the hardened BIP352 path needs the \
                 xprv/tprv (or pass the account xpub with --xpub)"
            ));
        }
        _ => {
            return Err(format!(
                "{what}: unknown version bytes {}",
                hex::encode(&payload[..4])
            ));
        }
    };
    if payload[45] != 0 {
        return Err(format!("{what}: private key bytes must start with 0x00"));
    }
    let mut key_bytes = Zeroizing::new([0u8; 32]);
    key_bytes.copy_from_slice(&payload[46..78]);
    let key = Scalar::from_repr_vartime((*key_bytes).into())
        .filter(|k| !bool::from(k.is_zero()))
        .map(Zeroizing::new)
        .ok_or_else(|| format!("{what}: private key is zero or not below the curve order"))?;
    let mut chain_code = Zeroizing::new([0u8; 32]);
    chain_code.copy_from_slice(&payload[13..45]);
    let mut parent_fingerprint = [0u8; 4];
    parent_fingerprint.copy_from_slice(&payload[5..9]);
    Ok(Xprv {
        test,
        depth: payload[4],
        child: u32::from_be_bytes([payload[9], payload[10], payload[11], payload[12]]),
        parent_fingerprint,
        chain_code,
        key,
    })
}

/// Child `index` (hardened when `>= 2^31`) of an extended private key:
/// `k_i = I_L + k` with `I = HMAC-SHA512(c, 0x00 ‖ ser256(k) ‖ ser32(i))`
/// (hardened) or `HMAC-SHA512(c, ser_P(k·G) ‖ ser32(i))`, `c_i = I_R`.
fn derive_private(parent: &Xprv, index: u32) -> Result<Xprv, String> {
    let mut mac = Hmac::<Sha512>::new_from_slice(parent.chain_code.as_slice())
        .map_err(|_| "root key: internal: hmac key length".to_string())?;
    if index >= 1 << 31 {
        let key: Zeroizing<[u8; 32]> = Zeroizing::new(parent.key.to_bytes().into());
        mac.update(&[0]);
        mac.update(key.as_slice());
    } else {
        let point = ProjectivePoint::GENERATOR * *parent.key;
        mac.update(point.to_affine().to_sec1_point(true).as_bytes());
    }
    mac.update(&index.to_be_bytes());
    let mut i = Zeroizing::new([0u8; 64]);
    i.copy_from_slice(&mac.finalize().into_bytes());
    let mut left = Zeroizing::new([0u8; 32]);
    left.copy_from_slice(&i[..32]);
    let tweak = Zeroizing::new(Scalar::from_repr_vartime((*left).into()).ok_or_else(|| {
        format!(
            "root key: child {} has I_L >= n (invalid child)",
            describe_child(index)
        )
    })?);
    let key = Zeroizing::new(tweak.add(&parent.key));
    if bool::from(key.is_zero()) {
        return Err(format!(
            "root key: child {} is the zero key (invalid child)",
            describe_child(index)
        ));
    }
    let mut chain_code = Zeroizing::new([0u8; 32]);
    chain_code.copy_from_slice(&i[32..]);
    Ok(Xprv {
        test: parent.test,
        depth: parent.depth.saturating_add(1),
        child: index,
        parent_fingerprint: [0; 4],
        chain_code,
        key,
    })
}

const HARDENED: u32 = 1 << 31;

/// The BIP352 keys of one account of a wallet.
pub struct WalletKeys {
    /// The scan secret `d` at `m/352'/coin'/account'/1'/0` (wiped on drop).
    pub scan: Zeroizing<Scalar>,
    /// The spend public key at `m/352'/coin'/account'/0'/0`.
    pub spend: ProjectivePoint,
    /// `Xpub` for an `xprv` (mainnet), `Tpub` for a `tprv` (test networks).
    pub version: Version,
    /// The scan key path, for messages.
    pub scan_path: String,
}

/// Derives the BIP352 keys of `account` from a wallet's root extended private
/// key (depth 0). The coin type follows the key: 0 for an `xprv`, 1 for a
/// `tprv`.
pub fn bip352_from_root(text: &str, account: u32) -> Result<WalletKeys, String> {
    if account >= HARDENED {
        return Err(format!(
            "--account: at most {}, got {account}",
            HARDENED - 1
        ));
    }
    let root = parse_xprv(text)?;
    if root.depth != 0 || root.child != 0 || root.parent_fingerprint != [0; 4] {
        return Err(format!(
            "root key: expected the wallet's root (master) key, depth 0; this one is at depth {} \
             (child {})",
            root.depth,
            describe_child(root.child)
        ));
    }
    let coin = u32::from(root.test);
    let path = |branch: u32| {
        [
            352 | HARDENED,
            coin | HARDENED,
            account | HARDENED,
            branch | HARDENED,
            0,
        ]
    };
    let derive = |steps: [u32; 5]| -> Result<Xprv, String> {
        let mut node = derive_private(&root, steps[0])?;
        for &index in &steps[1..] {
            node = derive_private(&node, index)?;
        }
        Ok(node)
    };
    let scan = derive(path(1))?;
    let spend = derive(path(0))?;
    Ok(WalletKeys {
        scan: Zeroizing::new(*scan.key),
        spend: ProjectivePoint::GENERATOR * *spend.key,
        version: if root.test {
            Version::Tpub
        } else {
            Version::Xpub
        },
        scan_path: format!("m/352'/{coin}'/{account}'/1'/0"),
    })
}

/// The BIP352 scan public key `…/1'/0` of the account xpub `m/352'/coin'/account'/1'`,
/// with the xpub's version. Rejects an xpub at any other depth or child number.
pub fn scan_account_pubkey(xpub: &str) -> Result<(ProjectivePoint, Version), String> {
    let parsed = parse(xpub)?;
    if parsed.depth != SCAN_ACCOUNT_DEPTH || parsed.child != SCAN_ACCOUNT_CHILD {
        return Err(format!(
            "xpub: expected the node m/352'/coin'/account'/1' (depth {SCAN_ACCOUNT_DEPTH}, \
             child 1' = 0x{SCAN_ACCOUNT_CHILD:08x}), got depth {} and child 0x{:08x} ({})",
            parsed.depth,
            parsed.child,
            describe_child(parsed.child)
        ));
    }
    Ok((derive_child(&parsed, 0)?, parsed.version))
}

/// `n` or `n'` for an error message.
fn describe_child(child: u32) -> String {
    if child >= 1 << 31 {
        format!("{}'", child - (1 << 31))
    } else {
        child.to_string()
    }
}

/// Non-hardened child `index` of `parent`: `K_i = K + I_L·G` with
/// `I = HMAC-SHA512(chain_code, ser_P(K) ‖ ser32(index))`.
fn derive_child(parent: &Xpub, index: u32) -> Result<ProjectivePoint, String> {
    if index >= 1 << 31 {
        return Err(format!(
            "xpub: child {index} is hardened; only non-hardened children can be derived from an xpub"
        ));
    }
    let mut mac = Hmac::<Sha512>::new_from_slice(&parent.chain_code)
        .map_err(|_| "xpub: internal: hmac key length".to_string())?;
    mac.update(parent.key.to_affine().to_sec1_point(true).as_bytes());
    mac.update(&index.to_be_bytes());
    let i = mac.finalize().into_bytes();
    let left: [u8; 32] = i[..32]
        .try_into()
        .map_err(|_| "xpub: internal: I_L slice")?;
    let tweak = Scalar::from_repr_vartime(left.into()).ok_or_else(|| {
        "xpub: child derivation yielded I_L >= n (invalid child, try the next index)".to_string()
    })?;
    let child = parent.key + ProjectivePoint::GENERATOR * tweak;
    if bool::from(child.is_identity()) {
        return Err(
            "xpub: child derivation yielded the point at infinity (invalid child)".to_string(),
        );
    }
    Ok(child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::compressed;

    // BIP32 test vector 1, m/0H/1/2H and its child m/0H/1/2H/2.
    const PARENT: &str = "xpub6D4BDPcP2GT577Vvch3R8wDkScZWzQzMMUm3PWbmWvVJrZwQY4VUNgqFJPMM3No2dFDFGTsxxpG5uJh7n7epu4trkrX7x7DogT5Uv6fcLW5";
    const CHILD: &str = "xpub6FHa3pjLCk84BayeJxFW2SP4XRrFd1JYnxeLeU8EqN3vDfZmbqBqaGJAyiLjTAwm6ZLRQUMv1ZACTj37sR62cfN7fe5JnJ7dh8zL4fiyLHV";
    const PARENT_PRV: &str = "xprv9z4pot5VBttmtdRTWfWQmoH1taj2axGVzFqSb8C9xaxKymcFzXBDptWmT7FwuEzG3ryjH4ktypQSAewRiNMjANTtpgP4mLTj34bhnZX7UiM";

    #[test]
    fn bip32_vector_1_child_2() {
        let parent = parse(PARENT).unwrap();
        let child = derive_child(&parent, 2).unwrap();
        assert_eq!(child, parse(CHILD).unwrap().key);
        assert_eq!(
            hex::encode(compressed(&child)),
            "02e8445082a72f29b75ca48748a914df60622a609cacfce8ed0e35804560741d29"
        );
    }

    /// The vector's parent re-serialised with other header fields.
    fn reserialize(xpub: &str, version: Option<[u8; 4]>, depth: u8, child: u32) -> String {
        let mut raw = bs58::decode(xpub).into_vec().unwrap();
        if let Some(version) = version {
            raw[..4].copy_from_slice(&version);
        }
        raw[4] = depth;
        raw[9..13].copy_from_slice(&child.to_be_bytes());
        let digest = Sha256::digest(Sha256::digest(&raw[..78]));
        raw[78..].copy_from_slice(&digest[..4]);
        bs58::encode(&raw).into_string()
    }

    #[test]
    fn parse_fields() {
        let parent = parse(PARENT).unwrap();
        assert_eq!(parent.version, Version::Xpub);
        assert_eq!(parent.depth, 3);
        assert_eq!(parent.child, 0x8000_0002);
        assert_eq!(
            hex::encode(parent.chain_code),
            "04466b9cc8e161e966409ca52986c584f07e9dc81f735db683c3ff6ec7b1503f"
        );
        assert_eq!(
            hex::encode(compressed(&parent.key)),
            "0357bfe1e341d01c69fe5654309956cbea516822fba8a601743a012a7896ee8dc2"
        );
    }

    #[test]
    fn scan_account_path_check() {
        let good = reserialize(PARENT, None, SCAN_ACCOUNT_DEPTH, SCAN_ACCOUNT_CHILD);
        let (key, version) = scan_account_pubkey(&good).unwrap();
        assert_eq!(key, derive_child(&parse(PARENT).unwrap(), 0).unwrap());
        assert_eq!(version, Version::Xpub);
        let err = scan_account_pubkey(PARENT).unwrap_err();
        assert!(err.contains("depth 3"), "{err}");
        assert!(err.contains("0x80000002 (2')"), "{err}");
        assert!(err.contains("m/352'/coin'/account'/1'"), "{err}");
        let wrong_child = reserialize(PARENT, None, SCAN_ACCOUNT_DEPTH, 1);
        let err = scan_account_pubkey(&wrong_child).unwrap_err();
        assert!(err.contains("depth 4 and child 0x00000001 (1)"), "{err}");
        let wrong_depth = reserialize(PARENT, None, 5, SCAN_ACCOUNT_CHILD);
        assert!(scan_account_pubkey(&wrong_depth).is_err());
        // derive_child itself stays path-agnostic.
        assert!(derive_child(&parse(&wrong_depth).unwrap(), 0).is_ok());
    }

    #[test]
    fn tpub_version() {
        // Re-serialise the vector's parent with the tpub version bytes.
        let tpub = reserialize(PARENT, Some(VERSION_TPUB), 3, 0x8000_0002);
        assert!(tpub.starts_with("tpub"), "{tpub}");
        let parsed = parse(&tpub).unwrap();
        assert_eq!(parsed.version, Version::Tpub);
        assert_eq!(
            derive_child(&parsed, 2).unwrap(),
            derive_child(&parse(PARENT).unwrap(), 2).unwrap()
        );
    }

    /// BIP32 test vector 1: seed 000102…0f.
    fn vector_1_master() -> Xprv {
        let seed: Vec<u8> = (0u8..16).collect();
        let mut mac = Hmac::<Sha512>::new_from_slice(b"Bitcoin seed").unwrap();
        mac.update(&seed);
        let i = mac.finalize().into_bytes();
        let key: [u8; 32] = i[..32].try_into().unwrap();
        let chain: [u8; 32] = i[32..].try_into().unwrap();
        Xprv {
            test: false,
            depth: 0,
            child: 0,
            parent_fingerprint: [0; 4],
            chain_code: Zeroizing::new(chain),
            key: Zeroizing::new(Scalar::from_repr_vartime(key.into()).unwrap()),
        }
    }

    /// Serialises an extended private key (test networks: `tprv`).
    fn serialize_xprv(node: &Xprv) -> String {
        let mut raw = Vec::with_capacity(82);
        raw.extend_from_slice(if node.test {
            &[0x04, 0x35, 0x83, 0x94]
        } else {
            &VERSION_XPRV
        });
        raw.push(node.depth);
        raw.extend_from_slice(&node.parent_fingerprint);
        raw.extend_from_slice(&node.child.to_be_bytes());
        raw.extend_from_slice(node.chain_code.as_slice());
        raw.push(0);
        raw.extend_from_slice(&node.key.to_bytes());
        let digest = Sha256::digest(Sha256::digest(&raw));
        raw.extend_from_slice(&digest[..4]);
        bs58::encode(&raw).into_string()
    }

    /// Serialises the public side of a node as an xpub/tpub (fingerprint
    /// left zero: `parse` does not check it).
    fn serialize_xpub(node: &Xprv) -> String {
        let mut raw = Vec::with_capacity(82);
        raw.extend_from_slice(if node.test {
            &VERSION_TPUB
        } else {
            &VERSION_XPUB
        });
        raw.push(node.depth);
        raw.extend_from_slice(&[0; 4]);
        raw.extend_from_slice(&node.child.to_be_bytes());
        raw.extend_from_slice(node.chain_code.as_slice());
        let point = ProjectivePoint::GENERATOR * *node.key;
        raw.extend_from_slice(point.to_affine().to_sec1_point(true).as_bytes());
        let digest = Sha256::digest(Sha256::digest(&raw));
        raw.extend_from_slice(&digest[..4]);
        bs58::encode(&raw).into_string()
    }

    /// Private derivation m/0'/1/2' of vector 1 gives the vector's xprv
    /// (hardened and non-hardened steps), and its public key the vector's xpub.
    #[test]
    fn private_derivation_matches_vector_1() {
        let master = vector_1_master();
        let root = parse_xprv(&serialize_xprv(&master)).unwrap();
        assert!(!root.test && root.depth == 0);
        let a = derive_private(&root, HARDENED).unwrap();
        let b = derive_private(&a, 1).unwrap();
        let c = derive_private(&b, 2 | HARDENED).unwrap();
        let expected = parse_xprv(PARENT_PRV).unwrap();
        assert_eq!(*c.key, *expected.key);
        assert_eq!(*c.chain_code, *expected.chain_code);
        assert_eq!(c.depth, 3);
        assert_eq!(
            ProjectivePoint::GENERATOR * *c.key,
            parse(PARENT).unwrap().key
        );
    }

    /// The BIP352 keys from the root agree with the public derivation from
    /// the account xpub, for both coin types, and the path is checked.
    #[test]
    fn bip352_keys_from_root() {
        for test in [false, true] {
            let mut master = vector_1_master();
            master.test = test;
            let text = serialize_xprv(&master);
            assert!(
                text.starts_with(if test { "tprv" } else { "xprv" }),
                "{text}"
            );
            let keys = bip352_from_root(&text, 3).unwrap();
            let coin = u32::from(test);
            assert_eq!(keys.scan_path, format!("m/352'/{coin}'/3'/1'/0"));
            assert_eq!(
                keys.version,
                if test { Version::Tpub } else { Version::Xpub }
            );
            let mut node = derive_private(&master, 352 | HARDENED).unwrap();
            for index in [coin | HARDENED, 3 | HARDENED, 1 | HARDENED] {
                node = derive_private(&node, index).unwrap();
            }
            let (scan_pub, version) = scan_account_pubkey(&serialize_xpub(&node)).unwrap();
            assert_eq!(version, keys.version);
            assert_eq!(scan_pub, ProjectivePoint::GENERATOR * *keys.scan);
            // The spend key is on the 0' branch, not the scan one.
            assert_ne!(keys.spend, scan_pub);
            let other = bip352_from_root(&text, 0).unwrap();
            assert_ne!(*other.scan, *keys.scan);
        }
    }

    #[test]
    fn root_key_errors() {
        let master = vector_1_master();
        let child = derive_private(&master, HARDENED).unwrap();
        let err = bip352_from_root(&serialize_xprv(&child), 0)
            .err()
            .expect("an error");
        assert!(err.contains("depth 1"), "{err}");
        let err = bip352_from_root(PARENT, 0).err().expect("an error");
        assert!(err.contains("*public*"), "{err}");
        let err = bip352_from_root(&serialize_xprv(&master), HARDENED)
            .err()
            .expect("an error");
        assert!(err.contains("--account"), "{err}");
        let mut text = serialize_xprv(&master);
        text.replace_range(20..21, if &text[20..21] == "a" { "b" } else { "a" });
        assert!(
            bip352_from_root(&text, 0)
                .err()
                .expect("an error")
                .contains("checksum")
        );
        // Surrounding whitespace (a file's newline) is fine.
        assert!(bip352_from_root(&format!("  {}\n", serialize_xprv(&master)), 0).is_ok());
    }

    #[test]
    fn version_of_network() {
        assert_eq!(Version::of(Network::Mainnet), Version::Xpub);
        for network in [Network::Testnet, Network::Signet, Network::Regtest] {
            assert_eq!(Version::of(network), Version::Tpub);
        }
    }

    #[test]
    fn rejects_xprv_checksum_and_hardened() {
        let err = parse(PARENT_PRV).unwrap_err();
        assert!(err.contains("private"), "{err}");
        let mut broken = PARENT.to_string();
        broken.replace_range(10..11, if &PARENT[10..11] == "a" { "b" } else { "a" });
        let err = parse(&broken).unwrap_err();
        assert!(err.contains("checksum"), "{err}");
        assert!(parse("not-base58-0OIl").is_err());
        assert!(parse("xpub").is_err());
        let err = derive_child(&parse(PARENT).unwrap(), 1 << 31).unwrap_err();
        assert!(err.contains("hardened"), "{err}");
    }
}
