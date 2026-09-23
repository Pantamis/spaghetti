//! BIP352 v0 silent payment address encoding via the `bech32` crate.
//!
//! `address = bech32m(hrp, [version 0] ++ convertbits8→5(ser_P(B_scan) ++ ser_P(B_m)))`.

use bech32::primitives::decode::CheckedHrpstring;
use bech32::{Bech32m, ByteIterExt, Fe32, Fe32IterExt, Hrp};
use clap::ValueEnum;

/// BIP352 defines the hrps `sp` (mainnet) and `tsp` (testnets); regtest uses
/// `sprt`, the hrp of the silent payment implementations that support it
/// (the BIP does not name one).
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
pub enum Network {
    /// hrp `sp`
    Mainnet,
    /// hrp `tsp`
    Testnet,
    /// hrp `tsp` (the BIP352 testnet hrp): same addresses and keys as testnet
    Signet,
    /// hrp `sprt`
    Regtest,
}

impl Network {
    pub fn hrp(self) -> Hrp {
        match self {
            Network::Mainnet => Hrp::parse_unchecked("sp"),
            Network::Testnet | Network::Signet => Hrp::parse_unchecked("tsp"),
            Network::Regtest => Hrp::parse_unchecked("sprt"),
        }
    }

    /// Whether wallets of this network export `tpub`s and derive with coin
    /// type `1'`: every network but mainnet.
    pub fn is_test(self) -> bool {
        self != Network::Mainnet
    }

    /// The network of a decoded hrp. `tsp` is shared by testnet and signet
    /// and decodes as [`Network::Testnet`]; an address cannot tell them apart.
    fn from_hrp(hrp: Hrp) -> Option<Network> {
        [Network::Mainnet, Network::Testnet, Network::Regtest]
            .into_iter()
            .find(|network| network.hrp() == hrp)
    }

    /// For messages: the network name(s) of this network's hrp and its xpub kind.
    pub fn describe(self) -> &'static str {
        match self {
            Network::Mainnet => "mainnet (sp1…, xpub)",
            Network::Testnet | Network::Signet => "testnet/signet (tsp1…, tpub)",
            Network::Regtest => "regtest (sprt1…, tpub)",
        }
    }
}

/// Encodes a v0 silent payment address from compressed scan and spend public keys.
pub fn encode(hrp: Hrp, scan: &[u8; 33], spend: &[u8; 33]) -> String {
    scan.iter()
        .chain(spend.iter())
        .copied()
        .bytes_to_fes()
        .with_checksum::<Bech32m>(&hrp)
        .with_witness_version(Fe32::Q)
        .chars()
        .collect()
}

/// Strictly decodes a v0 silent payment address into `(network, scan, spend)`:
/// bech32m, version `q`, exactly 66 payload bytes, zero padding bits (the
/// re-encoded keys must reproduce the input) and an hrp of [`Network`]
/// (`tsp` decodes as testnet, see [`Network::from_hrp`]).
pub fn decode(address: &str) -> Result<(Network, [u8; 33], [u8; 33]), String> {
    let mut parsed = CheckedHrpstring::new::<Bech32m>(address).map_err(|e| e.to_string())?;
    let version = parsed
        .remove_witness_version()
        .ok_or_else(|| "address has no version character".to_string())?;
    if version != Fe32::Q {
        return Err(format!("address version is '{version}', not the v0 'q'"));
    }
    let payload: Vec<u8> = parsed.fe32_iter().fes_to_bytes().collect();
    if payload.len() != 66 {
        return Err(format!(
            "address payload is {} bytes, expected 66 (two compressed keys)",
            payload.len()
        ));
    }
    let hrp = parsed.hrp();
    let mut scan = [0u8; 33];
    let mut spend = [0u8; 33];
    scan.copy_from_slice(&payload[..33]);
    spend.copy_from_slice(&payload[33..]);
    if encode(hrp, &scan, &spend) != address.to_ascii_lowercase() {
        return Err("address has non-zero padding bits".to_string());
    }
    let network = Network::from_hrp(hrp).ok_or_else(|| {
        format!("hrp '{hrp}' is not sp (mainnet), tsp (testnet/signet) or sprt (regtest)")
    })?;
    Ok((network, scan, spend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use k256::elliptic_curve::PrimeField;
    use k256::elliptic_curve::sec1::ToSec1Point;
    use k256::{ProjectivePoint, Scalar};

    fn compressed(secret_hex: &str) -> [u8; 33] {
        let bytes: [u8; 32] = hex::decode(secret_hex).unwrap().try_into().unwrap();
        let scalar = Scalar::from_repr_vartime(bytes.into()).unwrap();
        let point = (ProjectivePoint::GENERATOR * scalar).to_affine();
        point.to_sec1_point(true).as_bytes().try_into().unwrap()
    }

    /// bech32m string with version `q` over arbitrary payload bytes.
    fn raw(hrp: Hrp, payload: &[u8]) -> String {
        payload
            .iter()
            .copied()
            .bytes_to_fes()
            .with_checksum::<Bech32m>(&hrp)
            .with_witness_version(Fe32::Q)
            .chars()
            .collect()
    }

    const VECTOR_SCAN: &str = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c";
    const VECTOR_SPEND: &str = "9d6ad855ce3417ef84e836892e5a56392bfba05fa5d97ccea30e266f540e08b3";
    const VECTOR_ADDRESS: &str = "sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwv";

    #[test]
    fn bip352_test_vector() {
        let scan = compressed(VECTOR_SCAN);
        let spend = compressed(VECTOR_SPEND);
        let address = encode(Network::Mainnet.hrp(), &scan, &spend);
        assert_eq!(address, VECTOR_ADDRESS);
        assert_eq!(address.len(), 116);
    }

    #[test]
    fn decode_roundtrip() {
        let scan = compressed(VECTOR_SCAN);
        let spend = compressed(VECTOR_SPEND);
        let (network, got_scan, got_spend) = decode(VECTOR_ADDRESS).unwrap();
        assert_eq!(network, Network::Mainnet);
        assert_eq!(got_scan, scan);
        assert_eq!(got_spend, spend);
        // Uppercase is valid bech32; mixed case is not.
        assert!(decode(&VECTOR_ADDRESS.to_uppercase()).is_ok());
        let mut mixed = VECTOR_ADDRESS.to_string();
        mixed.replace_range(0..1, "S");
        assert!(decode(&mixed).is_err());
    }

    #[test]
    fn testnet_hrp() {
        let scan = compressed(VECTOR_SCAN);
        let spend = compressed(VECTOR_SPEND);
        let address = encode(Network::Testnet.hrp(), &scan, &spend);
        assert!(address.starts_with("tsp1qq"));
        let (network, got_scan, _) = decode(&address).unwrap();
        assert_eq!(network, Network::Testnet);
        assert_eq!(got_scan, scan);
    }

    #[test]
    fn signet_and_regtest_hrps() {
        let scan = compressed(VECTOR_SCAN);
        let spend = compressed(VECTOR_SPEND);
        let signet = encode(Network::Signet.hrp(), &scan, &spend);
        assert_eq!(signet, encode(Network::Testnet.hrp(), &scan, &spend));
        // Testnet and signet share an hrp: a signet address decodes as testnet.
        assert_eq!(decode(&signet).unwrap().0, Network::Testnet);
        let regtest = encode(Network::Regtest.hrp(), &scan, &spend);
        assert!(regtest.starts_with("sprt1qq"), "{regtest}");
        assert_eq!(regtest.len(), VECTOR_ADDRESS.len() + 2);
        let (network, got_scan, got_spend) = decode(&regtest).unwrap();
        assert_eq!(network, Network::Regtest);
        assert_eq!((got_scan, got_spend), (scan, spend));
        assert!(decode(&regtest.to_uppercase()).is_ok());
        // Only the mainnet network uses xpubs.
        assert!(!Network::Mainnet.is_test());
        assert!(Network::Testnet.is_test() && Network::Signet.is_test());
        assert!(Network::Regtest.is_test());
    }

    #[test]
    fn decode_is_strict() {
        let hrp = Network::Mainnet.hrp();
        let err = decode(&raw(hrp, &[2u8; 33])).unwrap_err();
        assert!(err.contains("33 bytes"), "{err}");
        let err = decode(&raw(hrp, &[7u8; 70])).unwrap_err();
        assert!(err.contains("70 bytes"), "{err}");
        let err = decode(&raw(Hrp::parse_unchecked("bc"), &[1u8; 66])).unwrap_err();
        assert!(err.contains("hrp 'bc'"), "{err}");
        // 66 bytes = 105.6 groups: the last group carries 2 padding bits.
        let mut fes: Vec<Fe32> = [1u8; 66].iter().copied().bytes_to_fes().collect();
        assert_eq!(fes.len(), 106);
        fes[105] = Fe32::try_from(fes[105].to_u8() | 0x03).unwrap();
        let dirty: String = fes
            .into_iter()
            .with_checksum::<Bech32m>(&hrp)
            .with_witness_version(Fe32::Q)
            .chars()
            .collect();
        let err = decode(&dirty).unwrap_err();
        assert!(err.contains("padding"), "{err}");
        // Other witness versions and the bech32 (non-m) checksum are rejected.
        let v1: String = [1u8; 66]
            .iter()
            .copied()
            .bytes_to_fes()
            .with_checksum::<Bech32m>(&hrp)
            .with_witness_version(Fe32::P)
            .chars()
            .collect();
        let err = decode(&v1).unwrap_err();
        assert!(err.contains("version"), "{err}");
        let b32: String = [1u8; 66]
            .iter()
            .copied()
            .bytes_to_fes()
            .with_checksum::<bech32::Bech32>(&hrp)
            .with_witness_version(Fe32::Q)
            .chars()
            .collect();
        assert!(decode(&b32).is_err());
    }
}
