//! BIP352 v0 silent payment address encoding via the `bech32` crate.
//!
//! `address = bech32m(hrp, [version 0] ++ convertbits8→5(ser_P(B_scan) ++ ser_P(B_m)))`.

use bech32::primitives::decode::CheckedHrpstring;
use bech32::{Bech32m, ByteIterExt, Fe32, Fe32IterExt, Hrp};

pub const HRP_MAINNET: Hrp = Hrp::parse_unchecked("sp");
pub const HRP_TESTNET: Hrp = Hrp::parse_unchecked("tsp");

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

/// Decodes a bech32m string into `(hrp, version, payload bytes)`.
/// Used only for the final independent sanity check of a found key.
pub fn decode(address: &str) -> Result<(Hrp, Fe32, Vec<u8>), String> {
    let mut parsed = CheckedHrpstring::new::<Bech32m>(address).map_err(|e| e.to_string())?;
    let version = parsed
        .remove_witness_version()
        .ok_or_else(|| "address has no version character".to_string())?;
    let payload: Vec<u8> = parsed.fe32_iter().fes_to_bytes().collect();
    Ok((parsed.hrp(), version, payload))
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

    const VECTOR_SCAN: &str = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c";
    const VECTOR_SPEND: &str = "9d6ad855ce3417ef84e836892e5a56392bfba05fa5d97ccea30e266f540e08b3";
    const VECTOR_ADDRESS: &str = "sp1qqgste7k9hx0qftg6qmwlkqtwuy6cycyavzmzj85c6qdfhjdpdjtdgqjuexzk6murw56suy3e0rd2cgqvycxttddwsvgxe2usfpxumr70xc9pkqwv";

    #[test]
    fn bip352_test_vector() {
        let scan = compressed(VECTOR_SCAN);
        let spend = compressed(VECTOR_SPEND);
        let address = encode(HRP_MAINNET, &scan, &spend);
        assert_eq!(address, VECTOR_ADDRESS);
        assert_eq!(address.len(), 116);
    }

    #[test]
    fn decode_roundtrip() {
        let scan = compressed(VECTOR_SCAN);
        let spend = compressed(VECTOR_SPEND);
        let (hrp, version, payload) = decode(VECTOR_ADDRESS).unwrap();
        assert_eq!(hrp, HRP_MAINNET);
        assert_eq!(version, Fe32::Q);
        assert_eq!(payload.len(), 66);
        assert_eq!(&payload[..33], &scan);
        assert_eq!(&payload[33..], &spend);
    }

    #[test]
    fn testnet_hrp() {
        let scan = compressed(VECTOR_SCAN);
        let spend = compressed(VECTOR_SPEND);
        let address = encode(HRP_TESTNET, &scan, &spend);
        assert!(address.starts_with("tsp1qq"));
        let (hrp, _, payload) = decode(&address).unwrap();
        assert_eq!(hrp, HRP_TESTNET);
        assert_eq!(&payload[..33], &scan);
    }
}
