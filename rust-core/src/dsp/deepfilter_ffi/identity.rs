//! Identity of the pinned DeepFilterNet models.
//!
//! The C library aborts the whole process when it cannot load or parse the
//! model it is given, and that cannot be caught from Rust. A model file is
//! therefore compared with the pinned size and SHA-256 (`release-assets.json`)
//! before the library sees it. Hashing runs on the control thread that builds
//! the engine, never inside a realtime region.
//!
//! A file that was verified is not hashed again while its size and
//! modification time stay the same: engines are rebuilt for every model
//! switch and suppressor rebuild.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::SystemTime;

/// Expected size and lowercase hex SHA-256 of a model file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ModelIdentity {
    pub size: u64,
    pub sha256: &'static str,
}

struct VerifiedFile {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
    sha256: String,
}

static VERIFIED: Mutex<Vec<VerifiedFile>> = Mutex::new(Vec::new());

/// Check `path` against `expected`; the error says what differs.
pub(super) fn verify(path: &Path, expected: &ModelIdentity) -> Result<(), String> {
    let mismatch = |detail: String| {
        format!(
            "DeepFilterNet model {} is not the pinned model ({detail})",
            path.display()
        )
    };
    let unreadable = |error: io::Error| {
        format!(
            "DeepFilterNet model {} is unreadable: {error}",
            path.display()
        )
    };
    let metadata = fs::metadata(path).map_err(unreadable)?;
    if metadata.len() != expected.size {
        return Err(mismatch(format!(
            "{} bytes, expected {}",
            metadata.len(),
            expected.size
        )));
    }
    let modified = metadata.modified().ok();
    // Held while hashing so concurrent engine builds hash a file once.
    let mut verified = VERIFIED.lock().unwrap_or_else(PoisonError::into_inner);
    let cached = verified.iter().find(|file| {
        file.path == path && file.size == metadata.len() && Some(file.modified) == modified
    });
    let sha256 = match cached {
        Some(file) => file.sha256.clone(),
        None => {
            let sha256 = hash_file(path).map_err(unreadable)?;
            if let Some(modified) = modified {
                verified.retain(|file| file.path != path);
                verified.push(VerifiedFile {
                    path: path.to_path_buf(),
                    size: metadata.len(),
                    modified,
                    sha256: sha256.clone(),
                });
            }
            sha256
        }
    };
    if sha256 != expected.sha256 {
        return Err(mismatch(format!(
            "SHA-256 {sha256}, expected {}",
            expected.sha256
        )));
    }
    Ok(())
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0_u8; 1 << 16];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            return Ok(hasher.finish_hex());
        }
        hasher.update(&chunk[..read]);
    }
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 (FIPS 180-4). The crate has no hashing dependency and the models
/// are the only thing hashed.
struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    length: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            filled: 0,
            length: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.length += data.len() as u64;
        if self.filled > 0 {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < 64 {
                return;
            }
            compress(&mut self.state, &self.block);
            self.filled = 0;
        }
        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks {
            compress(&mut self.state, block.try_into().expect("64-byte chunk"));
        }
        let rest = blocks.remainder();
        self.block[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    fn finish_hex(mut self) -> String {
        let bits = self.length.wrapping_mul(8);
        let mut padding = [0_u8; 72];
        padding[0] = 0x80;
        // 0x80, then zeros up to 56 bytes into a block, then the 8-byte length.
        let padded_to = if self.filled < 56 { 56 } else { 120 };
        self.update(&padding[..padded_to - self.filled]);
        self.update(&bits.to_be_bytes());
        self.state
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect()
    }
}

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0_u32; 64];
    for (word, bytes) in w.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(choose)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (word, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *word = word.wrapping_add(add);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::deepfilter_ffi::DeepFilterModel;
    use std::time::Duration;

    fn sha256_hex(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finish_hex()
    }

    fn model_file(name: &str, contents: &[u8]) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        // Test fixtures contain no privileged or secret data; uniqueness only
        // prevents parallel test collisions and is not a security boundary.
        // nosemgrep: rust.lang.security.temp-dir.temp-dir
        let dir = std::env::temp_dir().join(format!("audioforge-identity-{name}-{suffix}"));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.tar.gz");
        fs::write(&path, contents).unwrap();
        path
    }

    fn identity_of(contents: &[u8]) -> ModelIdentity {
        ModelIdentity {
            size: contents.len() as u64,
            sha256: Box::leak(sha256_hex(contents).into_boxed_str()),
        }
    }

    #[test]
    fn sha256_matches_the_fips_180_4_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let million_a = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&million_a),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha256_is_independent_of_how_the_input_is_split() {
        let data: Vec<u8> = (0..1_000_u32).map(|n| (n * 7 + 3) as u8).collect();
        let whole = sha256_hex(&data);
        for split in [1, 55, 56, 63, 64, 65, 129] {
            let mut hasher = Sha256::new();
            for piece in data.chunks(split) {
                hasher.update(piece);
            }
            assert_eq!(hasher.finish_hex(), whole, "{split}-byte pieces");
        }
    }

    #[test]
    fn verify_accepts_only_the_pinned_size_and_digest() {
        let contents = b"pinned model stand-in";
        let path = model_file("verify", contents);
        let expected = identity_of(contents);
        assert_eq!(verify(&path, &expected), Ok(()));

        let wrong_size = ModelIdentity {
            size: expected.size + 1,
            ..expected
        };
        assert!(verify(&path, &wrong_size).unwrap_err().contains("bytes"));
        let wrong_digest = ModelIdentity {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..expected
        };
        assert!(verify(&path, &wrong_digest)
            .unwrap_err()
            .contains("SHA-256"));
        assert!(verify(&path.with_file_name("missing.tar.gz"), &expected)
            .unwrap_err()
            .contains("unreadable"));
    }

    #[test]
    fn a_replaced_file_is_hashed_again() {
        let path = model_file("replaced", b"first contents");
        let expected = identity_of(b"first contents");
        assert_eq!(verify(&path, &expected), Ok(()));

        // Same size, different bytes, later modification time.
        fs::write(&path, b"other contents").unwrap();
        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(SystemTime::now() + Duration::from_secs(5))
            .unwrap();
        assert!(verify(&path, &expected).unwrap_err().contains("SHA-256"));
    }

    #[test]
    fn pinned_models_match_the_release_manifest_and_the_build_recipe() {
        let assets = include_str!("../../../../release-assets.json");
        let provenance = include_str!("../../../../build-support/deepfilter/provenance.json");
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let identity = model.identity();
            let entry = assets
                .find(&format!("models/{}", model.filename()))
                .map(|start| &assets[start..])
                .unwrap_or_else(|| panic!("{} missing from release-assets.json", model.filename()));
            let entry = &entry[..entry.find("\"source\"").unwrap()];
            assert!(entry.contains(&format!("\"size\": {},", identity.size)));
            assert!(entry.contains(&format!("\"sha256\": \"{}\"", identity.sha256)));
            let recipe = provenance
                .find(&format!("models/{}", model.filename()))
                .map(|start| &provenance[start..])
                .unwrap_or_else(|| panic!("{} missing from provenance.json", model.filename()));
            let recipe = &recipe[..recipe.find('}').unwrap()];
            assert!(recipe.contains(&format!("\"bytes\": {},", identity.size)));
            assert!(recipe.contains(&format!(
                "\"sha256\": \"{}\"",
                identity.sha256.to_ascii_uppercase()
            )));
        }
    }

    #[test]
    fn bundled_models_pass_their_identity_check_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        for model in [DeepFilterModel::LowLatency, DeepFilterModel::Standard] {
            let path = root.join("models").join(model.filename());
            if !path.is_file() {
                eprintln!("SKIPPED: {} is not present", path.display());
                continue;
            }
            assert_eq!(
                verify(&path, &model.identity()),
                Ok(()),
                "{}",
                path.display()
            );
        }
    }
}
