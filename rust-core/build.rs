use std::env;
use std::fs;
use std::path::PathBuf;

const RUNTIME_FILES: [&str; 2] = ["onnxruntime.dll", "onnxruntime_providers_shared.dll"];
const LINK_LIBRARY: &str = "onnxruntime.lib";

fn main() {
    println!("cargo:rerun-if-env-changed=ORT_LIB_LOCATION");
    println!("cargo:rerun-if-env-changed=ORT_PREFER_DYNAMIC_LINK");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || env::var_os("CARGO_FEATURE_VAD").is_none()
    {
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let project_root = manifest_dir
        .parent()
        .expect("rust-core must be directly under the project root");
    let configured = env::var_os("ORT_LIB_LOCATION")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.join("target/onnxruntime-cpu/lib"));
    let lib_dir = if configured.is_absolute() {
        configured
    } else {
        project_root.join(configured)
    };

    for name in RUNTIME_FILES {
        let source = lib_dir.join(name);
        println!("cargo:rerun-if-changed={}", source.display());
        if !source.is_file() {
            panic!(
                "pinned CPU ONNX Runtime asset is missing: {}; run `python python/tools/fetch_release_assets.py --only-cpu-runtime`",
                source.display()
            );
        }
    }
    let link_library = lib_dir.join(LINK_LIBRARY);
    println!("cargo:rerun-if-changed={}", link_library.display());
    if !link_library.is_file() {
        panic!(
            "pinned CPU ONNX Runtime import library is missing: {}; run `python python/tools/fetch_release_assets.py --only-cpu-runtime`",
            link_library.display()
        );
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR must be under target/<profile>/build")
        .to_path_buf();
    for destination_dir in [profile_dir.clone(), profile_dir.join("deps")] {
        fs::create_dir_all(&destination_dir).unwrap_or_else(|error| {
            panic!(
                "could not create Cargo runtime directory {}: {error}",
                destination_dir.display()
            )
        });
        for name in RUNTIME_FILES {
            let source = lib_dir.join(name);
            let destination = destination_dir.join(name);
            fs::copy(&source, &destination).unwrap_or_else(|error| {
                panic!(
                    "could not place pinned CPU ONNX Runtime asset at {}: {error}",
                    destination.display()
                )
            });
        }
    }
}
