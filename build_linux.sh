#!/usr/bin/env bash
# Build the self-contained Linux bundle in dist/AudioForge, the counterpart of
# build_exe.ps1. Package it afterwards with python/tools/linux_package.py.
#
# Usage: ./build_linux.sh [--clean]   (AUDIOFORGE_PYTHON overrides .venv/bin/python)
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
python=${AUDIOFORGE_PYTHON:-$root/.venv/bin/python}
pyinstaller_args=(-y)
if [[ ${1:-} == --clean ]]; then
    pyinstaller_args+=(--clean)
fi

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -x $python ]] || fail "build Python not found: $python"
"$python" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)' \
    || fail "build Python must be inside a virtual environment: $python"
echo "Using build Python: $python"
# maturin develop installs into the environment VIRTUAL_ENV names.
VIRTUAL_ENV=$("$python" -c 'import sys; print(sys.prefix)')
export VIRTUAL_ENV

ort=target/onnxruntime-cpu/lib/libonnxruntime.so.1.23.2
[[ -f $ort ]] || fail "$ort is missing; run python/tools/fetch_release_assets.py"

"$python" python/tools/verify_release_assets.py \
    || fail "release asset verification failed"

"$python" -m maturin develop --release --locked >/dev/null \
    || fail "Rust extension rebuild failed"
suffix=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
[[ -f python/mic_eq/mic_eq_core$suffix ]] || fail "python/mic_eq/mic_eq_core$suffix not found"

[[ -f libdf.so ]] || fail "libdf.so not found; run python/tools/build_deepfilter.py and copy it here"
for model in models/DeepFilterNet3_ll_onnx.tar.gz models/DeepFilterNet3_onnx.tar.gz models/silero_vad.onnx; do
    [[ -f $model ]] || fail "missing model asset: $model"
done

"$python" python/tools/license_inventory.py --platform linux \
    || fail "dependency license collection failed"

"$python" -m PyInstaller "${pyinstaller_args[@]}" AudioForge.spec

version=$("$python" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
printf '{\n  "schema_version": 1,\n  "version": "%s"\n}\n' "$version" \
    > dist/AudioForge/_internal/audioforge-build.json

echo
echo "Bundle: dist/AudioForge/AudioForge (version $version)"
