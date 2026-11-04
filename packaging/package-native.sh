#!/usr/bin/env bash
set -euo pipefail
binary=${1:?binary path required}
variant=${2:-cpu}
output=${3:-dist}
version=$(python3 -c 'import json; print(json.load(open("release.json"))["apps"]["media-engine"]["version"])')
if [ "$variant" = cpu ]; then
  test "$("$binary" --version)" = "media-engine-native $version"
fi
case "$variant" in cpu) suffix=linux-x64 ;; cuda) suffix=linux-x64-cuda ;; *) exit 2 ;; esac
mkdir -p "$output"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
install -m 0755 "$binary" "$stage/media-engine-native"
cp LICENSE "$stage/LICENSE"
cp native/README.md "$stage/README.md"
cp native/Cargo.lock "$stage/Cargo.lock"
if [ "$variant" = cuda ]; then
  cp "$(dirname "$binary")/THIRD-PARTY-LICENSES.txt" "$stage/THIRD-PARTY-LICENSES.txt"
else
  python3 packaging/licenses.py --kind rust --output "$stage/THIRD-PARTY-LICENSES.txt"
fi
tar -czf "$output/media-engine-native-v$version-$suffix.tar.gz" -C "$stage" media-engine-native LICENSE README.md Cargo.lock THIRD-PARTY-LICENSES.txt
