#!/usr/bin/env bash
# Differential harness: feed identical inputs to the shipped libysm-core.so
# (golden) and the freshly built one, byte-compare all observable outputs.
#
# Usage: ./harness/run.sh <path-to-new-libysm-core.so> [case ...]
set -u

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
GOLDEN_SO="${GOLDEN_SO:-/home/brokestar/workspace/ModernYSM/common/src/main/resources/natives/linux-x64/libysm-core.so}"
NEW_SO="${1:?usage: run.sh <new-libysm-core.so> [case...]}"
shift || true
CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then
  CASES=(normal translucent glow partMask hiddenSubtree emptyMesh state zeroScale culled gpuMesh)
fi

BUILD="$HARNESS_DIR/build"
mkdir -p "$BUILD/out"
find "$HARNESS_DIR/mock" -name '*.java' > "$BUILD/sources.txt"
echo "$HARNESS_DIR/Runner.java" >> "$BUILD/sources.txt"
javac -d "$BUILD/classes" @"$BUILD/sources.txt" || exit 1

if ! grep -q avx2 /proc/cpuinfo; then
  echo "WARNING: host CPU reports no AVX2; golden .so JNI_OnLoad will refuse to load." >&2
fi

total=0; pass=0; fail=0; failed_cases=()
for c in "${CASES[@]}"; do
  total=$((total+1))
  YSM_LIB="$GOLDEN_SO" java -cp "$BUILD/classes" "Runner" "$c" "$BUILD/out/$c.golden" 2>"$BUILD/out/$c.golden.err"
  g=$?
  YSM_LIB="$NEW_SO"   java -cp "$BUILD/classes" "Runner" "$c" "$BUILD/out/$c.new"    2>"$BUILD/out/$c.new.err"
  n=$?
  if [ $g -eq 0 ] && [ $n -eq 0 ] && diff -q "$BUILD/out/$c.golden" "$BUILD/out/$c.new" >/dev/null; then
    echo "PASS $c"; pass=$((pass+1))
  else
    echo "FAIL $c (golden=$g new=$n)"; fail=$((fail+1)); failed_cases+=("$c")
    diff "$BUILD/out/$c.golden" "$BUILD/out/$c.new" | head -20 | sed 's/^/    /'
  fi
done

echo "----------------------------------------"
echo "harness: $pass/$total passed"
[ $fail -gt 0 ] && printf 'failed: %s\n' "${failed_cases[*]}"
exit $fail
