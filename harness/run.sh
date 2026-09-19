#!/usr/bin/env bash
# Differential harness: feed identical inputs to the shipped libysm-core.so
# (golden) and the freshly built one, byte-compare all observable outputs.
#
# Two phases:
#   drift:     golden vs new must be byte-identical (no semantics change).
#              Cases touching offset9 / non-tree defence are semantic-only,
#              they diverge from golden BY DESIGN (golden doesn't consume
#              offset9; golden crashes on non-tree input).
#   semantic:  new lib only, real assertions (exit 1 on mismatch).
#
# Usage: ./harness/run.sh <path-to-new-libysm-core.so> [case ...]
set -u

HARNESS_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${GOLDEN_SO:-}" ]; then
  for c in \
    /home/brokestar/workspace/ModernYSM/src/main/resources/natives/linux-x64/libysm-core.so \
    /home/brokestar/workspace/ModernYSM/common/src/main/resources/natives/linux-x64/libysm-core.so; do
    [ -f "$c" ] && GOLDEN_SO="$c" && break
  done
fi
GOLDEN_SO="${GOLDEN_SO:?set GOLDEN_SO=<shipped libysm-core.so>}"
NEW_SO="${1:?usage: run.sh <new-libysm-core.so> [case...]}"
shift || true
CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then
  CASES=(normal translucent glow partMask hiddenSubtree emptyMesh state zeroScale culled gpuMultiQuad)
fi
SEMANTIC_CASES=(offset9Self offset9Root offset9Midtree offset9PlusSkip10 offset9GpuOnly nontreeBadParent nontreeCycle stagingFast)

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

sem_total=0; sem_pass=0; sem_failed=()
for c in "${SEMANTIC_CASES[@]}"; do
  sem_total=$((sem_total+1))
  YSM_SEMANTIC=1 YSM_LIB="$NEW_SO" java -cp "$BUILD/classes" "Runner" "$c" "$BUILD/out/$c.semantic" 2>"$BUILD/out/$c.semantic.err"
  if grep -q "semantic.result=PASS" "$BUILD/out/$c.semantic" 2>/dev/null; then
    echo "SEMANTIC-PASS $c (golden diverges by design: offset9 unconsumed / non-tree crash)"; sem_pass=$((sem_pass+1))
  else
    echo "SEMANTIC-FAIL $c"; sem_failed+=("$c")
    grep "semantic.result" "$BUILD/out/$c.semantic" 2>/dev/null | sed 's/^/    /'
  fi
done

echo "----------------------------------------"
echo "harness drift: $pass/$total passed; semantic: $sem_pass/$sem_total passed"
rc=0
[ $fail -gt 0 ] && { printf 'drift failed: %s\n' "${failed_cases[*]}"; rc=1; }
[ $sem_pass -ne $sem_total ] && { printf 'semantic failed: %s\n' "${sem_failed[*]}"; rc=1; }
exit $rc
