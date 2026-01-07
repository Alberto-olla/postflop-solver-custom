#!/bin/bash
# Compare two benchmark results
# Usage: ./compare_benchmarks.sh baseline.json new.json

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <baseline.json> <new.json>"
    echo ""
    echo "Example:"
    echo "  $0 benchmarks/baseline.json benchmarks/after_optimization.json"
    exit 1
fi

BASELINE="$1"
NEW="$2"

if ! command -v jq &> /dev/null; then
    echo "Error: jq is required for comparison. Install with: brew install jq"
    exit 1
fi

echo "=== Benchmark Comparison ==="
echo ""

# Extract stats
BASE_MEAN=$(jq -r '.results[0].mean' "$BASELINE")
BASE_STDDEV=$(jq -r '.results[0].stddev' "$BASELINE")
NEW_MEAN=$(jq -r '.results[0].mean' "$NEW")
NEW_STDDEV=$(jq -r '.results[0].stddev' "$NEW")

# Calculate difference
DIFF=$(echo "$NEW_MEAN - $BASE_MEAN" | bc -l)
PCT=$(echo "scale=2; ($DIFF / $BASE_MEAN) * 100" | bc -l)

echo "Baseline: $(printf '%.3f' $BASE_MEAN)s (±$(printf '%.3f' $BASE_STDDEV)s)"
echo "New:      $(printf '%.3f' $NEW_MEAN)s (±$(printf '%.3f' $NEW_STDDEV)s)"
echo ""

if (( $(echo "$DIFF < 0" | bc -l) )); then
    echo "Result: ✅ FASTER by $(printf '%.3f' $(echo "-1 * $DIFF" | bc -l))s (${PCT}%)"
elif (( $(echo "$DIFF > 0" | bc -l) )); then
    echo "Result: ❌ SLOWER by $(printf '%.3f' $DIFF)s (+${PCT}%)"
else
    echo "Result: ⚖️  NO CHANGE"
fi

# Statistical significance check (rough: if diff > 2*max_stddev)
MAX_STDDEV=$(echo "if ($BASE_STDDEV > $NEW_STDDEV) $BASE_STDDEV else $NEW_STDDEV" | bc -l)
THRESHOLD=$(echo "$MAX_STDDEV * 2" | bc -l)
ABS_DIFF=$(echo "if ($DIFF < 0) -1 * $DIFF else $DIFF" | bc -l)

echo ""
if (( $(echo "$ABS_DIFF > $THRESHOLD" | bc -l) )); then
    echo "Confidence: HIGH (diff > 2σ)"
else
    echo "Confidence: LOW (diff within noise, may not be significant)"
fi
