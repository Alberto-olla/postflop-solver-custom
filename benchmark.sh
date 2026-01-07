#!/bin/bash
# Benchmark script for postflop-solver
# Usage: ./benchmark.sh [config_file] [runs] [name]
#
# Examples:
#   ./benchmark.sh                                    # Default: turn config, 10 runs
#   ./benchmark.sh hands/7438/configs/hand_*.toml 20 # Specific config, 20 runs
#   ./benchmark.sh config.toml 15 my_test            # With custom name

set -e

# Defaults
CONFIG_FILE="${1:-hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml}"
RUNS="${2:-10}"
NAME="${3:-}"

# Create benchmarks directory
BENCHMARK_DIR="benchmarks"
mkdir -p "$BENCHMARK_DIR"

# Generate timestamp and filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .toml)

if [ -n "$NAME" ]; then
    OUTPUT_NAME="${NAME}_${TIMESTAMP}"
else
    OUTPUT_NAME="${CONFIG_BASENAME}_${TIMESTAMP}"
fi

JSON_FILE="$BENCHMARK_DIR/${OUTPUT_NAME}.json"
MD_FILE="$BENCHMARK_DIR/${OUTPUT_NAME}.md"

echo "=== Postflop Solver Benchmark ==="
echo "Config: $CONFIG_FILE"
echo "Runs: $RUNS"
echo "Output: $JSON_FILE"
echo ""

# Build first (not counted in benchmark)
echo "Building release..."
cargo build --release --example solve_from_config 2>/dev/null

echo ""
echo "Running benchmark..."
echo ""

# Run hyperfine
hyperfine \
    --runs "$RUNS" \
    --warmup 1 \
    --export-json "$JSON_FILE" \
    --export-markdown "$MD_FILE" \
    --command-name "solve_from_config" \
    "./target/release/examples/solve_from_config $CONFIG_FILE"

echo ""
echo "=== Results saved ==="
echo "  JSON: $JSON_FILE"
echo "  Markdown: $MD_FILE"
echo ""

# Extract and display key stats from JSON
if command -v jq &> /dev/null; then
    echo "=== Summary ==="
    jq -r '.results[0] | "Mean: \(.mean | . * 1000 | floor / 1000)s | Stddev: \(.stddev | . * 1000 | floor / 1000)s | Min: \(.min | . * 1000 | floor / 1000)s | Max: \(.max | . * 1000 | floor / 1000)s"' "$JSON_FILE"
fi
