#!/bin/bash
# ============================================================================
# Allocator Comparison Benchmark
# ============================================================================
# Compares performance of different allocator configurations:
# 1. Default (system allocator) - stable Rust
# 2. mimalloc - stable Rust
# 3. custom-alloc - nightly Rust
# 4. mimalloc + custom-alloc - nightly Rust
#
# Usage: ./benchmark_allocators.sh [config_file] [runs]
# Example: ./benchmark_allocators.sh benchmark_configs/quick_no_pruning.toml 5
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="${1:-benchmark_configs/quick_no_pruning.toml}"
RUNS="${2:-5}"
BENCHMARK_DIR="benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$BENCHMARK_DIR"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}              ALLOCATOR COMPARISON BENCHMARK${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "Config file: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Runs per config: ${GREEN}$RUNS${NC}"
echo -e "Output dir: ${GREEN}$BENCHMARK_DIR${NC}"
echo ""

# Check if hyperfine is installed
if ! command -v hyperfine &> /dev/null; then
    echo -e "${RED}Error: hyperfine is not installed${NC}"
    echo "Install with: brew install hyperfine"
    exit 1
fi

# Check if nightly is available
NIGHTLY_AVAILABLE=true
if ! rustup run nightly rustc --version &> /dev/null; then
    echo -e "${YELLOW}Warning: Nightly Rust not available. Skipping custom-alloc tests.${NC}"
    NIGHTLY_AVAILABLE=false
fi

# Array to store binary paths and labels
declare -a BINARIES
declare -a LABELS

# ============================================================================
# Build all configurations
# ============================================================================

echo -e "${YELLOW}Building configurations...${NC}"
echo ""

# 1. Default allocator (stable)
echo -e "  [1/4] Building ${GREEN}default${NC} (stable)..."
cargo build --release --example solve_from_config 2>/dev/null
cp target/release/examples/solve_from_config target/release/examples/solve_default
BINARIES+=("./target/release/examples/solve_default")
LABELS+=("default")

# 2. mimalloc (stable)
echo -e "  [2/4] Building ${GREEN}mimalloc${NC} (stable)..."
cargo build --release --example solve_from_config --features mimalloc 2>/dev/null
cp target/release/examples/solve_from_config target/release/examples/solve_mimalloc
BINARIES+=("./target/release/examples/solve_mimalloc")
LABELS+=("mimalloc")

# 3. custom-alloc (nightly)
if [ "$NIGHTLY_AVAILABLE" = true ]; then
    echo -e "  [3/4] Building ${GREEN}custom-alloc${NC} (nightly)..."
    cargo +nightly build --release --example solve_from_config --features custom-alloc 2>/dev/null
    cp target/release/examples/solve_from_config target/release/examples/solve_custom_alloc
    BINARIES+=("./target/release/examples/solve_custom_alloc")
    LABELS+=("custom-alloc")

    # 4. mimalloc + custom-alloc (nightly)
    echo -e "  [4/4] Building ${GREEN}mimalloc+custom-alloc${NC} (nightly)..."
    cargo +nightly build --release --example solve_from_config --features "mimalloc,custom-alloc" 2>/dev/null
    cp target/release/examples/solve_from_config target/release/examples/solve_mimalloc_custom
    BINARIES+=("./target/release/examples/solve_mimalloc_custom")
    LABELS+=("mimalloc+custom-alloc")
else
    echo -e "  [3/4] ${YELLOW}Skipped${NC} custom-alloc (nightly not available)"
    echo -e "  [4/4] ${YELLOW}Skipped${NC} mimalloc+custom-alloc (nightly not available)"
fi

echo ""
echo -e "${GREEN}All builds completed!${NC}"
echo ""

# ============================================================================
# Run benchmarks
# ============================================================================

echo -e "${YELLOW}Running benchmarks...${NC}"
echo ""

OUTPUT_JSON="$BENCHMARK_DIR/allocator_comparison_${TIMESTAMP}.json"
OUTPUT_MD="$BENCHMARK_DIR/allocator_comparison_${TIMESTAMP}.md"

# Build hyperfine command
HYPERFINE_ARGS=(
    --runs "$RUNS"
    --warmup 1
    --export-json "$OUTPUT_JSON"
    --export-markdown "$OUTPUT_MD"
)

# Add each configuration
for i in "${!BINARIES[@]}"; do
    HYPERFINE_ARGS+=(--command-name "${LABELS[$i]}" "${BINARIES[$i]} $CONFIG_FILE")
done

# Run hyperfine
hyperfine "${HYPERFINE_ARGS[@]}"

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}                         RESULTS SUMMARY${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Display results if jq is available
if command -v jq &> /dev/null; then
    echo "Configuration          | Mean (s)  | Stddev   | Min      | Max"
    echo "-----------------------|-----------|----------|----------|----------"

    jq -r '.results[] | "\(.command) | \(.mean | . * 1000 | floor / 1000) | \(.stddev | . * 1000 | floor / 1000) | \(.min | . * 1000 | floor / 1000) | \(.max | . * 1000 | floor / 1000)"' "$OUTPUT_JSON" | \
    while IFS='|' read -r cmd mean stddev min max; do
        printf "%-22s | %9s | %8s | %8s | %8s\n" "$cmd" "$mean" "$stddev" "$min" "$max"
    done

    echo ""

    # Calculate speedup relative to default
    DEFAULT_MEAN=$(jq -r '.results[] | select(.command == "default") | .mean' "$OUTPUT_JSON")
    if [ -n "$DEFAULT_MEAN" ]; then
        echo -e "${YELLOW}Speedup vs default allocator:${NC}"
        jq -r --arg default "$DEFAULT_MEAN" '.results[] | "\(.command): \(($default | tonumber) / .mean | . * 100 | floor / 100)x"' "$OUTPUT_JSON"
    fi
fi

echo ""
echo -e "${GREEN}Results saved to:${NC}"
echo "  JSON: $OUTPUT_JSON"
echo "  Markdown: $OUTPUT_MD"
echo ""

# Cleanup temporary binaries
echo -e "${YELLOW}Cleaning up temporary binaries...${NC}"
rm -f target/release/examples/solve_default
rm -f target/release/examples/solve_mimalloc
rm -f target/release/examples/solve_custom_alloc
rm -f target/release/examples/solve_mimalloc_custom

echo -e "${GREEN}Done!${NC}"