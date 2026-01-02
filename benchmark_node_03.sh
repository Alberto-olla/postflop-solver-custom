#!/bin/bash

echo "Starting benchmarks..."

# Create directory for logs
mkdir -p benchmark_logs

# Run DCFR
echo "Running DCFR..."
cargo run --release --example solve_from_config benchmark_dcfr.toml > benchmark_logs/dcfr.log 2>&1

# Run DCFR+
echo "Running DCFR+..."
cargo run --release --example solve_from_config benchmark_dcfr_plus.toml > benchmark_logs/dcfr_plus.log 2>&1

# Run SAPCFR+
echo "Running SAPCFR+..."
cargo run --release --example solve_from_config benchmark_sapcfr_plus.toml > benchmark_logs/sapcfr_plus.log 2>&1

echo "Benchmarks completed. Extracting results..."

echo "DCFR Results:"
grep "exploitability =" benchmark_logs/dcfr.log | tail -n 1
echo "DCFR+ Results:"
grep "exploitability =" benchmark_logs/dcfr_plus.log | tail -n 1
echo "SAPCFR+ Results:"
grep "exploitability =" benchmark_logs/sapcfr_plus.log | tail -n 1
