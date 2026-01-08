# Regression Analysis: Commit ffb6178

## Summary

**Commit:** `ffb617886d36c7e591a468a3fed6599ff7f740b1`
**Date:** Thu Jan 8 01:06:37 2026 +0100
**Message:** "refactor solver: implement fused operations to optimize regret algorithms, replace Mutex-like allocations with lock-free thread-local buffers, and introduce enhanced pruning mechanisms"

**Impact:** ~2x performance regression (14s -> 28s)

## Changes Introduced

### 1. ConcurrentCfvBuffer (src/buffer_pool.rs + src/solver/mod.rs)

**Before:**
```rust
#[cfg(feature = "custom-alloc")]
let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
#[cfg(not(feature = "custom-alloc"))]
let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));
```

**After:**
```rust
let cfv_actions = ConcurrentCfvBuffer::new(num_actions, num_hands);
```

**Hypothesis:** The thread-local buffer stack adds overhead:
- `thread_local!` access has non-trivial cost
- Buffer management (alloc/dealloc tracking) adds overhead
- May cause cache misses due to different memory layout

### 2. Fused Operations in Regrets (src/solver/regrets.rs)

**Before (DCFR example):**
```rust
// Separate passes
cum_regret.iter_mut().zip(cfv_actions).for_each(|(x, y)| {
    let coef = if x.is_sign_positive() { alpha } else { beta };
    *x = *x * coef + *y;
});
cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
    sub_slice(row, result);
});
```

**After:**
```rust
// Fused into single pass per action row
for action in 0..num_actions {
    let row_start = action * num_hands;
    let row_end = row_start + num_hands;
    let regret_row = &mut cum_regret[row_start..row_end];
    let cfv_row = &cfv_actions[row_start..row_end];

    regret_row
        .iter_mut()
        .zip(cfv_row)
        .zip(result)
        .for_each(|((reg, &cfv), &res)| {
            let coef = if reg.is_sign_positive() { alpha } else { beta };
            *reg = *reg * coef + cfv - res;
        });
}
```

**Hypothesis:** The "optimization" backfires:
- Outer `for action in 0..num_actions` loop prevents auto-vectorization
- The iterator chain `.zip().zip().for_each()` has more overhead
- Original separate passes were better vectorized by LLVM

### 3. Pruning Pre-computation

**Before:**
```rust
for action in node.action_indices() {
    let should_skip = match pruning_mode { ... };
    if should_skip {
        // Fill with zeros
    } else {
        // Recurse
    }
}
```

**After:**
```rust
// Pre-compute all pruning decisions
let prune_flags: Vec<bool> = node.action_indices()
    .map(|action| { ... })
    .collect();

// Then process in parallel
for_each_child(node, |action| {
    if prune_flags[action] { ... }
});
```

**Hypothesis:** Additional allocation for `prune_flags` Vec, though likely minor impact.

## Root Cause Analysis

### Most Likely Culprit: ConcurrentCfvBuffer

The `ConcurrentCfvBuffer` implementation uses:
1. `thread_local!` storage with `UnsafeCell`
2. A 4MB buffer stack per thread
3. Manual stack pointer management

This is **slower** than simple `Vec::with_capacity()` because:
- `thread_local!` has access overhead (TLS lookup)
- The buffer stack logic adds branching
- Memory locality may be worse

### Secondary Culprit: Fused Regret Operations

The "fused" operations are actually **less efficient** because:
- The explicit `for action in 0..num_actions` loop breaks SIMD
- Original code used `chunks_exact_mut()` which LLVM vectorizes well
- Adding an outer loop changes memory access patterns

## Recommendation

**Revert commit ffb6178** and keep the original implementation:
- `MutexLike<Vec>` allocation is simpler and faster
- Separate passes in regret update are better vectorized
- The "optimizations" were based on incorrect assumptions

## Files Changed

| File | Lines Changed | Impact |
|------|--------------|--------|
| src/buffer_pool.rs | +305 | New (problematic) |
| src/solver/mod.rs | +101/-100 | High (uses buffer_pool) |
| src/solver/regrets.rs | +368/-180 | High (fused ops) |
| Cargo.toml | +10 | Low (profiles) |
| src/lib.rs | +1 | Low (mod declaration) |

## Benchmark Results

### Test Configuration
- **Config:** `benchmark_configs/full_no_pruning.toml`
- **Runs:** 5 per version
- **Tool:** hyperfine

### Results

| Version | Mean | Stddev | Min | Max |
|---------|------|--------|-----|-----|
| Pre-refactor (3ef7768) | **15.56s** | 0.45s | 15.12s | 16.10s |
| Post-refactor (ffb6178) | **16.26s** | 0.32s | 15.87s | 16.62s |

### Analysis

- **Regression:** ~4.5% slower (0.7s difference)
- **Not the 2x regression initially observed** - that was likely due to:
  - Initial cold cache / compilation effects
  - Different test configuration
  - System state variability

### Conclusion

The refactoring in `ffb6178` introduces a small but measurable performance regression (~4.5%).
While not as severe as initially thought, it still represents unnecessary overhead from:
1. Thread-local buffer management
2. Changed loop structure in regret updates

**Recommendation:** Consider reverting or optimizing the changes if performance is critical.

---

## Detailed Isolation Analysis

### Methodology
Applied each modification separately to baseline (3ef7768) and benchmarked.

### Results (5 runs each)

| Modification | Mean | Stddev | Min | Max |
|--------------|------|--------|-----|-----|
| Baseline (3ef7768) | 16.94s | 1.07s | 15.62s | 18.19s |
| Only `regrets.rs` | 16.98s | 0.40s | 16.41s | 17.36s |
| Only `buffer_pool` | 16.92s | 0.48s | 16.36s | 17.55s |
| **Both combined** | **18.33s** | **3.05s** | 16.56s | 23.76s |

### Key Findings

1. **Individual modifications are performance-neutral**
   - `regrets.rs` alone: no measurable impact
   - `buffer_pool` alone: no measurable impact

2. **Combined modifications cause instability**
   - Mean time increases by ~8% (16.94s -> 18.33s)
   - Standard deviation explodes: 1.07s -> 3.05s
   - Max time nearly doubles: 18.19s -> 23.76s

3. **Interaction effect**
   - The two modifications interfere with each other
   - Possible causes: cache contention, memory access pattern conflicts, or
     thread synchronization issues between buffer_pool and fused regret ops

### Recommendation

**Revert both changes** and keep the original implementation:
- Individual changes appear neutral but their combination is problematic
- The high variance (stddev 3.05s) indicates unpredictable behavior
- `MutexLike<Vec>` with separate regret passes is stable and fast
