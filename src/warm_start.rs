use crate::action_tree::*;
use crate::game::*;
use crate::interface::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Interpolation mode for bet size mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationMode {
    /// Linear interpolation based on pot percentage
    #[default]
    Linear,
    /// Logarithmic interpolation (better for strategic distance)
    Logarithmic,
}

/// Result of action mapping between small and large trees
#[derive(Debug, Clone)]
pub enum ActionMatch {
    /// Direct 1:1 correspondence between actions
    Direct(usize),

    /// Interpolation between two actions
    Interpolated {
        low: usize,  // Index of lower bet size action
        high: usize, // Index of higher bet size action
        weight: f32, // Interpolation weight [0, 1], favoring high
    },

    /// Nearest neighbor (extrapolation fallback)
    Nearest(usize),
}

/// Weight mode for warm-start transfer
#[derive(Debug, Clone, PartialEq)]
pub enum WarmStartWeightMode {
    /// Fixed weight value
    Fixed(f32),
    /// Adaptive weight based on convergence quality
    Adaptive {
        base_weight: f32,
        target_exploitability_pct: f32,
    },
    /// Automatic search for optimal weight (minimizes exploitability)
    Auto {
        /// Candidate weights to try
        candidates: Vec<f32>,
    },
}

impl Default for WarmStartWeightMode {
    fn default() -> Self {
        WarmStartWeightMode::Fixed(10.0)
    }
}

/// Configuration for warm-start transfer
#[derive(Debug, Clone)]
pub struct WarmStartConfig {
    /// Interpolation mode for bet sizing
    pub interpolation_mode: InterpolationMode,
    /// Enable parallel transfer (requires rayon feature)
    pub parallel: bool,
    /// Weight mode (fixed, adaptive, or auto-search)
    pub weight_mode: WarmStartWeightMode,
}

impl Default for WarmStartConfig {
    fn default() -> Self {
        Self {
            interpolation_mode: InterpolationMode::Logarithmic,
            parallel: true,
            weight_mode: WarmStartWeightMode::Fixed(10.0),
        }
    }
}

impl WarmStartConfig {
    /// Create config with fixed weight (backward compatible)
    pub fn with_fixed_weight(weight: f32) -> Self {
        Self {
            weight_mode: WarmStartWeightMode::Fixed(weight),
            ..Default::default()
        }
    }

    /// Create config with adaptive weight
    pub fn with_adaptive_weight(base_weight: f32, target_expl_pct: f32) -> Self {
        Self {
            weight_mode: WarmStartWeightMode::Adaptive {
                base_weight,
                target_exploitability_pct: target_expl_pct,
            },
            ..Default::default()
        }
    }

    /// Create config with automatic weight search
    pub fn with_auto_weight() -> Self {
        Self {
            weight_mode: WarmStartWeightMode::Auto {
                candidates: vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            },
            ..Default::default()
        }
    }

    /// Create config with custom candidate weights for auto search
    pub fn with_auto_weight_candidates(candidates: Vec<f32>) -> Self {
        Self {
            weight_mode: WarmStartWeightMode::Auto { candidates },
            ..Default::default()
        }
    }

    // Legacy compatibility getters
    pub fn base_weight(&self) -> f32 {
        match &self.weight_mode {
            WarmStartWeightMode::Fixed(w) => *w,
            WarmStartWeightMode::Adaptive { base_weight, .. } => *base_weight,
            WarmStartWeightMode::Auto { .. } => 10.0,
        }
    }

    pub fn adaptive_weight(&self) -> bool {
        matches!(self.weight_mode, WarmStartWeightMode::Adaptive { .. })
    }

    pub fn target_exploitability_pct(&self) -> f32 {
        match &self.weight_mode {
            WarmStartWeightMode::Adaptive { target_exploitability_pct, .. } => *target_exploitability_pct,
            _ => 0.5,
        }
    }
}

/// Entry point for warm-start transfer (legacy API, backward compatible)
///
/// Transfers accumulated regrets from a solved small tree to a large tree,
/// using linear interpolation for parametric actions (bet/raise).
///
/// # Arguments
/// * `small_game` - The solved minimal tree
/// * `large_game` - The target full tree (must be ready, with memory allocated)
/// * `small_iterations` - Number of iterations completed on small tree
/// * `warmstart_weight` - Normalization weight (default: 10.0)
///
/// # Returns
/// Ok(()) on success, or error message
pub fn apply_warm_start(
    small_game: &PostFlopGame,
    large_game: &mut PostFlopGame,
    small_iterations: u32,
    warmstart_weight: f32,
) -> Result<(), String> {
    let config = WarmStartConfig {
        weight_mode: WarmStartWeightMode::Fixed(warmstart_weight),
        interpolation_mode: InterpolationMode::Linear, // Legacy uses linear
        parallel: false, // Legacy uses sequential
    };

    apply_warm_start_with_config(small_game, large_game, small_iterations, &config)
        .map(|_| ()) // Discard WarmStartResult for legacy API
}

/// Result of warm-start application, including the effective weight used
#[derive(Debug, Clone)]
pub struct WarmStartResult {
    /// The effective weight that was used
    pub effective_weight: f32,
    /// Exploitability after warm-start (if auto mode was used)
    pub exploitability: Option<f32>,
    /// All candidates tested (if auto mode was used)
    pub search_results: Option<Vec<(f32, f32)>>, // (weight, exploitability)
}

/// Entry point for warm-start transfer with full configuration
///
/// # Arguments
/// * `small_game` - The solved minimal tree
/// * `large_game` - The target full tree (must be ready, with memory allocated)
/// * `small_iterations` - Number of iterations completed on small tree
/// * `config` - Warm-start configuration
///
/// # Returns
/// Ok(WarmStartResult) with the effective weight used, or error message
pub fn apply_warm_start_with_config(
    small_game: &PostFlopGame,
    large_game: &mut PostFlopGame,
    small_iterations: u32,
    config: &WarmStartConfig,
) -> Result<WarmStartResult, String> {
    // Validate compatibility
    validate_games(small_game, large_game)?;

    // Determine effective weight based on mode
    match &config.weight_mode {
        WarmStartWeightMode::Fixed(weight) => {
            apply_warmstart_with_weight(small_game, large_game, small_iterations, *weight, config)?;
            Ok(WarmStartResult {
                effective_weight: *weight,
                exploitability: None,
                search_results: None,
            })
        }
        WarmStartWeightMode::Adaptive { base_weight, target_exploitability_pct } => {
            let effective_weight = calculate_adaptive_weight(
                small_game,
                small_iterations,
                *base_weight,
                *target_exploitability_pct,
            );
            apply_warmstart_with_weight(small_game, large_game, small_iterations, effective_weight, config)?;
            Ok(WarmStartResult {
                effective_weight,
                exploitability: None,
                search_results: None,
            })
        }
        WarmStartWeightMode::Auto { candidates } => {
            search_optimal_weight(small_game, large_game, small_iterations, candidates, config)
        }
    }
}

/// Apply warmstart with a specific weight
fn apply_warmstart_with_weight(
    small_game: &PostFlopGame,
    large_game: &mut PostFlopGame,
    small_iterations: u32,
    weight: f32,
    config: &WarmStartConfig,
) -> Result<(), String> {
    let avg_factor = weight / (small_iterations as f32);

    let ctx = TransferContext {
        small_game,
        large_game,
        avg_factor,
        interpolation_mode: config.interpolation_mode,
    };

    let small_root = small_game.root();
    let mut large_root = large_game.root();

    #[cfg(feature = "rayon")]
    if config.parallel {
        transfer_regrets_parallel(&*small_root, &mut *large_root, &ctx)?;
    } else {
        transfer_regrets_recursive(&*small_root, &mut *large_root, &ctx)?;
    }

    #[cfg(not(feature = "rayon"))]
    transfer_regrets_recursive(&*small_root, &mut *large_root, &ctx)?;

    Ok(())
}

/// Search for the optimal weight using Golden Section Search
///
/// Golden Section Search finds the minimum of a unimodal function in O(log(n)) evaluations.
/// It uses the golden ratio φ = (1 + √5) / 2 ≈ 1.618 to efficiently narrow down the search interval.
///
/// For a search range [1, 200] with tolerance 0.5, this requires ~15 evaluations
/// instead of testing hundreds of candidates.
fn search_optimal_weight(
    small_game: &PostFlopGame,
    large_game: &mut PostFlopGame,
    small_iterations: u32,
    candidates: &[f32],
    config: &WarmStartConfig,
) -> Result<WarmStartResult, String> {
    // Extract search range from candidates (min, max)
    let min_weight = candidates.iter().cloned().fold(f32::INFINITY, f32::min).max(0.1);
    let max_weight = candidates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Tolerance: stop when interval is smaller than this
    let tolerance = 0.5;

    eprintln!("  Golden Section Search: range [{:.1}, {:.1}], tolerance {:.1}",
              min_weight, max_weight, tolerance);

    let result = golden_section_search(
        small_game,
        large_game,
        small_iterations,
        min_weight,
        max_weight,
        tolerance,
        config,
    )?;

    Ok(result)
}

/// Golden Section Search implementation
///
/// The algorithm maintains an interval [a, b] and two interior points c, d.
/// At each iteration, it evaluates f(c) and f(d) and narrows the interval
/// by eliminating the part that cannot contain the minimum.
///
/// The golden ratio ensures that one of the interior points can be reused
/// in the next iteration, reducing the number of function evaluations.
fn golden_section_search(
    small_game: &PostFlopGame,
    large_game: &mut PostFlopGame,
    small_iterations: u32,
    mut a: f32,
    mut b: f32,
    tolerance: f32,
    config: &WarmStartConfig,
) -> Result<WarmStartResult, String> {
    use crate::compute_exploitability;

    // Golden ratio conjugate: φ - 1 = (√5 - 1) / 2 ≈ 0.618
    const GOLDEN_RATIO: f32 = 0.6180339887;

    // Tolerance for detecting flat function (all values equal)
    const FLAT_TOLERANCE: f32 = 0.01;

    let mut results: Vec<(f32, f32)> = Vec::new();
    let mut eval_count = 0;

    // Helper to evaluate exploitability at a given weight
    let mut evaluate = |weight: f32| -> Result<f32, String> {
        apply_warmstart_with_weight(small_game, large_game, small_iterations, weight, config)?;
        let expl = compute_exploitability(large_game);
        reset_regrets(large_game)?;
        eval_count += 1;
        results.push((weight, expl));
        eprintln!("    [{:2}] weight={:7.2} → exploitability={:.6}", eval_count, weight, expl);
        Ok(expl)
    };

    // Initial interior points
    let mut c = b - GOLDEN_RATIO * (b - a);
    let mut d = a + GOLDEN_RATIO * (b - a);

    // Evaluate at initial interior points
    let mut fc = evaluate(c)?;
    let mut fd = evaluate(d)?;

    // Early exit: if first two evaluations are equal, function is likely flat
    // Try one more point at the boundary to confirm
    if (fc - fd).abs() < FLAT_TOLERANCE {
        let fa = evaluate(a)?;

        // Check if all three are equal (flat function)
        if (fa - fc).abs() < FLAT_TOLERANCE && (fa - fd).abs() < FLAT_TOLERANCE {
            eprintln!("  Function is flat (all values equal) - using minimum weight");

            // Use the minimum weight (a) since all give same exploitability
            let best_weight = a;
            let best_expl = fa;

            apply_warmstart_with_weight(small_game, large_game, small_iterations, best_weight, config)?;

            return Ok(WarmStartResult {
                effective_weight: best_weight,
                exploitability: Some(best_expl),
                search_results: Some(results),
            });
        }
    }

    // Main loop: narrow the interval until it's smaller than tolerance
    while (b - a).abs() > tolerance {
        if fc < fd {
            // Minimum is in [a, d]
            b = d;
            d = c;
            fd = fc;
            c = b - GOLDEN_RATIO * (b - a);
            fc = evaluate(c)?;
        } else {
            // Minimum is in [c, b]
            a = c;
            c = d;
            fc = fd;
            d = a + GOLDEN_RATIO * (b - a);
            fd = evaluate(d)?;
        }

        // Early exit if values converged (difference is negligible)
        if (fc - fd).abs() < FLAT_TOLERANCE {
            eprintln!("  Values converged - stopping early");
            break;
        }
    }

    // Find the best result
    let (best_weight, best_expl) = results
        .iter()
        .cloned()
        .min_by(|(_, e1), (_, e2)| e1.partial_cmp(e2).unwrap())
        .unwrap_or(((a + b) / 2.0, f32::MAX));

    eprintln!("  Golden Section complete: {} evaluations", eval_count);
    eprintln!("  Best weight: {:.2} (exploitability: {:.6})", best_weight, best_expl);

    // Apply the best weight permanently
    apply_warmstart_with_weight(small_game, large_game, small_iterations, best_weight, config)?;

    Ok(WarmStartResult {
        effective_weight: best_weight,
        exploitability: Some(best_expl),
        search_results: Some(results),
    })
}

/// Reset all regrets in the game tree to zero
fn reset_regrets(game: &mut PostFlopGame) -> Result<(), String> {
    fn reset_node_recursive(node: &mut PostFlopNode, regret_bits: u8) {
        if node.is_terminal() {
            return;
        }

        // Reset regrets for this node
        if !node.is_chance() {
            match regret_bits {
                32 => {
                    for r in node.regrets_mut() {
                        *r = 0.0;
                    }
                }
                16 => {
                    for r in node.regrets_compressed_mut() {
                        *r = 0;
                    }
                    node.set_regret_scale(0.0);
                }
                8 => {
                    for r in node.regrets_u8_mut() {
                        *r = 0;
                    }
                    node.set_regret_scale(0.0);
                }
                4 => {
                    for r in node.regrets_u8_mut() {
                        *r = 0;
                    }
                    node.set_regret_scale(0.0);
                }
                _ => {}
            }
        }

        // Recurse to children
        for i in 0..node.num_actions() {
            let mut child = node.play(i);
            reset_node_recursive(&mut *child, regret_bits);
        }
    }

    let regret_bits = game.regret_bits();
    let mut root = game.root();
    reset_node_recursive(&mut *root, regret_bits);
    Ok(())
}

/// Calculate adaptive warmstart weight based on minimal tree convergence
///
/// The idea: if the minimal tree converged well (low exploitability),
/// we can trust its regrets more and use a higher weight.
/// If it converged poorly, use a lower weight to let the full tree correct faster.
fn calculate_adaptive_weight(
    small_game: &PostFlopGame,
    small_iterations: u32,
    base_weight: f32,
    target_exploitability_pct: f32,
) -> f32 {
    use crate::compute_exploitability;

    let starting_pot = small_game.tree_config().starting_pot as f32;
    let target_expl = starting_pot * target_exploitability_pct / 100.0;

    // Get actual exploitability of minimal tree
    let actual_expl = compute_exploitability(small_game);

    // Calculate convergence ratio (how well did we converge?)
    // ratio = 1.0 means we hit target exactly
    // ratio < 1.0 means we converged better than target
    // ratio > 1.0 means we didn't converge enough
    let convergence_ratio = actual_expl / target_expl;

    // Adaptive weight formula:
    // - If converged well (ratio < 1): increase weight (trust the regrets more)
    // - If converged poorly (ratio > 1): decrease weight (let full tree correct)
    //
    // We use a smooth function: weight = base_weight / sqrt(convergence_ratio)
    // This gives:
    // - ratio = 0.25 → weight = base_weight * 2.0
    // - ratio = 1.0  → weight = base_weight * 1.0
    // - ratio = 4.0  → weight = base_weight * 0.5
    let adaptive_factor = (1.0 / convergence_ratio.max(0.1)).sqrt();

    // Clamp to reasonable range [0.5x, 3x base weight]
    let adaptive_weight = base_weight * adaptive_factor.clamp(0.5, 3.0);

    // Also consider iteration count: more iterations = more stable regrets
    // Scale up slightly for well-solved trees
    let iteration_factor = (small_iterations as f32 / 100.0).min(1.5).max(0.5);

    let final_weight = adaptive_weight * iteration_factor;

    eprintln!(
        "  Adaptive weight: base={:.1}, convergence_ratio={:.3}, iter_factor={:.2} → final={:.2}",
        base_weight, convergence_ratio, iteration_factor, final_weight
    );

    final_weight
}

/// Context for transfer operations
struct TransferContext<'a> {
    small_game: &'a PostFlopGame,
    large_game: &'a PostFlopGame,
    avg_factor: f32,
    interpolation_mode: InterpolationMode,
}

// Make TransferContext Sync for parallel access
unsafe impl<'a> Sync for TransferContext<'a> {}

/// Validates that small and large games are compatible for warm-start
fn validate_games(small: &PostFlopGame, large: &PostFlopGame) -> Result<(), String> {
    if small.is_memory_allocated().is_none() {
        return Err("Source game must have memory allocated".into());
    }
    if large.is_memory_allocated().is_none() {
        return Err("Target game must have memory allocated".into());
    }

    if small.card_config() != large.card_config() {
        return Err("Card configurations must match (same ranges, flop, turn)".into());
    }

    if small.tree_config().starting_pot != large.tree_config().starting_pot {
        return Err("Starting pot must match".into());
    }
    if small.tree_config().effective_stack != large.tree_config().effective_stack {
        return Err("Effective stack must match".into());
    }

    if small.regret_bits() != large.regret_bits() {
        eprintln!(
            "Warning: Different regret precision ({} vs {} bits). Will convert during transfer.",
            small.regret_bits(),
            large.regret_bits()
        );
    }

    Ok(())
}

/// Recursively transfers regrets from small tree to large tree (sequential version)
fn transfer_regrets_recursive(
    small_node: &PostFlopNode,
    large_node: &mut PostFlopNode,
    ctx: &TransferContext,
) -> Result<(), String> {
    // Base case: terminal or chance nodes
    if small_node.is_terminal() || small_node.is_chance() {
        for child_idx in 0..small_node.num_actions() {
            let small_child = small_node.play(child_idx);
            let small_action = get_action_at_index(small_node, child_idx)?;
            let large_child_idx = find_matching_child(large_node, &small_action)?;

            if let Some(idx) = large_child_idx {
                let mut large_child = large_node.play(idx);
                transfer_regrets_recursive(&*small_child, &mut *large_child, ctx)?;
            } else {
                return Err(format!(
                    "Matching child not found for chance/terminal action {:?}",
                    small_action
                ));
            }
        }
        return Ok(());
    }

    // Player node: transfer regrets
    transfer_node_regrets(small_node, large_node, ctx)?;

    // Recurse to children
    for small_idx in 0..small_node.num_actions() {
        let small_action = get_action_at_index(small_node, small_idx)?;
        let matching_idx = find_matching_child(large_node, &small_action)?;

        if let Some(large_idx) = matching_idx {
            let small_child = small_node.play(small_idx);
            let mut large_child = large_node.play(large_idx);
            transfer_regrets_recursive(&*small_child, &mut *large_child, ctx)?;
        }
    }

    Ok(())
}

/// Parallel version of regret transfer (requires rayon feature)
#[cfg(feature = "rayon")]
fn transfer_regrets_parallel(
    small_node: &PostFlopNode,
    large_node: &mut PostFlopNode,
    ctx: &TransferContext,
) -> Result<(), String> {
    // Base case: terminal or chance nodes - use sequential for simplicity
    if small_node.is_terminal() || small_node.is_chance() {
        return transfer_regrets_recursive(small_node, large_node, ctx);
    }

    // Player node: transfer regrets (this part is sequential per node)
    transfer_node_regrets(small_node, large_node, ctx)?;

    // Collect child pairs for parallel processing
    let mut child_pairs: Vec<(usize, usize)> = Vec::new();

    for small_idx in 0..small_node.num_actions() {
        let small_action = get_action_at_index(small_node, small_idx)?;
        if let Some(large_idx) = find_matching_child(large_node, &small_action)? {
            child_pairs.push((small_idx, large_idx));
        }
    }

    // Parallel recursion on children
    // Note: We use par_iter on the indices and re-acquire locks inside
    // This is safe because each child is independent
    let results: Vec<Result<(), String>> = child_pairs
        .par_iter()
        .map(|&(small_idx, large_idx)| {
            let small_child = small_node.play(small_idx);
            let mut large_child = large_node.play(large_idx);
            transfer_regrets_parallel(&*small_child, &mut *large_child, ctx)
        })
        .collect();

    // Check for errors
    for result in results {
        result?;
    }

    Ok(())
}

/// Transfer regrets for a single node
fn transfer_node_regrets(
    small_node: &PostFlopNode,
    large_node: &mut PostFlopNode,
    ctx: &TransferContext,
) -> Result<(), String> {
    let small_regrets = extract_regrets(small_node, ctx.small_game)?;
    let num_hands = small_regrets.len() / small_node.num_actions();

    // Initialize accumulator for large tree
    let mut large_regrets = vec![0.0f32; large_node.num_actions() * num_hands];

    // Calculate pot size for bet percentage calculations
    let pot_size = calculate_pot_size(small_node, ctx.small_game);

    // Get large tree actions once
    let large_actions = get_all_actions(large_node)?;

    // Map each small action to large action(s)
    for small_idx in 0..small_node.num_actions() {
        let small_action = get_action_at_index(small_node, small_idx)?;
        let small_action_regrets =
            &small_regrets[small_idx * num_hands..(small_idx + 1) * num_hands];

        // Map the action with configured interpolation mode
        let action_match = map_action(&small_action, &large_actions, pot_size, ctx.interpolation_mode);

        // Interpolate regrets into large tree
        interpolate_regrets(
            small_action_regrets,
            action_match,
            &mut large_regrets,
            num_hands,
            ctx.avg_factor,
        );
    }

    // Inject regrets into large tree
    inject_regrets(large_node, &large_regrets, ctx.large_game)?;

    Ok(())
}

/// Maps a small tree action to large tree action(s) with interpolation
fn map_action(
    small_action: &Action,
    large_actions: &[Action],
    pot_size: i32,
    interpolation_mode: InterpolationMode,
) -> ActionMatch {
    match small_action {
        // Fixed actions: direct 1:1 match
        Action::Fold | Action::Call | Action::Check => {
            for (idx, large_action) in large_actions.iter().enumerate() {
                if std::mem::discriminant(small_action) == std::mem::discriminant(large_action) {
                    return ActionMatch::Direct(idx);
                }
            }
            panic!("Fixed action {:?} not found in large tree", small_action);
        }

        // Parametric actions: interpolation based on bet size
        Action::Bet(small_bet) | Action::Raise(small_bet) | Action::AllIn(small_bet) => {
            let small_pct = *small_bet as f32 / pot_size as f32;

            // Extract all bet/raise/allin actions from large tree with their indices
            let mut large_bets: Vec<(usize, i32)> = large_actions
                .iter()
                .enumerate()
                .filter_map(|(idx, action)| match action {
                    Action::Bet(amt) | Action::Raise(amt) | Action::AllIn(amt) => Some((idx, *amt)),
                    _ => None,
                })
                .collect();

            // Sort by amount
            large_bets.sort_by_key(|(_, amt)| *amt);

            // Case 1: Exact match
            for &(idx, amt) in &large_bets {
                if amt == *small_bet {
                    return ActionMatch::Direct(idx);
                }
            }

            // Case 2: Bracketing interpolation
            for i in 0..large_bets.len().saturating_sub(1) {
                let (low_idx, low_amt) = large_bets[i];
                let (high_idx, high_amt) = large_bets[i + 1];

                if low_amt < *small_bet && *small_bet < high_amt {
                    let low_pct = low_amt as f32 / pot_size as f32;
                    let high_pct = high_amt as f32 / pot_size as f32;

                    // Calculate weight based on interpolation mode
                    let weight = match interpolation_mode {
                        InterpolationMode::Linear => {
                            // Linear interpolation
                            (small_pct - low_pct) / (high_pct - low_pct)
                        }
                        InterpolationMode::Logarithmic => {
                            // Logarithmic interpolation (better for strategic distance)
                            // Use log(1 + pct) to handle pct=0 case gracefully
                            let log_small = (1.0 + small_pct).ln();
                            let log_low = (1.0 + low_pct).ln();
                            let log_high = (1.0 + high_pct).ln();

                            if (log_high - log_low).abs() < 1e-6 {
                                0.5 // Avoid division by zero
                            } else {
                                (log_small - log_low) / (log_high - log_low)
                            }
                        }
                    };

                    return ActionMatch::Interpolated {
                        low: low_idx,
                        high: high_idx,
                        weight: weight.clamp(0.0, 1.0),
                    };
                }
            }

            // Case 3: Extrapolation below first bet (interpolate with 0% baseline: Check or Call)
            if !large_bets.is_empty() && *small_bet < large_bets[0].1 {
                let zero_baseline_idx = large_actions
                    .iter()
                    .position(|a| matches!(a, Action::Check | Action::Call))
                    .expect("No Check or Call action found for 0% baseline extrapolation");

                let first_bet_pct = large_bets[0].1 as f32 / pot_size as f32;

                // For extrapolation below, also use configured interpolation mode
                let weight = match interpolation_mode {
                    InterpolationMode::Linear => {
                        small_pct / first_bet_pct
                    }
                    InterpolationMode::Logarithmic => {
                        // log(1 + 0) = 0, so we just use log(1 + small) / log(1 + first)
                        let log_small = (1.0 + small_pct).ln();
                        let log_first = (1.0 + first_bet_pct).ln();
                        if log_first.abs() < 1e-6 {
                            0.5
                        } else {
                            log_small / log_first
                        }
                    }
                };

                return ActionMatch::Interpolated {
                    low: zero_baseline_idx,
                    high: large_bets[0].0,
                    weight: weight.clamp(0.0, 1.0),
                };
            }

            // Case 4: Extrapolation above last bet
            // Try to interpolate toward all-in if available, otherwise nearest neighbor
            if !large_bets.is_empty() {
                let last_bet = large_bets.last().unwrap();

                // Check if there's an all-in action we can interpolate toward
                if let Some(allin_idx) = large_actions.iter().position(|a| matches!(a, Action::AllIn(_))) {
                    if let Action::AllIn(allin_amt) = large_actions[allin_idx] {
                        let last_pct = last_bet.1 as f32 / pot_size as f32;
                        let allin_pct = allin_amt as f32 / pot_size as f32;

                        if allin_pct > last_pct && small_pct > last_pct {
                            // Interpolate between last bet and all-in
                            let weight = match interpolation_mode {
                                InterpolationMode::Linear => {
                                    ((small_pct - last_pct) / (allin_pct - last_pct)).min(1.0)
                                }
                                InterpolationMode::Logarithmic => {
                                    let log_small = (1.0 + small_pct).ln();
                                    let log_last = (1.0 + last_pct).ln();
                                    let log_allin = (1.0 + allin_pct).ln();
                                    if (log_allin - log_last).abs() < 1e-6 {
                                        1.0
                                    } else {
                                        ((log_small - log_last) / (log_allin - log_last)).min(1.0)
                                    }
                                }
                            };

                            return ActionMatch::Interpolated {
                                low: last_bet.0,
                                high: allin_idx,
                                weight: weight.clamp(0.0, 1.0),
                            };
                        }
                    }
                }

                // Fallback to nearest neighbor
                return ActionMatch::Nearest(last_bet.0);
            }

            panic!("No bet actions found in large tree for mapping");
        }

        _ => panic!("Unexpected action type: {:?}", small_action),
    }
}

/// Interpolates regrets from small action to large action(s)
fn interpolate_regrets(
    small_regrets: &[f32],
    action_match: ActionMatch,
    large_regrets: &mut [f32],
    num_hands: usize,
    avg_factor: f32,
) {
    match action_match {
        ActionMatch::Direct(idx) => {
            let target = &mut large_regrets[idx * num_hands..(idx + 1) * num_hands];
            for (t, &s) in target.iter_mut().zip(small_regrets) {
                *t = s * avg_factor;
            }
        }

        ActionMatch::Interpolated { low, high, weight } => {
            for (i, &r) in small_regrets.iter().enumerate() {
                let scaled = r * avg_factor;
                large_regrets[low * num_hands + i] += scaled * (1.0 - weight);
                large_regrets[high * num_hands + i] += scaled * weight;
            }
        }

        ActionMatch::Nearest(idx) => {
            let target = &mut large_regrets[idx * num_hands..(idx + 1) * num_hands];
            for (t, &s) in target.iter_mut().zip(small_regrets) {
                *t = s * avg_factor;
            }
        }
    }
}

/// Extracts regrets from a node, handling all quantization modes
fn extract_regrets(node: &PostFlopNode, game: &PostFlopGame) -> Result<Vec<f32>, String> {
    let regret_bits = game.regret_bits();

    match regret_bits {
        32 => Ok(node.regrets().to_vec()),
        16 => {
            let scale = node.regret_scale();
            let compressed = node.regrets_compressed();
            let decoder = scale / i16::MAX as f32;
            Ok(compressed.iter().map(|&r| r as f32 * decoder).collect())
        }
        8 => {
            use crate::CfrAlgorithm;
            let scale = node.regret_scale();

            match game.cfr_algorithm() {
                CfrAlgorithm::DCFRPlus | CfrAlgorithm::SAPCFRPlus | CfrAlgorithm::PDCFRPlus => {
                    let compressed = node.regrets_u8();
                    let decoder = scale / u8::MAX as f32;
                    Ok(compressed.iter().map(|&r| r as f32 * decoder).collect())
                }
                _ => {
                    let compressed = node.regrets_i8();
                    let decoder = scale / i8::MAX as f32;
                    Ok(compressed.iter().map(|&r| r as f32 * decoder).collect())
                }
            }
        }
        4 => {
            let scale = node.regret_scale();
            let compressed = node.regrets_u8();
            let storage_size = compressed.len();
            let num_elements = storage_size * 2;
            let decoder = scale / 7.0;
            let mut regrets = Vec::with_capacity(num_elements);

            for i in 0..num_elements {
                let byte = compressed[i / 2];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let val = ((nibble << 4) as i8) >> 4;
                regrets.push(val as f32 * decoder);
            }

            Ok(regrets)
        }
        _ => Err(format!("Unsupported regret_bits: {}", regret_bits)),
    }
}

/// Injects regrets into a node, handling all quantization modes
fn inject_regrets(
    node: &mut PostFlopNode,
    regrets: &[f32],
    game: &PostFlopGame,
) -> Result<(), String> {
    use crate::utility::*;

    let regret_bits = game.regret_bits();

    match regret_bits {
        32 => {
            node.regrets_mut().copy_from_slice(regrets);
            Ok(())
        }
        16 => {
            let dst = node.regrets_compressed_mut();
            let scale = encode_signed_slice(dst, regrets);
            node.set_regret_scale(scale);
            Ok(())
        }
        8 => {
            use crate::CfrAlgorithm;
            let seed = node as *const _ as u32;

            match game.cfr_algorithm() {
                CfrAlgorithm::DCFRPlus | CfrAlgorithm::SAPCFRPlus | CfrAlgorithm::PDCFRPlus => {
                    let dst = node.regrets_u8_mut();
                    let scale = encode_unsigned_regrets_u8(dst, regrets, seed);
                    node.set_regret_scale(scale);
                    Ok(())
                }
                _ => {
                    let dst = node.regrets_i8_mut();
                    let scale = encode_signed_i8(dst, regrets, seed);
                    node.set_regret_scale(scale);
                    Ok(())
                }
            }
        }
        4 => {
            let seed = node as *const _ as u32;
            let dst = node.regrets_u8_mut();
            let scale = encode_signed_i4_packed(dst, regrets, seed);
            node.set_regret_scale(scale);
            Ok(())
        }
        _ => Err(format!("Unsupported regret_bits: {}", regret_bits)),
    }
}

/// Calculates pot size at a given node
fn calculate_pot_size(node: &PostFlopNode, game: &PostFlopGame) -> i32 {
    let starting_pot = game.tree_config().starting_pot;
    starting_pot + node.amount * 2
}

/// Finds matching child index in large tree for a given small tree action
fn find_matching_child(
    large_node: &PostFlopNode,
    small_action: &Action,
) -> Result<Option<usize>, String> {
    if large_node.is_chance() {
        if let Action::Chance(card) = small_action {
            for idx in 0..large_node.num_actions() {
                let large_action = get_action_at_index(large_node, idx)?;
                if let Action::Chance(large_card) = large_action {
                    if large_card == *card {
                        return Ok(Some(idx));
                    }
                }
            }
        }
        return Ok(None);
    }

    for idx in 0..large_node.num_actions() {
        let large_action = get_action_at_index(large_node, idx)?;
        if actions_equivalent(small_action, &large_action) {
            return Ok(Some(idx));
        }
    }

    Ok(None)
}

/// Checks if two actions are equivalent
fn actions_equivalent(a1: &Action, a2: &Action) -> bool {
    match (a1, a2) {
        (Action::Fold, Action::Fold) => true,
        (Action::Check, Action::Check) => true,
        (Action::Call, Action::Call) => true,
        (Action::Bet(amt1), Action::Bet(amt2)) => amt1 == amt2,
        (Action::Raise(amt1), Action::Raise(amt2)) => amt1 == amt2,
        (Action::AllIn(amt1), Action::AllIn(amt2)) => amt1 == amt2,
        (Action::Chance(c1), Action::Chance(c2)) => c1 == c2,
        _ => false,
    }
}

/// Helper: Gets action at a specific index from a node
fn get_action_at_index(node: &PostFlopNode, index: usize) -> Result<Action, String> {
    if index >= node.num_actions() {
        return Err(format!(
            "Action index {} out of bounds (max: {})",
            index,
            node.num_actions()
        ));
    }

    let child = node.play(index);
    Ok(child.prev_action)
}

/// Helper: Gets all actions from a node
fn get_all_actions(node: &PostFlopNode) -> Result<Vec<Action>, String> {
    let mut actions = Vec::with_capacity(node.num_actions());

    for idx in 0..node.num_actions() {
        let child = node.play(idx);
        actions.push(child.prev_action);
    }

    Ok(actions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_modes() {
        // Test logarithmic vs linear interpolation weights

        // Example: small bet is 50% pot, large tree has 25% and 100%
        let _pot_size: i32 = 100;
        let small_pct: f32 = 0.5; // 50%
        let low_pct: f32 = 0.25;  // 25%
        let high_pct: f32 = 1.0;  // 100%

        // Linear weight
        let linear_weight = (small_pct - low_pct) / (high_pct - low_pct);
        // = (0.5 - 0.25) / (1.0 - 0.25) = 0.25 / 0.75 = 0.333

        // Logarithmic weight
        let log_small = (1.0_f32 + small_pct).ln();
        let log_low = (1.0_f32 + low_pct).ln();
        let log_high = (1.0_f32 + high_pct).ln();
        let log_weight = (log_small - log_low) / (log_high - log_low);
        // log(1.5) = 0.405, log(1.25) = 0.223, log(2.0) = 0.693
        // = (0.405 - 0.223) / (0.693 - 0.223) = 0.182 / 0.470 = 0.387

        println!("Linear weight: {:.4}", linear_weight);
        println!("Logarithmic weight: {:.4}", log_weight);

        // Logarithmic should give slightly higher weight to the higher bet
        // because in log space, 50% is "closer" to 100% than in linear space
        assert!(log_weight > linear_weight);
    }

    #[test]
    fn test_config_default() {
        let config = WarmStartConfig::default();
        assert_eq!(config.interpolation_mode, InterpolationMode::Logarithmic);
        assert!(config.parallel);
        assert!(config.adaptive_weight);
    }
}
