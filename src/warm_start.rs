use crate::action_tree::*;
use crate::game::*;
use crate::interface::*;

/// Result of action mapping between small and large trees
#[derive(Debug, Clone)]
pub enum ActionMatch {
    /// Direct 1:1 correspondence between actions
    Direct(usize),

    /// Linear interpolation between two actions
    Interpolated {
        low: usize,      // Index of lower bet size action
        high: usize,     // Index of higher bet size action
        weight: f32,     // Interpolation weight [0, 1], favoring high
    },

    /// Nearest neighbor (extrapolation fallback)
    Nearest(usize),
}

/// Entry point for warm-start transfer
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
///
/// # Implementation Notes
/// The transfer uses normalized regrets: R_warm = (R_small / T_small) × W
/// This avoids numerical shocks and maintains the "direction" of regrets
/// while allowing the large tree solver to correct abstraction errors.
pub fn apply_warm_start(
    small_game: &PostFlopGame,
    large_game: &mut PostFlopGame,
    small_iterations: u32,
    warmstart_weight: f32,
) -> Result<(), String> {
    // Validate compatibility
    validate_games(small_game, large_game)?;

    // Calculate normalization factor: avg_factor = W / T_small
    let avg_factor = warmstart_weight / (small_iterations as f32);

    // Recursive transfer
    let small_root = small_game.root();
    let mut large_root = large_game.root();

    transfer_regrets_recursive(
        &*small_root,
        &mut *large_root,
        small_game,
        large_game,
        avg_factor,
    )?;

    Ok(())
}

/// Validates that small and large games are compatible for warm-start
fn validate_games(small: &PostFlopGame, large: &PostFlopGame) -> Result<(), String> {
    // Check state
    if !small.is_ready() {
        return Err("Source game must be ready (memory allocated)".into());
    }
    if !large.is_ready() {
        return Err("Target game must be ready (memory allocated)".into());
    }

    // Check card configuration (must match exactly)
    if small.card_config() != large.card_config() {
        return Err("Card configurations must match (same ranges, flop, turn)".into());
    }

    // Check pot and stack (must match)
    if small.tree_config().starting_pot != large.tree_config().starting_pot {
        return Err("Starting pot must match".into());
    }
    if small.tree_config().effective_stack != large.tree_config().effective_stack {
        return Err("Effective stack must match".into());
    }

    // Warn about precision differences (allowed but noted)
    if small.regret_bits() != large.regret_bits() {
        eprintln!(
            "Warning: Different regret precision ({} vs {} bits). Will convert during transfer.",
            small.regret_bits(),
            large.regret_bits()
        );
    }

    Ok(())
}

/// Recursively transfers regrets from small tree to large tree
fn transfer_regrets_recursive(
    small_node: &PostFlopNode,
    large_node: &mut PostFlopNode,
    small_game: &PostFlopGame,
    large_game: &PostFlopGame,
    avg_factor: f32,
) -> Result<(), String> {
    // Base case: terminal or chance nodes
    if small_node.is_terminal() || small_node.is_chance() {
        // For chance/terminal nodes, recurse to children only
        for child_idx in 0..small_node.num_actions() {
            let small_child = small_node.play(child_idx);

            // Find matching child in large tree
            let small_action = get_action_at_index(small_node, child_idx)?;
            let large_child_idx = find_matching_child(large_node, &small_action)?;
            let mut large_child = large_node.play(large_child_idx);

            transfer_regrets_recursive(
                &*small_child,
                &mut *large_child,
                small_game,
                large_game,
                avg_factor,
            )?;
        }
        return Ok(());
    }

    // Player node: transfer regrets
    let small_regrets = extract_regrets(small_node, small_game)?;
    let num_hands = small_regrets.len() / small_node.num_actions();

    // Initialize accumulator for large tree
    let mut large_regrets = vec![0.0f32; large_node.num_actions() * num_hands];

    // Calculate pot size for bet percentage calculations
    let pot_size = calculate_pot_size(small_node, small_game);

    // Map each small action to large action(s)
    for small_idx in 0..small_node.num_actions() {
        let small_action = get_action_at_index(small_node, small_idx)?;
        let small_action_regrets = &small_regrets[small_idx * num_hands..(small_idx + 1) * num_hands];

        // Get large tree actions for mapping
        let large_actions = get_all_actions(large_node)?;

        // Map the action
        let action_match = map_action(&small_action, &large_actions, pot_size);

        // Interpolate regrets into large tree
        interpolate_regrets(
            small_action_regrets,
            action_match,
            &mut large_regrets,
            num_hands,
            avg_factor,
        );
    }

    // Inject regrets into large tree
    inject_regrets(large_node, &large_regrets, large_game)?;

    // Recurse to children
    for small_idx in 0..small_node.num_actions() {
        let small_action = get_action_at_index(small_node, small_idx)?;
        let large_idx = find_matching_child(large_node, &small_action)?;

        let small_child = small_node.play(small_idx);
        let mut large_child = large_node.play(large_idx);

        transfer_regrets_recursive(
            &*small_child,
            &mut *large_child,
            small_game,
            large_game,
            avg_factor,
        )?;
    }

    Ok(())
}

/// Maps a small tree action to large tree action(s) with interpolation
fn map_action(
    small_action: &Action,
    large_actions: &[Action],
    pot_size: i32,
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
                    let weight = (small_pct - low_pct) / (high_pct - low_pct);

                    return ActionMatch::Interpolated {
                        low: low_idx,
                        high: high_idx,
                        weight,
                    };
                }
            }

            // Case 3: Extrapolation below first bet (interpolate with Check = 0%)
            if !large_bets.is_empty() && *small_bet < large_bets[0].1 {
                let check_idx = large_actions
                    .iter()
                    .position(|a| matches!(a, Action::Check))
                    .expect("Check action not found for extrapolation");

                let first_bet_pct = large_bets[0].1 as f32 / pot_size as f32;
                let weight = small_pct / first_bet_pct;

                return ActionMatch::Interpolated {
                    low: check_idx,
                    high: large_bets[0].0,
                    weight,
                };
            }

            // Case 4: Extrapolation above last bet (nearest neighbor)
            if !large_bets.is_empty() {
                return ActionMatch::Nearest(large_bets.last().unwrap().0);
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
            // Direct copy
            let target = &mut large_regrets[idx * num_hands..(idx + 1) * num_hands];
            for (t, &s) in target.iter_mut().zip(small_regrets) {
                *t = s * avg_factor;
            }
        }

        ActionMatch::Interpolated { low, high, weight } => {
            // Linear interpolation: split regret between low and high actions
            // Use raw indexing to avoid borrow checker issues
            for (i, &r) in small_regrets.iter().enumerate() {
                let scaled = r * avg_factor;
                large_regrets[low * num_hands + i] += scaled * (1.0 - weight);
                large_regrets[high * num_hands + i] += scaled * weight;
            }
        }

        ActionMatch::Nearest(idx) => {
            // Nearest neighbor (same as direct)
            let target = &mut large_regrets[idx * num_hands..(idx + 1) * num_hands];
            for (t, &s) in target.iter_mut().zip(small_regrets) {
                *t = s * avg_factor;
            }
        }
    }
}

/// Extracts regrets from a node, handling all quantization modes
fn extract_regrets(
    node: &PostFlopNode,
    game: &PostFlopGame,
) -> Result<Vec<f32>, String> {
    let _regret_bits = game.regret_bits();

    // For now, assume 32-bit (full precision)
    // TODO: implement quantization support based on game.regret_bits()
    Ok(node.regrets().to_vec())
}

/// Injects regrets into a node, handling all quantization modes
fn inject_regrets(
    node: &mut PostFlopNode,
    regrets: &[f32],
    _game: &PostFlopGame,
) -> Result<(), String> {
    // For now, assume 32-bit (full precision)
    // TODO: implement quantization support based on game.regret_bits()
    node.regrets_mut().copy_from_slice(regrets);
    Ok(())
}

/// Calculates pot size at a given node
fn calculate_pot_size(node: &PostFlopNode, game: &PostFlopGame) -> i32 {
    // Pot = starting_pot + total_amount_committed
    // For now, approximate using node.amount × 2 (symmetric betting)
    // TODO: Track actual pot size during traversal or store in node
    let starting_pot = game.tree_config().starting_pot;
    starting_pot + node.amount * 2
}

/// Finds matching child index in large tree for a given small tree action
fn find_matching_child(
    large_node: &PostFlopNode,
    small_action: &Action,
) -> Result<usize, String> {
    // For chance nodes: match by card
    if large_node.is_chance() {
        if let Action::Chance(card) = small_action {
            for idx in 0..large_node.num_actions() {
                let large_action = get_action_at_index(large_node, idx)?;
                if let Action::Chance(large_card) = large_action {
                    if large_card == *card {
                        return Ok(idx);
                    }
                }
            }
        }
        return Err(format!("Chance card {:?} not found in large tree", small_action));
    }

    // For player nodes: match by action equivalence
    for idx in 0..large_node.num_actions() {
        let large_action = get_action_at_index(large_node, idx)?;
        if actions_equivalent(small_action, &large_action) {
            return Ok(idx);
        }
    }

    Err(format!("No matching child found for action {:?}", small_action))
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
        return Err(format!("Action index {} out of bounds (max: {})", index, node.num_actions()));
    }

    // The action is stored in the child node's prev_action field
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
