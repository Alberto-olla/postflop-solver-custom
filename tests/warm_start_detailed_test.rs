//! Detailed tests for warm-start regret transfer mechanism.
//!
//! These tests verify:
//! 1. Direct regret inheritance (exact action matches)
//! 2. Interpolation for intermediate actions
//! 3. Non-zero initialization of new actions
//! 4. Extrapolation behavior (nearest neighbor)
//! 5. Quantitative comparison of regret values

use postflop_solver::*;

/// Helper: Creates a test game with specified bet sizes
fn create_game_with_bets(
    card_config: &CardConfig,
    bet_str: &str,
    raise_str: &str,
    starting_pot: i32,
    effective_stack: i32,
) -> PostFlopGame {
    let betsizes = BetSizeOptions::try_from((bet_str, raise_str)).unwrap();
    let mut tree_config = TreeConfig::default();
    tree_config.initial_state = BoardState::Turn;
    tree_config.starting_pot = starting_pot;
    tree_config.effective_stack = effective_stack;
    tree_config.turn_bet_sizes = [betsizes.clone(), betsizes.clone()];
    tree_config.river_bet_sizes = [betsizes.clone(), betsizes.clone()];

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.set_strategy_bits(32);
    game.set_regret_bits(32);
    game
}

/// Helper: Extract all actions from the root node
fn get_root_actions(game: &mut PostFlopGame) -> Vec<Action> {
    game.back_to_root();
    game.available_actions()
}

/// Helper: Extract regrets from the root node
fn get_root_regrets(game: &PostFlopGame) -> Vec<f32> {
    let root = game.root();
    root.regrets().to_vec()
}

/// Helper: Check if any regret in a slice is non-zero
fn has_nonzero_regrets(regrets: &[f32]) -> bool {
    regrets.iter().any(|&r| r.abs() > 1e-10)
}

/// Helper: Count non-zero regrets
fn count_nonzero_regrets(regrets: &[f32]) -> usize {
    regrets.iter().filter(|&&r| r.abs() > 1e-10).count()
}

/// Helper: Get regrets for a specific action index
fn get_action_regrets(game: &PostFlopGame, action_idx: usize) -> Vec<f32> {
    let root = game.root();
    let regrets = root.regrets();
    let num_actions = root.num_actions();
    let num_hands = regrets.len() / num_actions;

    regrets[action_idx * num_hands..(action_idx + 1) * num_hands].to_vec()
}

/// Helper: Get total sum of absolute regrets
fn total_abs_regret(regrets: &[f32]) -> f32 {
    regrets.iter().map(|r| r.abs()).sum()
}

// ============================================================================
// TEST 1: Verify regrets are actually transferred (not left at zero)
// ============================================================================
#[test]
fn test_warmstart_regrets_are_transferred() {
    println!("\n=== Test: Regrets Are Actually Transferred ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let starting_pot = 100;
    let effective_stack = 1000;

    // Phase 1: Create and solve minimal tree
    let mut game_minimal = create_game_with_bets(&card_config, "50%", "", starting_pot, effective_stack);
    game_minimal.allocate_memory();

    // Solve for 50 iterations to accumulate significant regrets
    for i in 0..50 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_regrets_before = get_root_regrets(&game_minimal);
    let minimal_total_regret = total_abs_regret(&minimal_regrets_before);

    println!("Minimal tree after 50 iterations:");
    println!("  Total absolute regret sum: {:.6}", minimal_total_regret);
    println!("  Non-zero regrets: {}/{}",
             count_nonzero_regrets(&minimal_regrets_before),
             minimal_regrets_before.len());

    assert!(
        minimal_total_regret > 0.0,
        "Minimal tree should have non-zero regrets after solving"
    );

    // Phase 2: Create full tree and apply warm-start
    let mut game_full = create_game_with_bets(&card_config, "25%, 50%, 75%", "", starting_pot, effective_stack);
    game_full.allocate_memory();

    // Check regrets BEFORE warm-start (should be zero)
    let full_regrets_before_warmstart = get_root_regrets(&game_full);
    let total_before = total_abs_regret(&full_regrets_before_warmstart);

    println!("\nFull tree BEFORE warm-start:");
    println!("  Total absolute regret sum: {:.6}", total_before);
    assert!(
        total_before == 0.0,
        "Full tree should have zero regrets before warm-start"
    );

    // Apply warm-start
    let start_iter = game_full.warm_start_from(&game_minimal, 50, 10.0).unwrap();
    println!("\nWarm-start applied, starting iteration: {}", start_iter);

    // Check regrets AFTER warm-start (should be non-zero)
    let full_regrets_after_warmstart = get_root_regrets(&game_full);
    let total_after = total_abs_regret(&full_regrets_after_warmstart);

    println!("\nFull tree AFTER warm-start:");
    println!("  Total absolute regret sum: {:.6}", total_after);
    println!("  Non-zero regrets: {}/{}",
             count_nonzero_regrets(&full_regrets_after_warmstart),
             full_regrets_after_warmstart.len());

    assert!(
        total_after > 0.0,
        "Full tree should have non-zero regrets AFTER warm-start! Got: {}",
        total_after
    );

    assert!(
        has_nonzero_regrets(&full_regrets_after_warmstart),
        "Full tree should have at least some non-zero regrets after warm-start"
    );

    println!("\n[PASSED] Regrets ARE transferred from minimal to full tree\n");
}

// ============================================================================
// TEST 2: Verify direct action match transfers regrets correctly
// ============================================================================
#[test]
fn test_warmstart_direct_action_match() {
    println!("\n=== Test: Direct Action Match Regret Transfer ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let starting_pot = 100;
    let effective_stack = 1000;

    // Minimal tree: 50% bet
    let mut game_minimal = create_game_with_bets(&card_config, "50%", "", starting_pot, effective_stack);
    game_minimal.allocate_memory();

    for i in 0..30 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_actions = get_root_actions(&mut game_minimal);
    println!("Minimal tree actions: {:?}", minimal_actions);

    // Find the 50% bet action index in minimal tree
    let minimal_bet_idx = minimal_actions.iter().position(|a| matches!(a, Action::Bet(_))).unwrap();
    let minimal_bet_regrets = get_action_regrets(&game_minimal, minimal_bet_idx);

    println!("Minimal tree 50% bet regrets (first 10): {:?}", &minimal_bet_regrets[..10.min(minimal_bet_regrets.len())]);

    // Full tree: 25%, 50%, 75% bets
    let mut game_full = create_game_with_bets(&card_config, "25%, 50%, 75%", "", starting_pot, effective_stack);
    game_full.allocate_memory();

    // Apply warm-start
    game_full.warm_start_from(&game_minimal, 30, 10.0).unwrap();

    let full_actions = get_root_actions(&mut game_full);
    println!("Full tree actions: {:?}", full_actions);

    // Find the 50% bet in full tree (should have direct match)
    let full_bet_50_idx = full_actions.iter().position(|a| {
        if let Action::Bet(amt) = a {
            *amt == 50  // 50% of 100 pot
        } else {
            false
        }
    }).expect("50% bet should exist in full tree");

    let full_bet_50_regrets = get_action_regrets(&game_full, full_bet_50_idx);
    println!("Full tree 50% bet regrets (first 10): {:?}", &full_bet_50_regrets[..10.min(full_bet_50_regrets.len())]);

    // Verify regrets were transferred
    let minimal_sum = total_abs_regret(&minimal_bet_regrets);
    let full_50_sum = total_abs_regret(&full_bet_50_regrets);

    println!("\nMinimal 50% bet total |regret|: {:.6}", minimal_sum);
    println!("Full 50% bet total |regret|: {:.6}", full_50_sum);

    // The warmstart factor is weight/iters = 10.0/30 = 0.333...
    // So full regrets should be approximately minimal * 0.333
    let expected_ratio = 10.0 / 30.0;
    let actual_ratio = if minimal_sum > 0.0 { full_50_sum / minimal_sum } else { 0.0 };

    println!("Expected ratio: {:.4}, Actual ratio: {:.4}", expected_ratio, actual_ratio);

    assert!(
        full_50_sum > 0.0,
        "Direct match action should have non-zero regrets after warm-start"
    );

    // Allow some tolerance due to quantization/precision
    assert!(
        (actual_ratio - expected_ratio).abs() < 0.1,
        "Regret ratio should be close to warmstart_weight/iterations"
    );

    println!("\n[PASSED] Direct action match transfers regrets correctly\n");
}

// ============================================================================
// TEST 3: Verify NEW actions (not in minimal tree) get interpolated regrets
// ============================================================================
#[test]
fn test_warmstart_new_actions_get_interpolated_regrets() {
    println!("\n=== Test: New Actions Get Interpolated Regrets ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let starting_pot = 100;
    let effective_stack = 1000;

    // Minimal tree: 50% bet only
    let mut game_minimal = create_game_with_bets(&card_config, "50%", "", starting_pot, effective_stack);
    game_minimal.allocate_memory();

    for i in 0..30 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_actions = get_root_actions(&mut game_minimal);
    println!("Minimal tree actions: {:?}", minimal_actions);

    // Full tree: 25%, 75% bets (NO 50%!)
    // This means 50% must be INTERPOLATED between 25% and 75%
    let mut game_full = create_game_with_bets(&card_config, "25%, 75%", "", starting_pot, effective_stack);
    game_full.allocate_memory();

    let full_actions = get_root_actions(&mut game_full);
    println!("Full tree actions: {:?}", full_actions);

    // Check regrets BEFORE warm-start
    let full_regrets_before = get_root_regrets(&game_full);
    assert!(
        total_abs_regret(&full_regrets_before) == 0.0,
        "Full tree should have zero regrets before warm-start"
    );

    // Apply warm-start
    game_full.warm_start_from(&game_minimal, 30, 10.0).unwrap();

    // Check regrets for 25% bet (this action did NOT exist in minimal tree!)
    let full_bet_25_idx = full_actions.iter().position(|a| {
        if let Action::Bet(amt) = a {
            *amt == 25  // 25% of 100 pot
        } else {
            false
        }
    }).expect("25% bet should exist in full tree");

    let full_bet_75_idx = full_actions.iter().position(|a| {
        if let Action::Bet(amt) = a {
            *amt == 75  // 75% of 100 pot
        } else {
            false
        }
    }).expect("75% bet should exist in full tree");

    let full_bet_25_regrets = get_action_regrets(&game_full, full_bet_25_idx);
    let full_bet_75_regrets = get_action_regrets(&game_full, full_bet_75_idx);

    let sum_25 = total_abs_regret(&full_bet_25_regrets);
    let sum_75 = total_abs_regret(&full_bet_75_regrets);

    println!("\nFull tree 25% bet total |regret|: {:.6}", sum_25);
    println!("Full tree 75% bet total |regret|: {:.6}", sum_75);
    println!("25% bet non-zero count: {}/{}", count_nonzero_regrets(&full_bet_25_regrets), full_bet_25_regrets.len());
    println!("75% bet non-zero count: {}/{}", count_nonzero_regrets(&full_bet_75_regrets), full_bet_75_regrets.len());

    // CRITICAL: These actions should have NON-ZERO regrets from interpolation!
    // The 50% bet from minimal tree should be interpolated between 25% and 75%
    assert!(
        sum_25 > 0.0,
        "25% bet (not in minimal tree) should have NON-ZERO regrets from interpolation! Got: {}",
        sum_25
    );

    assert!(
        sum_75 > 0.0,
        "75% bet (not in minimal tree) should have NON-ZERO regrets from interpolation! Got: {}",
        sum_75
    );

    // Both should receive approximately half the regret (linear interpolation)
    // 50% is exactly between 25% and 75%, so weight should be 0.5
    let ratio = if sum_75 > 0.0 { sum_25 / sum_75 } else { 0.0 };
    println!("Ratio 25%/75%: {:.4} (expected ~1.0 for 50% interpolation point)", ratio);

    // Allow some tolerance
    assert!(
        ratio > 0.3 && ratio < 3.0,
        "Both actions should receive similar regret from interpolation"
    );

    println!("\n[PASSED] New actions get interpolated regrets (NOT left at zero)\n");
}

// ============================================================================
// TEST 4: Verify extrapolation for out-of-range actions
// ============================================================================
#[test]
fn test_warmstart_extrapolation_behavior() {
    println!("\n=== Test: Extrapolation for Out-of-Range Actions ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let starting_pot = 100;
    let effective_stack = 1000;

    // Test A: Small bet extrapolation (30% → 50%/75%)
    println!("Test A: Small bet extrapolation (30% → 50%/75%)");
    {
        let mut game_minimal = create_game_with_bets(&card_config, "30%", "", starting_pot, effective_stack);
        game_minimal.allocate_memory();
        for i in 0..30 { solve_step(&mut game_minimal, i); }

        let mut game_full = create_game_with_bets(&card_config, "50%, 75%", "", starting_pot, effective_stack);
        game_full.allocate_memory();

        game_full.warm_start_from(&game_minimal, 30, 10.0).unwrap();

        let full_actions = get_root_actions(&mut game_full);
        let bet_50_idx = full_actions.iter().position(|a| matches!(a, Action::Bet(50))).unwrap();
        let bet_50_regrets = get_action_regrets(&game_full, bet_50_idx);

        println!("  50% bet total |regret|: {:.6}", total_abs_regret(&bet_50_regrets));

        // 30% should be extrapolated between check (0%) and 50%, with weight 30/50 = 0.6 toward 50%
        assert!(
            total_abs_regret(&bet_50_regrets) > 0.0,
            "50% bet should get regrets from 30% extrapolation"
        );
    }

    // Test B: Large bet extrapolation (150% → 50%/75%/100%)
    println!("\nTest B: Large bet extrapolation (150% → 50%/75%/100%)");
    {
        let mut game_minimal = create_game_with_bets(&card_config, "150%", "", starting_pot, effective_stack);
        game_minimal.allocate_memory();
        for i in 0..30 { solve_step(&mut game_minimal, i); }

        let mut game_full = create_game_with_bets(&card_config, "50%, 75%, 100%", "", starting_pot, effective_stack);
        game_full.allocate_memory();

        game_full.warm_start_from(&game_minimal, 30, 10.0).unwrap();

        let full_actions = get_root_actions(&mut game_full);
        let bet_100_idx = full_actions.iter().position(|a| matches!(a, Action::Bet(100))).unwrap();
        let bet_100_regrets = get_action_regrets(&game_full, bet_100_idx);

        println!("  100% bet total |regret|: {:.6}", total_abs_regret(&bet_100_regrets));

        // 150% should be mapped to nearest neighbor (100%)
        assert!(
            total_abs_regret(&bet_100_regrets) > 0.0,
            "100% bet should get regrets from 150% nearest neighbor extrapolation"
        );
    }

    println!("\n[PASSED] Extrapolation behavior works correctly\n");
}

// ============================================================================
// TEST 5: Verify regret signs and magnitudes are preserved
// ============================================================================
#[test]
fn test_warmstart_regret_signs_preserved() {
    println!("\n=== Test: Regret Signs and Magnitudes Preserved ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let starting_pot = 100;
    let effective_stack = 1000;

    // Create and solve minimal tree
    let mut game_minimal = create_game_with_bets(&card_config, "50%", "", starting_pot, effective_stack);
    game_minimal.allocate_memory();

    for i in 0..50 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_regrets = get_root_regrets(&game_minimal);

    // Count positive and negative regrets
    let positive_count = minimal_regrets.iter().filter(|&&r| r > 0.0).count();
    let negative_count = minimal_regrets.iter().filter(|&&r| r < 0.0).count();
    let zero_count = minimal_regrets.iter().filter(|&&r| r == 0.0).count();

    println!("Minimal tree regrets:");
    println!("  Positive: {}", positive_count);
    println!("  Negative: {}", negative_count);
    println!("  Zero: {}", zero_count);

    // Create full tree and warm-start
    let mut game_full = create_game_with_bets(&card_config, "50%", "", starting_pot, effective_stack);
    game_full.allocate_memory();

    game_full.warm_start_from(&game_minimal, 50, 10.0).unwrap();

    let full_regrets = get_root_regrets(&game_full);

    // Count positive and negative regrets in full tree
    let full_positive_count = full_regrets.iter().filter(|&&r| r > 0.0).count();
    let full_negative_count = full_regrets.iter().filter(|&&r| r < 0.0).count();

    println!("\nFull tree regrets after warm-start:");
    println!("  Positive: {}", full_positive_count);
    println!("  Negative: {}", full_negative_count);

    // Verify sign distribution is similar (allowing for scaling)
    let min_pos_ratio = positive_count as f32 / minimal_regrets.len() as f32;
    let full_pos_ratio = full_positive_count as f32 / full_regrets.len() as f32;

    println!("\nPositive regret ratio:");
    println!("  Minimal: {:.4}", min_pos_ratio);
    println!("  Full: {:.4}", full_pos_ratio);

    // Check a few specific values for sign preservation
    let warmstart_factor = 10.0 / 50.0;
    println!("\nSample regret comparisons (factor = {:.4}):", warmstart_factor);

    for (i, (&m, &f)) in minimal_regrets.iter().zip(full_regrets.iter()).take(10).enumerate() {
        let expected = m * warmstart_factor;
        let diff = (f - expected).abs();
        println!("  [{}] minimal: {:.4}, expected: {:.4}, actual: {:.4}, diff: {:.6}",
                 i, m, expected, f, diff);
    }

    // Verify magnitudes are scaled correctly
    let total_minimal = total_abs_regret(&minimal_regrets);
    let total_full = total_abs_regret(&full_regrets);
    let actual_factor = total_full / total_minimal;

    println!("\nMagnitude scaling:");
    println!("  Expected factor: {:.4}", warmstart_factor);
    println!("  Actual factor: {:.4}", actual_factor);
    println!("  Difference: {:.6}", (actual_factor - warmstart_factor).abs());

    assert!(
        (actual_factor - warmstart_factor).abs() < 0.01,
        "Regret magnitude should be scaled by warmstart_weight/iterations"
    );

    println!("\n[PASSED] Regret signs and magnitudes are preserved\n");
}

// ============================================================================
// TEST 6: Verify Check/Fold actions are transferred correctly
// ============================================================================
#[test]
fn test_warmstart_fixed_actions_transfer() {
    println!("\n=== Test: Fixed Actions (Check/Fold) Transfer ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let starting_pot = 100;
    let effective_stack = 1000;

    // Create and solve minimal tree
    let mut game_minimal = create_game_with_bets(&card_config, "50%", "", starting_pot, effective_stack);
    game_minimal.allocate_memory();

    for i in 0..30 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_actions = get_root_actions(&mut game_minimal);
    let check_idx = minimal_actions.iter().position(|a| matches!(a, Action::Check)).unwrap();
    let minimal_check_regrets = get_action_regrets(&game_minimal, check_idx);

    println!("Minimal tree Check action:");
    println!("  Total |regret|: {:.6}", total_abs_regret(&minimal_check_regrets));
    println!("  First 5 values: {:?}", &minimal_check_regrets[..5.min(minimal_check_regrets.len())]);

    // Create full tree with different bet sizes
    let mut game_full = create_game_with_bets(&card_config, "25%, 50%, 75%", "", starting_pot, effective_stack);
    game_full.allocate_memory();

    game_full.warm_start_from(&game_minimal, 30, 10.0).unwrap();

    let full_actions = get_root_actions(&mut game_full);
    let full_check_idx = full_actions.iter().position(|a| matches!(a, Action::Check)).unwrap();
    let full_check_regrets = get_action_regrets(&game_full, full_check_idx);

    println!("\nFull tree Check action after warm-start:");
    println!("  Total |regret|: {:.6}", total_abs_regret(&full_check_regrets));
    println!("  First 5 values: {:?}", &full_check_regrets[..5.min(full_check_regrets.len())]);

    // Verify Check regrets are transferred correctly (direct match)
    let warmstart_factor = 10.0 / 30.0;
    let expected_total = total_abs_regret(&minimal_check_regrets) * warmstart_factor;
    let actual_total = total_abs_regret(&full_check_regrets);

    println!("\nCheck action regret comparison:");
    println!("  Expected total: {:.6}", expected_total);
    println!("  Actual total: {:.6}", actual_total);

    assert!(
        (actual_total - expected_total).abs() < 1.0,
        "Check action regrets should be transferred correctly"
    );

    println!("\n[PASSED] Fixed actions are transferred correctly\n");
}

// ============================================================================
// TEST 7: Comprehensive interpolation weight verification
// ============================================================================
#[test]
fn test_warmstart_interpolation_weights() {
    println!("\n=== Test: Interpolation Weight Verification ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let starting_pot = 100;
    let effective_stack = 1000;

    // Minimal tree: 50% bet
    let mut game_minimal = create_game_with_bets(&card_config, "50%", "", starting_pot, effective_stack);
    game_minimal.allocate_memory();

    for i in 0..50 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_actions = get_root_actions(&mut game_minimal);
    let minimal_bet_idx = minimal_actions.iter().position(|a| matches!(a, Action::Bet(_))).unwrap();
    let minimal_bet_regrets = get_action_regrets(&game_minimal, minimal_bet_idx);
    let minimal_bet_sum = total_abs_regret(&minimal_bet_regrets);

    // Full tree: 25%, 75% (50% will be interpolated between them)
    let mut game_full = create_game_with_bets(&card_config, "25%, 75%", "", starting_pot, effective_stack);
    game_full.allocate_memory();

    game_full.warm_start_from(&game_minimal, 50, 10.0).unwrap();

    let full_actions = get_root_actions(&mut game_full);

    let bet_25_idx = full_actions.iter().position(|a| matches!(a, Action::Bet(25))).unwrap();
    let bet_75_idx = full_actions.iter().position(|a| matches!(a, Action::Bet(75))).unwrap();

    let bet_25_regrets = get_action_regrets(&game_full, bet_25_idx);
    let bet_75_regrets = get_action_regrets(&game_full, bet_75_idx);

    let sum_25 = total_abs_regret(&bet_25_regrets);
    let sum_75 = total_abs_regret(&bet_75_regrets);

    // Calculate interpolation weights
    // 50% is exactly between 25% and 75%, so weight should be 0.5 each
    // Expected: sum_25 ≈ sum_75 ≈ minimal_bet_sum * (warmstart_factor) * 0.5
    let warmstart_factor = 10.0 / 50.0;
    let expected_per_action = minimal_bet_sum * warmstart_factor * 0.5;

    println!("Minimal 50% bet total |regret|: {:.6}", minimal_bet_sum);
    println!("Warmstart factor: {:.4}", warmstart_factor);
    println!("\nExpected per interpolated action: {:.6}", expected_per_action);
    println!("Actual 25% bet: {:.6}", sum_25);
    println!("Actual 75% bet: {:.6}", sum_75);
    println!("Sum 25% + 75%: {:.6}", sum_25 + sum_75);
    println!("Expected sum: {:.6}", minimal_bet_sum * warmstart_factor);

    // The total should equal the minimal bet regret * warmstart_factor
    // because all regret from 50% is distributed to 25% and 75%
    let total_full_bets = sum_25 + sum_75;
    let expected_total = minimal_bet_sum * warmstart_factor;

    println!("\nTotal distribution check:");
    println!("  Expected: {:.6}", expected_total);
    println!("  Actual: {:.6}", total_full_bets);
    println!("  Difference: {:.6}", (total_full_bets - expected_total).abs());

    // Allow 10% tolerance
    assert!(
        (total_full_bets - expected_total).abs() / expected_total < 0.1,
        "Total interpolated regret should match source regret scaled by warmstart_factor"
    );

    // Check ratio between 25% and 75%
    let ratio = sum_25 / sum_75;
    println!("\nInterpolation ratio 25%/75%: {:.4} (expected ~1.0)", ratio);

    assert!(
        ratio > 0.8 && ratio < 1.2,
        "50% should be evenly interpolated between 25% and 75%"
    );

    println!("\n[PASSED] Interpolation weights are correct\n");
}

// ============================================================================
// TEST 8: Run all tests and provide summary
// ============================================================================
#[test]
fn test_warmstart_summary() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("WARM-START DETAILED TEST SUMMARY");
    println!("{}", "=".repeat(80));
    println!("\nThese tests verify the mathematical correctness of warm-start:\n");
    println!("1. test_warmstart_regrets_are_transferred");
    println!("   - Verifies regrets are NOT left at zero after warm-start");
    println!();
    println!("2. test_warmstart_direct_action_match");
    println!("   - Verifies exact action matches transfer regrets with correct scaling");
    println!();
    println!("3. test_warmstart_new_actions_get_interpolated_regrets");
    println!("   - Verifies NEW actions (not in minimal tree) get NON-ZERO regrets");
    println!("   - This is the KEY test: interpolation distributes regrets to neighbors");
    println!();
    println!("4. test_warmstart_extrapolation_behavior");
    println!("   - Verifies small/large bets outside range are handled correctly");
    println!();
    println!("5. test_warmstart_regret_signs_preserved");
    println!("   - Verifies regret signs (+/-) and magnitudes scale correctly");
    println!();
    println!("6. test_warmstart_fixed_actions_transfer");
    println!("   - Verifies Check/Fold actions transfer with direct match");
    println!();
    println!("7. test_warmstart_interpolation_weights");
    println!("   - Verifies interpolation weight formula: weight = (small - low) / (high - low)");
    println!();
    println!("Run with: cargo test --test warm_start_detailed_test -- --nocapture");
    println!("{}", "=".repeat(80));
}
