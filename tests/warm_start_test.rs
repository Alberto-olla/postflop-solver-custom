use postflop_solver::*;
use std::str::FromStr;

#[test]
fn test_warm_start_basic() {
    println!("\n=== Basic Warm-Start Test ===\n");

    // Card configuration: Simple turn tree
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    // PHASE 1: Solve minimal tree (50% bet only)
    println!("Phase 1: Solving minimal tree (50% bet only)...");

    let minimal_betsizes = BetSizeOptions::try_from(("50%", "")).unwrap();
    let mut tree_config_minimal = TreeConfig::default();
    tree_config_minimal.initial_state = BoardState::Turn;
    tree_config_minimal.starting_pot = 100;
    tree_config_minimal.effective_stack = 1000;
    tree_config_minimal.turn_bet_sizes = [minimal_betsizes.clone(), minimal_betsizes.clone()];
    tree_config_minimal.river_bet_sizes = [minimal_betsizes.clone(), minimal_betsizes.clone()];

    let action_tree_minimal = ActionTree::new(tree_config_minimal).unwrap();
    let mut game_minimal = PostFlopGame::with_config(card_config.clone(), action_tree_minimal).unwrap();
    game_minimal.allocate_memory();

    // Solve for 20 iterations
    for i in 0..20 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_expl = compute_exploitability(&game_minimal);
    println!("  Minimal tree after 20 iters: exploitability = {:.6}", minimal_expl);

    // PHASE 2: Warm-start full tree (25%, 50%, 75% bets)
    println!("\nPhase 2: Warm-starting full tree (25%, 50%, 75% bets)...");

    let full_betsizes = BetSizeOptions::try_from(("25%, 50%, 75%", "")).unwrap();
    let mut tree_config_full = TreeConfig::default();
    tree_config_full.initial_state = BoardState::Turn;
    tree_config_full.starting_pot = 100;
    tree_config_full.effective_stack = 1000;
    tree_config_full.turn_bet_sizes = [full_betsizes.clone(), full_betsizes.clone()];
    tree_config_full.river_bet_sizes = [full_betsizes.clone(), full_betsizes.clone()];

    let action_tree_full = ActionTree::new(tree_config_full).unwrap();
    let mut game_full_warm = PostFlopGame::with_config(card_config, action_tree_full).unwrap();
    game_full_warm.allocate_memory();

    // Apply warm-start
    let result = game_full_warm.warm_start_from(&game_minimal, 20, 10.0);

    // Verify warm-start succeeded
    assert!(result.is_ok(), "Warm-start should succeed: {:?}", result.err());

    let start_iter = result.unwrap();
    println!("  Applied warm-start: starting at iteration {}", start_iter);
    assert_eq!(start_iter, 10, "Should start at iteration 10");

    // Solve a few more iterations to verify it doesn't crash
    for i in start_iter..(start_iter + 10) {
        solve_step(&mut game_full_warm, i);
    }

    let warm_expl = compute_exploitability(&game_full_warm);
    println!("  Warm-started tree after {} total iters: exploitability = {:.6}", start_iter + 10, warm_expl);

    println!("\n✓ Basic warm-start test PASSED\n");
}

#[test]
fn test_warm_start_validation() {
    println!("\n=== Warm-Start Validation Test ===\n");

    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let betsizes = BetSizeOptions::try_from(("50%", "")).unwrap();
    let mut tree_config = TreeConfig::default();
    tree_config.initial_state = BoardState::Turn;
    tree_config.starting_pot = 100;
    tree_config.effective_stack = 1000;
    tree_config.turn_bet_sizes = [betsizes.clone(), betsizes.clone()];

    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game1 = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game1.allocate_memory();

    // Test 1: Should fail if source game not ready
    let action_tree2 = ActionTree::new(tree_config.clone()).unwrap();
    let mut game2 = PostFlopGame::with_config(card_config.clone(), action_tree2).unwrap();
    // Don't allocate memory for game2

    let mut game3 = PostFlopGame::with_config(card_config.clone(), ActionTree::new(tree_config.clone()).unwrap()).unwrap();
    game3.allocate_memory();

    let result = game3.warm_start_from(&game2, 10, 10.0);
    assert!(result.is_err(), "Should fail when source game not ready");
    println!("  ✓ Correctly rejects source game without allocated memory");

    // Test 2: Should fail if target game not ready
    let mut game4 = PostFlopGame::with_config(card_config, ActionTree::new(tree_config).unwrap()).unwrap();
    // Don't allocate memory for game4

    for i in 0..10 {
        solve_step(&mut game1, i);
    }

    let result = game4.warm_start_from(&game1, 10, 10.0);
    assert!(result.is_err(), "Should fail when target game not ready");
    println!("  ✓ Correctly rejects target game without allocated memory");

    println!("\n✓ Validation test PASSED\n");
}

#[test]
#[ignore] // Run with: cargo test --test warm_start_test test_warm_start_acceleration -- --ignored --nocapture
fn test_warm_start_acceleration() {
    println!("\n=== CS-CFR Warm-Start Acceleration Test ===\n");

    // Card configuration from hand_0000007438_node_03_turn
    let oop_range_str = "AhQh";
    let ip_range_str = "Q6o:0.01,T7o:0.02,T6o:0.01,Q5o:0.01,93s:0.01,32o:0.01,22:0.01";

    let card_config = CardConfig {
        range: [
            Range::from_str(oop_range_str).unwrap(),
            Range::from_str(ip_range_str).unwrap(),
        ],
        flop: flop_from_str("7h 6d 6h").unwrap(),
        turn: card_from_str("5s").unwrap(),
        river: NOT_DEALT,
    };

    // PHASE 1: Solve minimal tree (50% bet, 2x raise)
    println!("Phase 1: Solving minimal tree (50% bet, 2x raise)...");

    let minimal_bet = BetSizeOptions::try_from(("50%", "2x")).unwrap();
    let mut tree_config_minimal = TreeConfig::default();
    tree_config_minimal.initial_state = BoardState::Turn;
    tree_config_minimal.starting_pot = 3900;
    tree_config_minimal.effective_stack = 17600;
    tree_config_minimal.turn_bet_sizes = [minimal_bet.clone(), minimal_bet.clone()];
    tree_config_minimal.river_bet_sizes = [minimal_bet.clone(), minimal_bet.clone()];
    tree_config_minimal.add_allin_threshold = 1.5;
    tree_config_minimal.force_allin_threshold = 0.15;
    tree_config_minimal.merging_threshold = 0.1;

    let action_tree_minimal = ActionTree::new(tree_config_minimal).unwrap();
    let mut game_minimal = PostFlopGame::with_config(card_config.clone(), action_tree_minimal).unwrap();
    game_minimal.allocate_memory();

    // Solve for 40 iterations
    for i in 0..40 {
        solve_step(&mut game_minimal, i);
    }

    let minimal_expl = compute_exploitability(&game_minimal);
    println!("  Minimal tree after 40 iters: exploitability = {:.6}", minimal_expl);

    // PHASE 2: Warm-start full tree
    println!("\nPhase 2: Warm-starting full tree (using 'full' bet sizes)...");

    let full_bet = BetSizeOptions::try_from(("full", "full")).unwrap();
    let mut tree_config_full = TreeConfig::default();
    tree_config_full.initial_state = BoardState::Turn;
    tree_config_full.starting_pot = 3900;
    tree_config_full.effective_stack = 17600;
    tree_config_full.turn_bet_sizes = [full_bet.clone(), full_bet.clone()];
    tree_config_full.river_bet_sizes = [full_bet.clone(), full_bet.clone()];
    tree_config_full.add_allin_threshold = 1.5;
    tree_config_full.force_allin_threshold = 0.15;
    tree_config_full.merging_threshold = 0.1;

    let action_tree_full_warm = ActionTree::new(tree_config_full.clone()).unwrap();
    let mut game_full_warm = PostFlopGame::with_config(card_config.clone(), action_tree_full_warm).unwrap();
    game_full_warm.allocate_memory();

    // Apply warm-start
    let start_iter = game_full_warm.warm_start_from(&game_minimal, 40, 10.0).unwrap();
    println!("  Applied warm-start: starting at iteration {}", start_iter);

    // Solve warm-started tree to target exploitability
    let target_exploitability = 0.005; // 0.5%
    let mut warm_iters = 0u32;

    println!("  Solving warm-started tree to {:.1}% exploitability...", target_exploitability * 100.0);

    for i in start_iter..1000 {
        let expl = compute_exploitability(&game_full_warm);

        if (i - start_iter) % 20 == 0 {
            println!("    Iteration {}: exploitability = {:.6}", i, expl);
        }

        if expl <= target_exploitability {
            warm_iters = i - start_iter;
            println!("  Warm-start reached target at iteration {} (warm iters: {})", i, warm_iters);
            break;
        }

        solve_step(&mut game_full_warm, i);
    }

    let total_warm_iters = start_iter + warm_iters;
    let warm_final_expl = compute_exploitability(&game_full_warm);

    // PHASE 3: Baseline cold-start
    println!("\nPhase 3: Baseline cold-start (no warm-start)...");

    let action_tree_full_cold = ActionTree::new(tree_config_full).unwrap();
    let mut game_full_cold = PostFlopGame::with_config(card_config, action_tree_full_cold).unwrap();
    game_full_cold.allocate_memory();

    let mut cold_iters = 0u32;

    println!("  Solving cold-started tree to {:.1}% exploitability...", target_exploitability * 100.0);

    for i in 0..1000 {
        let expl = compute_exploitability(&game_full_cold);

        if i % 20 == 0 {
            println!("    Iteration {}: exploitability = {:.6}", i, expl);
        }

        if expl <= target_exploitability {
            cold_iters = i;
            println!("  Cold-start reached target at iteration {}", i);
            break;
        }

        solve_step(&mut game_full_cold, i);
    }

    let cold_final_expl = compute_exploitability(&game_full_cold);

    // RESULTS
    println!("\n=== RESULTS ===");
    println!("Warm-start:");
    println!("  Total iterations: {} (start={}, solve={})", total_warm_iters, start_iter, warm_iters);
    println!("  Final exploitability: {:.6}", warm_final_expl);
    println!("\nCold-start:");
    println!("  Total iterations: {}", cold_iters);
    println!("  Final exploitability: {:.6}", cold_final_expl);

    let speedup = cold_iters as f32 / total_warm_iters as f32;
    println!("\nSpeedup: {:.2}x", speedup);

    if total_warm_iters < cold_iters {
        let saved_iters = cold_iters - total_warm_iters;
        let reduction = (saved_iters as f32 / cold_iters as f32) * 100.0;
        println!("Iterations saved: {} ({:.1}% reduction)", saved_iters, reduction);
    }

    println!("\n✓ CS-CFR Warm-Start Acceleration Test COMPLETED\n");

    // Optional assertion: warm-start should be faster (at least 1.2x speedup)
    // Uncomment if you want to enforce speedup requirement
    // assert!(
    //     speedup >= 1.2,
    //     "Expected at least 1.2x speedup, got {:.2}x",
    //     speedup
    // );
}
