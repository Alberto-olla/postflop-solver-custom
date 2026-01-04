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
    game_minimal.set_strategy_bits(32);
    game_minimal.set_regret_bits(32);
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
    game_full_warm.set_strategy_bits(32);
    game_full_warm.set_regret_bits(32);
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
    let game2 = PostFlopGame::with_config(card_config.clone(), action_tree2).unwrap();
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
fn test_solve_from_config_reproduction() {
    println!("\n=== Reproduction of solve_from_config (Hand 7438) ===\n");

    // 1. Configuration (hardcoded from hand_0000007438_node_03_turn_DeepStack.toml)
    // ----------------------
    let oop_range_str = "AhQh";
    let ip_range_str = "AJs:0.00,QTs:0.00,QTo:0.01,Q9o:0.00,Q7o:0.00,Q6s:0.00,Q6o:0.01,Q5o:0.01,Q5s:0.00,Q3o:0.00,Q2s:0.00,JTo:0.00,J9o:0.00,J9s:0.00,J8s:0.00,TT:0.00,T9o:0.00,T9s:0.00,T8s:0.00,T8o:0.00,T7o:0.02,T7s:0.00,T6o:0.01,T6s:0.00,T5o:0.00,T4o:0.00,T4s:0.00,T3s:0.00,98o:0.00,98s:0.00,97o:0.00,97s:0.00,96o:0.00,95o:0.00,95s:0.00,94o:0.00,93o:0.00,93s:0.01,93o:0.00,92o:0.00,88:0.00,87o:0.00,87s:0.00,86o:0.00,86s:0.00,85o:0.00,84o:0.00,83o:0.00,83s:0.00,82o:0.00,77:0.00,76o:0.00,75o:0.00,74s:0.00,74o:0.00,73o:0.00,73s:0.00,72o:0.00,72s:0.00,66:0.00,65s:0.00,64o:0.00,63o:0.00,62o:0.00,62s:0.00,55:0.00,54s:0.00,54o:0.00,53o:0.00,52s:0.00,52o:0.00,43o:0.00,42s:0.00,42o:0.00,33:0.00,32s:0.00,32o:0.01,22:0.01";

    let card_config = CardConfig {
        range: [
            Range::from_str(oop_range_str).unwrap(),
            Range::from_str(ip_range_str).unwrap(),
        ],
        flop: flop_from_str("7h 6d 6h").unwrap(),
        turn: card_from_str("5s").unwrap(),
        river: NOT_DEALT,
    };

    // Tree Parameters
    let starting_pot = 3900i32;
    let effective_stack = 17600i32;

    // Use `parse_bet_sizes_with_preset` to exactly match regression tests and config loading
    // Turn: OOP and IP all actions "full"
    let turn_bet_sizes = parse_bet_sizes_with_preset(
        "full", "full", // OOP bet, raise
        "full", "full", // IP bet, raise
        starting_pot,
        effective_stack,
        "turn"
    ).expect("Failed to parse turn bet sizes");

    // River: OOP and IP all actions "full"
    let river_bet_sizes = parse_bet_sizes_with_preset(
        "full", "full",
        "full", "full",
        starting_pot,
        effective_stack,
        "river"
    ).expect("Failed to parse river bet sizes");

    let mut tree_config = TreeConfig::default();
    tree_config.initial_state = BoardState::Turn;
    tree_config.starting_pot = starting_pot;
    tree_config.effective_stack = effective_stack;
    
    // Config params from TOML
    tree_config.add_allin_threshold = 1.5;
    tree_config.force_allin_threshold = 0.15;
    tree_config.merging_threshold = 0.1;

    tree_config.turn_bet_sizes = turn_bet_sizes;
    tree_config.river_bet_sizes = river_bet_sizes;

    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    
    // Solver Settings from TOML
    game.set_strategy_bits(32);
    game.set_regret_bits(32);
    game.set_ip_bits(32);
    game.set_chance_bits(32);
    game.set_cfr_algorithm(CfrAlgorithm::DCFRPlus); // DCFR+
    
    game.allocate_memory();

    // Solve loop
    let max_iterations = 200;
    let target_exploitability = starting_pot as f32 * 0.005; // 0.5% of pot

    println!("Solving (max {} iterations, target expl: {:.4})...", max_iterations, target_exploitability);

    let start_time = std::time::Instant::now();
    let (iters, expl) = solve_and_monitor(
        &mut game,
        0,
        max_iterations,
        target_exploitability,
        "Reproduction"
    );
    let duration = start_time.elapsed();

    println!("\n✓ Reproduction completed in {:.2}s", duration.as_secs_f64());
    println!("  Final exploitability: {:.6}", expl);
    println!("  Iterations: {}", iters);
    
    // Assert expectations (should be fast, similar to "4s" report)
    assert!(duration.as_secs_f64() < 10.0, "Execution time should be small (matches example), not large like 'Minimal Tree' test");
}


#[test]
#[ignore] // Run with: cargo test --test warm_start_test test_warm_start_acceleration -- --ignored --nocapture
fn test_warm_start_acceleration() {
    println!("\n=== CS-CFR Warm-Start Acceleration Test ===\n");

    // 1. Shared Configuration
    // ----------------------
    let oop_range_str = "AhQh";
    let ip_range_str = "AJs:0.00,QTs:0.00,QTo:0.01,Q9o:0.00,Q7o:0.00,Q6s:0.00,Q6o:0.01,Q5o:0.01,Q5s:0.00,Q3o:0.00,Q2s:0.00,JTo:0.00,J9o:0.00,J9s:0.00,J8s:0.00,TT:0.00,T9o:0.00,T9s:0.00,T8s:0.00,T8o:0.00,T7o:0.02,T7s:0.00,T6o:0.01,T6s:0.00,T5o:0.00,T4o:0.00,T4s:0.00,T3s:0.00,98o:0.00,98s:0.00,97o:0.00,97s:0.00,96o:0.00,95o:0.00,95s:0.00,94s:0.00,94o:0.00,93s:0.01,93o:0.00,92o:0.00,88:0.00,87o:0.00,87s:0.00,86o:0.00,86s:0.00,85o:0.00,84o:0.00,83o:0.00,83s:0.00,82o:0.00,77:0.00,76o:0.00,75o:0.00,74s:0.00,74o:0.00,73o:0.00,73s:0.00,72o:0.00,72s:0.00,66:0.00,65s:0.00,64o:0.00,63o:0.00,62o:0.00,62s:0.00,55:0.00,54s:0.00,54o:0.00,53o:0.00,52s:0.00,52o:0.00,43o:0.00,42s:0.00,42o:0.00,33:0.00,32s:0.00,32o:0.01,22:0.01";

    let card_config = CardConfig {
        range: [
            Range::from_str(oop_range_str).unwrap(),
            Range::from_str(ip_range_str).unwrap(),
        ],
        flop: flop_from_str("7h 6d 6h").unwrap(),
        turn: card_from_str("5s").unwrap(),
        river: NOT_DEALT,
    };

    // Shared Tree Parameters
    let starting_pot = 3900i32;
    let effective_stack = 17600i32;

    // Phase 1 Params (Minimal Tree)
    let minimal_bet_str = ("50%", "2x");

    // Use `parse_bet_sizes_with_preset` to match node 03 expansion
    let full_turn_bet_sizes = parse_bet_sizes_with_preset(
        "full", "full", "full", "full",
        starting_pot, effective_stack, "turn"
    ).expect("Failed to parse full turn bet sizes");

    let full_river_bet_sizes = parse_bet_sizes_with_preset(
        "full", "full", "full", "full",
        starting_pot, effective_stack, "river"
    ).expect("Failed to parse full river bet sizes");

    // ALIGN TARGET EXPLOITABILITY WITH TOML (0.5% of pot)
    let target_exploitability = starting_pot as f32 * 0.005; 


    // 2. Configuration Diagnostics
    // ----------------------------
    println!("\n--- Configuration Diagnostics ---");
    let minimal_bet = BetSizeOptions::try_from(minimal_bet_str).unwrap();
    let mut tree_config_minimal = create_common_tree_config(starting_pot, effective_stack);
    tree_config_minimal.turn_bet_sizes = [minimal_bet.clone(), minimal_bet.clone()];
    tree_config_minimal.river_bet_sizes = [minimal_bet.clone(), minimal_bet.clone()];

    let action_tree_minimal = ActionTree::new(tree_config_minimal).unwrap();
    let mut game_minimal = PostFlopGame::with_config(card_config.clone(), action_tree_minimal).unwrap();
    game_minimal.set_strategy_bits(32);
    game_minimal.set_regret_bits(32);
    game_minimal.set_ip_bits(32); // Match TOML
    game_minimal.set_chance_bits(32); // Match TOML
    game_minimal.set_cfr_algorithm(CfrAlgorithm::DCFRPlus);
    game_minimal.allocate_memory();

    let mut tree_config_full = create_common_tree_config(starting_pot, effective_stack);
    tree_config_full.turn_bet_sizes = full_turn_bet_sizes;
    tree_config_full.river_bet_sizes = full_river_bet_sizes;

    let action_tree_full_cold = ActionTree::new(tree_config_full.clone()).unwrap(); 
    let mut game_full_cold = PostFlopGame::with_config(card_config.clone(), action_tree_full_cold).unwrap();
    game_full_cold.set_strategy_bits(32);
    game_full_cold.set_regret_bits(32);
    game_full_cold.set_ip_bits(32); // Match TOML
    game_full_cold.set_chance_bits(32); // Match TOML
    game_full_cold.set_cfr_algorithm(CfrAlgorithm::DCFRPlus);
    game_full_cold.allocate_memory();

    print_and_compare_configs(&game_minimal, &game_full_cold, "Minimal (Phase 1)", "Cold (Phase 3)");


    // 3. Phase 1: Minimal Tree Solve
    // ------------------------------
    println!("\nPhase 1: Solving minimal tree (50% bet, 2x raise)...");

    // Set minimal target to 1% of pot for quick abstraction base
    let minimal_target = starting_pot as f32 * 0.01; 

    let start_time_minimal = std::time::Instant::now();
    let (minimal_iters, minimal_expl) = solve_and_monitor(
        &mut game_minimal,
        0,
        5000,
        minimal_target,
        "Minimal tree"
    );
    let duration_minimal = start_time_minimal.elapsed();

    println!("  Minimal tree finished after {} iters: exploitability = {:.6} (Time: {:.2?}s)", minimal_iters, minimal_expl, duration_minimal.as_secs_f32());


    // 4. Phase 2: Warm-Start Full Tree
    // --------------------------------
    println!("\nPhase 2: Warm-starting same tree but more exploitability...");

    let action_tree_full_warm = ActionTree::new(tree_config_full.clone()).unwrap();
    let mut game_full_warm = PostFlopGame::with_config(card_config.clone(), action_tree_full_warm).unwrap();
    game_full_warm.set_strategy_bits(32);
    game_full_warm.set_regret_bits(32);
    game_full_warm.set_ip_bits(32); // Match TOML
    game_full_warm.set_chance_bits(32); // Match TOML
    game_full_warm.set_cfr_algorithm(CfrAlgorithm::DCFRPlus); // Use DCFR
    game_full_warm.allocate_memory();

    // Apply warm-start
    let start_iter = game_full_warm.warm_start_from(&game_minimal, minimal_iters, 10.0).unwrap();
    println!("  Applied warm-start: starting at iteration {}", start_iter);

    println!("  Solving warm-started tree to {:.1}% exploitability...", target_exploitability * 100.0);
    
    let start_time_warm = std::time::Instant::now();
    let (warm_solve_iters, warm_final_expl) = solve_and_monitor(
        &mut game_full_warm,
        start_iter,
        start_iter + 2000, 
        target_exploitability,
        "Warm-start"
    );
    let duration_warm = start_time_warm.elapsed();
    
    let total_warm_iters = warm_solve_iters; // solve_and_monitor returns the iteration it finished at
    let warm_steps = total_warm_iters - start_iter;


    // 5. Phase 3: Cold-Start Full Tree
    // --------------------------------
    println!("\nPhase 3: Baseline cold-start (no warm-start)...");

    // Uses game_full_cold created in diagnostics step
    println!("  Solving cold-started tree to {:.1}% exploitability...", target_exploitability / starting_pot as f32 * 100.0);

    let start_time_cold = std::time::Instant::now();
    let (cold_iters, cold_final_expl) = solve_and_monitor(
        &mut game_full_cold,
        0,
        2000,
        target_exploitability,
        "Cold-start"
    );
    let duration_cold = start_time_cold.elapsed();


    // 6. Results
    // ----------
    println!("\n=== RESULTS ===");
    println!("Warm-start:");
    println!("  Total iterations: {} (start={}, solve={})", total_warm_iters, start_iter, warm_steps);
    println!("  Final exploitability: {:.6}", warm_final_expl);
    println!("  Phase 1 (Minimal) time: {:.2?}s", duration_minimal.as_secs_f32());
    println!("  Phase 2 (Warm) time:    {:.2?}s", duration_warm.as_secs_f32());
    let total_warm_time = duration_minimal + duration_warm;
    println!("  Total Warm-start time:  {:.2?}s", total_warm_time.as_secs_f32());
    
    println!("\nCold-start:");
    println!("  Total iterations: {}", cold_iters);
    println!("  Final exploitability: {:.6}", cold_final_expl);
    println!("  Total Cold-start time:  {:.2?}s", duration_cold.as_secs_f32()); 

    let speedup_iter = cold_iters as f32 / total_warm_iters as f32;
    println!("\nSpeedup (Iterations): {:.2}x", speedup_iter);
    
    let speedup_time = duration_cold.as_secs_f32() / total_warm_time.as_secs_f32();
    println!("Speedup (Time):       {:.2}x", speedup_time);

    if total_warm_iters < cold_iters {
        let saved_iters = cold_iters - total_warm_iters;
        let reduction = (saved_iters as f32 / cold_iters as f32) * 100.0;
        println!("Iterations saved: {} ({:.1}% reduction)", saved_iters, reduction);
    }
    
    println!("\n✓ CS-CFR Warm-Start Acceleration Test COMPLETED\n");
}

// --- Helper Functions ---

fn print_and_compare_configs(g1: &PostFlopGame, g2: &PostFlopGame, label1: &str, label2: &str) {
    println!("Comparing configurations: {} vs {}", label1, label2);
    
    let check = |name: &str, v1: String, v2: String| {
        let match_str = if v1 == v2 { "MATCH" } else { "MISMATCH" };
        println!("  {:<25} | {:<20} | {:<20} | {}", name, v1, v2, match_str);
        if v1 != v2 {
            println!("    WARNING: Configuration mismatch detected for {}!", name);
        }
    };

    println!("  {:<25} | {:<20} | {:<20} | Status", "Parameter", label1, label2);
    println!("  {:-<80}", "");

    check("CFR Algorithm", format!("{:?}", g1.cfr_algorithm()), format!("{:?}", g2.cfr_algorithm()));
    check("Strategy Bits", format!("{}", g1.strategy_bits()), format!("{}", g2.strategy_bits()));
    check("Regret Bits", format!("{}", g1.regret_bits()), format!("{}", g2.regret_bits()));
    check("IP Bits", format!("{}", g1.ip_bits()), format!("{}", g2.ip_bits()));
    check("Chance Bits", format!("{}", g1.chance_bits()), format!("{}", g2.chance_bits()));
    check("Pruning Enabled", format!("{}", g1.enable_pruning()), format!("{}", g2.enable_pruning()));
    check("Lazy Norm Enabled", format!("{}", g1.is_lazy_normalization_enabled()), format!("{}", g2.is_lazy_normalization_enabled()));
    check("Compression Enabled", format!("{}", g1.is_compression_enabled()), format!("{}", g2.is_compression_enabled()));
    check("Quantization Mode", format!("{:?}", g1.quantization_mode()), format!("{:?}", g2.quantization_mode()));
    
    println!("  {:-<80}", "");
}

// --- Helper Functions ---

fn create_common_tree_config(starting_pot: i32, effective_stack: i32) -> TreeConfig {
    let mut config = TreeConfig::default();
    config.initial_state = BoardState::Turn;
    config.starting_pot = starting_pot;
    config.effective_stack = effective_stack;
    
    // Shared constants
    config.add_allin_threshold = 1.5;
    config.force_allin_threshold = 0.15;
    config.merging_threshold = 0.1;
    
    config
}

fn solve_and_monitor(
    game: &mut PostFlopGame, 
    start_iter: u32, 
    max_iter_limit: u32, 
    target_expl: f32,
    label: &str
) -> (u32, f32) {
    let mut current_iter = start_iter;

    loop {
        // Standard CFR step
        solve_step(game, current_iter);

        // Check exploitability periodically (e.g. every 20 iters)
        let rel_iter = current_iter - start_iter;
        
        if rel_iter > 0 && rel_iter % 20 == 0 {
            let expl = compute_exploitability(game);
            println!("    Iteration {}: exploitability = {:.6}", current_iter, expl);

            if expl <= target_expl {
                println!("  {} reached target at iteration {}", label, current_iter);
                return (current_iter, expl);
            }
        }

        current_iter += 1;

        if current_iter >= max_iter_limit {
             // Calculate final if we hit limit
            let expl = compute_exploitability(game);
            println!("  {} stopped at max iters ({})", label, max_iter_limit);
            return (current_iter, expl);
        }
    }
}
