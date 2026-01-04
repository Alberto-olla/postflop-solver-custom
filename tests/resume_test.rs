
use postflop_solver::*;

#[test]
fn test_resume_with_checkpoint() {
    // Setup configuration: a game that converges in ~40 iterations
    let card_config = CardConfig {
        range: [Range::ones(); 2],
        flop: flop_from_str("Ks 7d 2c").unwrap(),
        turn: card_from_str("As").unwrap(),
        ..Default::default()
    };

    let bet_sizes = BetSizeOptions::try_from(("50%", "")).unwrap();

    let mut tree_config = TreeConfig::default();
    tree_config.initial_state = BoardState::Turn;
    tree_config.starting_pot = 100;
    tree_config.effective_stack = 1000;
    tree_config.flop_bet_sizes = [bet_sizes.clone(), bet_sizes.clone()];
    tree_config.turn_bet_sizes = [bet_sizes.clone(), bet_sizes.clone()];
    tree_config.river_bet_sizes = [bet_sizes.clone(), bet_sizes.clone()];

    // BASELINE: Run 40 iterations continuously
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_baseline = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game_baseline.allocate_memory();

    for i in 0..40 {
        solve_step(&mut game_baseline, i);
    }
    let exploitability_baseline = compute_exploitability(&game_baseline);

    // RESUME TEST: Run 30 iterations, save checkpoint, load, run remaining 10 iterations
    let action_tree_resume = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_resume = PostFlopGame::with_config(card_config.clone(), action_tree_resume).unwrap();
    game_resume.allocate_memory();

    // Part 1: First 30 iterations
    for i in 0..30 {
        solve_step(&mut game_resume, i);
    }

    // Save checkpoint at iteration 30
    let checkpoint_path = "test_checkpoint.bin";
    save_checkpoint(&game_resume, 30, checkpoint_path).expect("Failed to save checkpoint");

    // Load checkpoint
    let checkpoint = load_checkpoint(checkpoint_path).expect("Failed to load checkpoint");
    let mut game_loaded = checkpoint.game;
    let loaded_iteration = checkpoint.current_iteration;

    // Verify loaded iteration count
    assert_eq!(loaded_iteration, 30, "Loaded iteration count should be 30");

    // Verify game state is valid
    assert!(game_loaded.is_ready(), "Loaded game should be ready");

    // Part 2: Resume from iteration 30 to 40 (10 more iterations)
    for i in loaded_iteration..(loaded_iteration + 10) {
        solve_step(&mut game_loaded, i);
    }

    let exploitability_resumed = compute_exploitability(&game_loaded);

    // Cleanup
    let _ = std::fs::remove_file(checkpoint_path);

    println!("Baseline (40 continuous):  {:.6}", exploitability_baseline);
    println!("Resumed (30+10):           {:.6}", exploitability_resumed);

    // The resumed run should produce the same result as the baseline
    // Allow small tolerance for floating-point differences
    let diff = (exploitability_baseline - exploitability_resumed).abs();
    let relative_error = diff / exploitability_baseline;

    println!("Absolute difference:       {:.6}", diff);
    println!("Relative error:            {:.6}%", relative_error * 100.0);

    // Assert that the difference is negligible (< 0.1% relative error)
    assert!(
        relative_error < 0.001,
        "Resumed run differs too much from baseline! Relative error: {:.6}%",
        relative_error * 100.0
    );
}
