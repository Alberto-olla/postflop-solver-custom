
use postflop_solver::*;
// use postflop_solver::PostFlopGame; // Available at root
// use postflop_solver::solver::solve_step; // solve_step is in solver but solver is re-exported?
// lib.rs says: pub use solver::*;
// So solve_step is available at root.

#[test]
fn test_resume_capability() {
    // 1. Setup Common Config
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

    // 2. Baseline Run (0 to 20)
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_baseline = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game_baseline.allocate_memory();
    
    for i in 0..20 {
        solve_step(&mut game_baseline, i);
    }
    let exploitability_baseline = compute_exploitability(&game_baseline);
    
    // 3. Split Run
    // Part A: 0 to 10
    let action_tree_split = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_split = PostFlopGame::with_config(card_config.clone(), action_tree_split).unwrap();
    game_split.allocate_memory();
    
    for i in 0..10 {
        solve_step(&mut game_split, i);
    }
    
    // Save
    let save_path = "test_resume.bin";
    // Usually standard IO errors can be panicked on in tests
    save_gametree(&game_split, save_path).expect("Failed to save gametree");
    
    // Load
    let mut game_loaded = load_gametree(save_path).expect("Failed to load gametree");
    
    // Verify State Validity (optional but good)
    assert!(game_loaded.is_ready(), "Loaded game should be ready");
    
    // Part B: 10 to 20 (Resume)
    for i in 10..20 {
        solve_step(&mut game_loaded, i);
    }
    
    let exploitability_split = compute_exploitability(&game_loaded);
    
    // Cleanup
    let _ = std::fs::remove_file(save_path);
    
    println!("Baseline Exploitability: {:.6}", exploitability_baseline);
    println!("Resumed Exploitability:  {:.6}", exploitability_split);
    
    // Assert closeness
    // We expect NEAR EXACT match because we passed correct iteration index to solve_step
    // However, floating point serialization might introduce tiny errors? 
    // bincode uses standard float serialization, should be consistent.
    // Threading (parallelism) order might affect floating point sums (non-associative).
    // If rayon is enabled, order is non-deterministic -> results differ slightly.
    // We check for "close enough".
    let diff = (exploitability_baseline - exploitability_split).abs();
    assert!(diff < 1e-5, "Resumed result differed significantly! Diff: {}", diff);
}
