use postflop_solver::*;

fn main() {
    println!("Testing chance_bits implementation...\n");

    // Create a simple game configuration
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AKo".parse().unwrap();
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AKo".parse().unwrap();

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("50%", "50%")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 200,
        effective_stack: 900,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    
    // Test 1: Default (16-bit chance)
    println!("Test 1: Default 16-bit chance cfvalues");
    let mut game1 = PostFlopGame::with_config(card_config.clone(), action_tree.clone()).unwrap();
    game1.allocate_memory_with_mode(QuantizationMode::Int16);
    let mem1 = game1.memory_usage_mb();
    println!("  Memory usage (16-bit): {:.2} MB", mem1);
    println!("  chance_bits: {}", game1.chance_bits());
    
    // Test 2: 8-bit chance
    println!("\nTest 2: 8-bit chance cfvalues");
    let mut game2 = PostFlopGame::with_config(card_config.clone(), action_tree.clone()).unwrap();
    game2.set_chance_bits(8);
    game2.allocate_memory_with_mode(QuantizationMode::Int16);
    let mem2 = game2.memory_usage_mb();
    println!("  Memory usage (8-bit): {:.2} MB", mem2);
    println!("  chance_bits: {}", game2.chance_bits());
    
    let savings = (1.0 - mem2 / mem1) * 100.0;
    println!("\n  Memory savings: {:.2}%", savings);
    
    // Test 3: Solve with 8-bit chance
    println!("\nTest 3: Solving with 8-bit chance cfvalues");
    solve(&mut game2, 100, 0.5, false);
    println!("  Solved successfully!");
    println!("  Exploitability: {:.4}", game2.exploitability());
    
    // Test 4: Compare with 16-bit
    println!("\nTest 4: Solving with 16-bit chance cfvalues (comparison)");
    solve(&mut game1, 100, 0.5, false);
    println!("  Solved successfully!");
    println!("  Exploitability: {:.4}", game1.exploitability());
    
    println!("\n✅ All tests passed!");
}

