use postflop_solver::*;

fn main() {
    let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s".parse().unwrap();
    let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+".parse().unwrap();

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("Qc").unwrap(),
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 200,
        effective_stack: 900,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: Some(DonkSizeOptions::try_from("50%").unwrap()),
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    // Test 1: 16-bit baseline
    println!("=== TEST 1: 16-bit baseline ===");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory_with_mode(QuantizationMode::Int16);

    let (s1, s2, sip, schance) = game.memory_usage_breakdown_mb();
    println!("  storage1 (strategy): {:.2} MB", s1);
    println!("  storage2 (regrets):  {:.2} MB", s2);
    println!("  storage_ip:          {:.2} MB", sip);
    println!("  storage_chance:      {:.2} MB", schance);
    println!("  TOTAL:               {:.2} MB", game.memory_usage_mb());
    let baseline = game.memory_usage_mb();

    // Test 2: chance_bits=8
    println!("\n=== TEST 2: chance_bits=8 ===");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game2 = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game2.set_chance_bits(8);
    game2.allocate_memory_with_mode(QuantizationMode::Int16);
    let (s1, s2, sip, schance) = game2.memory_usage_breakdown_mb();
    println!("  storage1 (strategy): {:.2} MB", s1);
    println!("  storage2 (regrets):  {:.2} MB", s2);
    println!("  storage_ip:          {:.2} MB", sip);
    println!("  storage_chance:      {:.2} MB", schance);
    println!("  TOTAL:               {:.2} MB", game2.memory_usage_mb());
    println!("  Savings: {:.2} MB ({:.1}%)",
             baseline - game2.memory_usage_mb(),
             (1.0 - game2.memory_usage_mb() / baseline) * 100.0);

    // Test 3: strategy_bits=8
    println!("\n=== TEST 3: strategy_bits=8 ===");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game3 = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game3.set_strategy_bits(8);
    game3.allocate_memory_with_mode(QuantizationMode::Int16);
    let (s1, s2, sip, schance) = game3.memory_usage_breakdown_mb();
    println!("  storage1 (strategy): {:.2} MB", s1);
    println!("  storage2 (regrets):  {:.2} MB", s2);
    println!("  storage_ip:          {:.2} MB", sip);
    println!("  storage_chance:      {:.2} MB", schance);
    println!("  TOTAL:               {:.2} MB", game3.memory_usage_mb());
    println!("  Savings: {:.2} MB ({:.1}%)",
             baseline - game3.memory_usage_mb(),
             (1.0 - game3.memory_usage_mb() / baseline) * 100.0);

    // Test 4: strategy_bits=8 + chance_bits=8
    println!("\n=== TEST 4: strategy_bits=8 + chance_bits=8 ===");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game4 = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game4.set_strategy_bits(8);
    game4.set_chance_bits(8);
    game4.allocate_memory_with_mode(QuantizationMode::Int16);
    let (s1, s2, sip, schance) = game4.memory_usage_breakdown_mb();
    println!("  storage1 (strategy): {:.2} MB", s1);
    println!("  storage2 (regrets):  {:.2} MB", s2);
    println!("  storage_ip:          {:.2} MB", sip);
    println!("  storage_chance:      {:.2} MB", schance);
    println!("  TOTAL:               {:.2} MB", game4.memory_usage_mb());
    println!("  Savings: {:.2} MB ({:.1}%)",
             baseline - game4.memory_usage_mb(),
             (1.0 - game4.memory_usage_mb() / baseline) * 100.0);
}

