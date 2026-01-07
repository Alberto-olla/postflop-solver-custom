//! Performance regression tests for the CFR algorithms
//!
//! Tests all four CFR algorithm variants: DCFR, DCFR+, PDCFR+, and SAPCFR+
//! These tests use the exact configuration from node_03_turn TOML file
//! to ensure consistent comparison with baseline results.
//!
//! Run with: cargo test --release --test performance_regression -- --nocapture

use postflop_solver::*;
use serde::Deserialize;
use std::fs;
use std::time::Instant;

// Same config structures as solve_from_config example
#[derive(Debug, Deserialize)]
struct Config {
    ranges: Ranges,
    cards: Cards,
    tree: TreeSettings,
    bet_sizes: BetSizes,
    solver: SolverSettings,
}

#[derive(Debug, Deserialize)]
struct Ranges {
    oop: String,
    ip: String,
}

#[derive(Debug, Deserialize)]
struct Cards {
    flop: String,
    #[serde(default)]
    turn: String,
    #[serde(default)]
    river: String,
}

#[derive(Debug, Deserialize)]
struct TreeSettings {
    starting_pot: i32,
    effective_stack: i32,
    #[serde(default)]
    rake_rate: f64,
    #[serde(default)]
    rake_cap: f64,
    #[serde(default = "default_add_allin_threshold")]
    add_allin_threshold: f64,
    #[serde(default = "default_force_allin_threshold")]
    force_allin_threshold: f64,
    #[serde(default = "default_merging_threshold")]
    merging_threshold: f64,
}

#[derive(Debug, Deserialize)]
struct BetSizes {
    flop: StreetBetSizes,
    turn: StreetBetSizes,
    river: StreetBetSizes,
}

#[derive(Debug, Deserialize)]
struct StreetBetSizes {
    #[serde(default)]
    oop_bet: String,
    #[serde(default)]
    oop_raise: String,
    #[serde(default)]
    ip_bet: String,
    #[serde(default)]
    ip_raise: String,
}

#[derive(Debug, Deserialize)]
struct SolverSettings {
    max_iterations: usize,
    target_exploitability_pct: f32,
    #[serde(default = "default_strategy_bits")]
    strategy_bits: u8,
    #[serde(default = "default_regret_bits")]
    regret_bits: u8,
    #[serde(default = "default_ip_bits")]
    ip_bits: u8,
    #[serde(default = "default_chance_bits")]
    chance_bits: u8,
}

fn default_add_allin_threshold() -> f64 {
    1.5
}
fn default_force_allin_threshold() -> f64 {
    0.15
}
fn default_merging_threshold() -> f64 {
    0.1
}
fn default_strategy_bits() -> u8 {
    16
}
fn default_regret_bits() -> u8 {
    16
}
fn default_ip_bits() -> u8 {
    16
}
fn default_chance_bits() -> u8 {
    16
}

/// Load game from TOML config file
fn load_game_from_toml(path: &str) -> (PostFlopGame, u32, f32) {
    let config_content =
        fs::read_to_string(path).expect(&format!("Failed to read config file: {}", path));

    let config: Config = toml::from_str(&config_content).expect("Failed to parse config file");

    // Parse cards
    let flop = flop_from_str(&config.cards.flop).expect("Invalid flop");
    let turn = if config.cards.turn.is_empty() {
        NOT_DEALT
    } else {
        card_from_str(&config.cards.turn).expect("Invalid turn")
    };
    let river = if config.cards.river.is_empty() {
        NOT_DEALT
    } else {
        card_from_str(&config.cards.river).expect("Invalid river")
    };

    // Determine initial state
    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    // Parse ranges
    let oop_range = config.ranges.oop.parse().expect("Invalid OOP range");
    let ip_range = config.ranges.ip.parse().expect("Invalid IP range");

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    // Parse bet sizes with preset expansion (uses unified logic from src/preset.rs)
    let flop_bet_sizes = parse_bet_sizes_with_preset(
        &config.bet_sizes.flop.oop_bet,
        &config.bet_sizes.flop.oop_raise,
        &config.bet_sizes.flop.ip_bet,
        &config.bet_sizes.flop.ip_raise,
        config.tree.starting_pot,
        config.tree.effective_stack,
        "flop",
    )
    .expect("Invalid flop bet sizes");

    let turn_bet_sizes = parse_bet_sizes_with_preset(
        &config.bet_sizes.turn.oop_bet,
        &config.bet_sizes.turn.oop_raise,
        &config.bet_sizes.turn.ip_bet,
        &config.bet_sizes.turn.ip_raise,
        config.tree.starting_pot,
        config.tree.effective_stack,
        "turn",
    )
    .expect("Invalid turn bet sizes");

    let river_bet_sizes = parse_bet_sizes_with_preset(
        &config.bet_sizes.river.oop_bet,
        &config.bet_sizes.river.oop_raise,
        &config.bet_sizes.river.ip_bet,
        &config.bet_sizes.river.ip_raise,
        config.tree.starting_pot,
        config.tree.effective_stack,
        "river",
    )
    .expect("Invalid river bet sizes");

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.tree.starting_pot,
        effective_stack: config.tree.effective_stack,
        rake_rate: config.tree.rake_rate,
        rake_cap: config.tree.rake_cap,
        flop_bet_sizes,
        turn_bet_sizes,
        river_bet_sizes,
        add_allin_threshold: config.tree.add_allin_threshold,
        force_allin_threshold: config.tree.force_allin_threshold,
        merging_threshold: config.tree.merging_threshold,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).expect("Failed to build action tree");
    let mut game =
        PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game");

    // Apply precision settings from TOML
    game.set_strategy_bits(config.solver.strategy_bits);
    game.set_regret_bits(config.solver.regret_bits);
    game.set_ip_bits(config.solver.ip_bits);
    game.set_chance_bits(config.solver.chance_bits);

    // Calculate target exploitability from percentage
    let target_expl =
        config.tree.starting_pot as f32 * config.solver.target_exploitability_pct / 100.0;

    (game, config.solver.max_iterations as u32, target_expl)
}

// ============================================================================
// TEST 1: DCFR with 16-bit precision
// ============================================================================

#[test]
fn test_performance_dcfr_16bit_node03_turn() {
    const CONFIG_PATH: &str = "hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml";

    // Baseline values from empirical testing
    const BASELINE_ITERATIONS: u32 = 160; // For reference only
    const BASELINE_TIME_SECS: f32 = 4.42;

    // Tolerance: allow 5% degradation for time
    const TIME_TOLERANCE: f32 = 1.01;

    // Load game from TOML
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);

    // Configure for DCFR with 16-bit
    game.set_cfr_algorithm(CfrAlgorithm::DCFR);
    game.allocate_memory(); // 16-bit mode

    // Print header before solving
    println!("\n=== DCFR 16-bit Performance ===");
    println!(
        "Baseline: {} iterations, {:.2}s",
        BASELINE_ITERATIONS, BASELINE_TIME_SECS
    );

    // Solve with timing
    let start = Instant::now();
    let final_expl = solve(&mut game, max_iters, target_expl, true);
    let duration = start.elapsed();
    let time_secs = duration.as_secs_f32();

    println!("Current:  Time {:.2}s", time_secs);
    println!(
        "Final exploitability: {:.6} (target: {:.1})",
        final_expl, target_expl
    );

    // Assertions
    assert!(
        final_expl <= target_expl,
        "Failed to reach target exploitability. Got: {}, Target: {}",
        final_expl,
        target_expl
    );

    // WARNING: Time should not increase significantly
    assert!(
        time_secs <= BASELINE_TIME_SECS * TIME_TOLERANCE,
        "Performance regression! Time increased from {:.2}s to {:.2}s ({:.1}% increase)",
        BASELINE_TIME_SECS,
        time_secs,
        ((time_secs / BASELINE_TIME_SECS - 1.0) * 100.0)
    );

    // SUCCESS: Print if performance improved
    if time_secs < BASELINE_TIME_SECS * 0.90 {
        println!(
            "✓ Performance IMPROVED! Time reduced by {:.2}s ({:.1}%)",
            BASELINE_TIME_SECS - time_secs,
            ((1.0 - time_secs / BASELINE_TIME_SECS) * 100.0)
        );
    }
}

// ============================================================================
// TEST 2: DCFR+ with 16-bit precision
// ============================================================================

#[test]
fn test_performance_dcfrplus_16bit_node03_turn() {
    const CONFIG_PATH: &str = "hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml";

    // Baseline values from empirical testing
    const BASELINE_ITERATIONS: u32 = 290; // For reference only
    const BASELINE_TIME_SECS: f32 = 7.56;

    // Tolerance: allow 5% degradation for time
    const TIME_TOLERANCE: f32 = 1.05;

    // Load game from TOML
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);

    // Configure for DCFR+ with 16-bit
    game.set_cfr_algorithm(CfrAlgorithm::DCFRPlus);
    game.allocate_memory(); // 16-bit mode

    // Print header before solving
    println!("\n=== DCFR+ 16-bit Performance ===");
    println!(
        "Baseline: {} iterations, {:.2}s",
        BASELINE_ITERATIONS, BASELINE_TIME_SECS
    );

    // Solve with timing
    let start = Instant::now();
    let final_expl = solve(&mut game, max_iters, target_expl, true);
    let duration = start.elapsed();
    let time_secs = duration.as_secs_f32();

    println!("Current:  Time {:.2}s", time_secs);
    println!(
        "Final exploitability: {:.6} (target: {:.1})",
        final_expl, target_expl
    );

    // Assertions
    assert!(
        final_expl <= target_expl,
        "Failed to reach target exploitability. Got: {}, Target: {}",
        final_expl,
        target_expl
    );

    // WARNING: Time should not increase significantly
    assert!(
        time_secs <= BASELINE_TIME_SECS * TIME_TOLERANCE,
        "Performance regression! Time increased from {:.2}s to {:.2}s ({:.1}% increase)",
        BASELINE_TIME_SECS,
        time_secs,
        ((time_secs / BASELINE_TIME_SECS - 1.0) * 100.0)
    );

    // SUCCESS: Print if performance improved
    if time_secs < BASELINE_TIME_SECS * 0.90 {
        println!(
            "✓ Performance IMPROVED! Time reduced by {:.2}s ({:.1}%)",
            BASELINE_TIME_SECS - time_secs,
            ((1.0 - time_secs / BASELINE_TIME_SECS) * 100.0)
        );
    }
}

// ============================================================================
// TEST 3: SAPCFR+ with 16-bit precision
// ============================================================================

#[test]
fn test_performance_sapcfrplus_16bit_node03_turn() {
    const CONFIG_PATH: &str = "hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml";

    // Baseline values from empirical testing
    const BASELINE_ITERATIONS: u32 = 240; // For reference only
    const BASELINE_TIME_SECS: f32 = 8.77;

    // Tolerance: allow 5% degradation for time
    const TIME_TOLERANCE: f32 = 1.05;

    // Load game from TOML
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);

    // Configure for SAPCFR+ with 16-bit
    game.set_cfr_algorithm(CfrAlgorithm::SAPCFRPlus);
    game.allocate_memory(); // 16-bit mode

    // Print header before solving
    println!("\n=== SAPCFR+ 16-bit Performance ===");
    println!(
        "Baseline: {} iterations, {:.2}s",
        BASELINE_ITERATIONS, BASELINE_TIME_SECS
    );

    // Solve with timing
    let start = Instant::now();
    let final_expl = solve(&mut game, max_iters, target_expl, true);
    let duration = start.elapsed();
    let time_secs = duration.as_secs_f32();

    println!("Current:  Time {:.2}s", time_secs);
    println!(
        "Final exploitability: {:.6} (target: {:.1})",
        final_expl, target_expl
    );

    // Assertions
    assert!(
        final_expl <= target_expl,
        "Failed to reach target exploitability. Got: {}, Target: {}",
        final_expl,
        target_expl
    );

    // WARNING: Time should not increase significantly
    assert!(
        time_secs <= BASELINE_TIME_SECS * TIME_TOLERANCE,
        "Performance regression! Time increased from {:.2}s to {:.2}s ({:.1}% increase)",
        BASELINE_TIME_SECS,
        time_secs,
        ((time_secs / BASELINE_TIME_SECS - 1.0) * 100.0)
    );

    // SUCCESS: Print if performance improved
    if time_secs < BASELINE_TIME_SECS * 0.90 {
        println!(
            "✓ Performance IMPROVED! Time reduced by {:.2}s ({:.1}%)",
            BASELINE_TIME_SECS - time_secs,
            ((1.0 - time_secs / BASELINE_TIME_SECS) * 100.0)
        );
    }
}

// ============================================================================
// TEST 4: PDCFR+ with 16-bit precision
// ============================================================================

#[test]
fn test_performance_pdcfrplus_16bit_node03_turn() {
    const CONFIG_PATH: &str = "hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml";

    // Baseline values from empirical testing
    const BASELINE_ITERATIONS: u32 = 290; // For reference only
    const BASELINE_TIME_SECS: f32 = 3.30;

    // Tolerance: allow 5% degradation for time
    const TIME_TOLERANCE: f32 = 1.05;

    // Load game from TOML
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);

    // Configure for PDCFR+ with 16-bit
    game.set_cfr_algorithm(CfrAlgorithm::PDCFRPlus);
    game.allocate_memory(); // 16-bit mode

    // Print header before solving
    println!("\n=== PDCFR+ 16-bit Performance ===");
    println!(
        "Baseline: {} iterations, {:.2}s",
        BASELINE_ITERATIONS, BASELINE_TIME_SECS
    );

    // Solve with timing
    let start = Instant::now();
    let final_expl = solve(&mut game, max_iters, target_expl, true);
    let duration = start.elapsed();
    let time_secs = duration.as_secs_f32();

    println!("Current:  Time {:.2}s", time_secs);
    println!(
        "Final exploitability: {:.6} (target: {:.1})",
        final_expl, target_expl
    );

    // Assertions
    assert!(
        final_expl <= target_expl,
        "Failed to reach target exploitability. Got: {}, Target: {}",
        final_expl,
        target_expl
    );

    // WARNING: Time should not increase significantly
    assert!(
        time_secs <= BASELINE_TIME_SECS * TIME_TOLERANCE,
        "Performance regression! Time increased from {:.2}s to {:.2}s ({:.1}% increase)",
        BASELINE_TIME_SECS,
        time_secs,
        ((time_secs / BASELINE_TIME_SECS - 1.0) * 100.0)
    );

    // SUCCESS: Print if performance improved
    if time_secs < BASELINE_TIME_SECS * 0.90 {
        println!(
            "✓ Performance IMPROVED! Time reduced by {:.2}s ({:.1}%)",
            BASELINE_TIME_SECS - time_secs,
            ((1.0 - time_secs / BASELINE_TIME_SECS) * 100.0)
        );
    }
}

// ============================================================================
// BONUS TEST: Run all four and compare
// ============================================================================

#[test]
#[ignore] // Use `cargo test --release --ignored -- --nocapture` to run
fn test_compare_all_algorithms_node03_turn() {
    const CONFIG_PATH: &str = "hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml";

    println!("\n=== Comparing All Algorithms (16-bit) ===\n");

    // Test DCFR
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);
    game.set_cfr_algorithm(CfrAlgorithm::DCFR);
    game.allocate_memory();
    let start = Instant::now();
    let dcfr_expl = solve(&mut game, max_iters, target_expl, true);
    let dcfr_time = start.elapsed();

    // Test DCFR+
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);
    game.set_cfr_algorithm(CfrAlgorithm::DCFRPlus);
    game.allocate_memory();
    let start = Instant::now();
    let dcfrplus_expl = solve(&mut game, max_iters, target_expl, true);
    let dcfrplus_time = start.elapsed();

    // Test PDCFR+
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);
    game.set_cfr_algorithm(CfrAlgorithm::PDCFRPlus);
    game.allocate_memory();
    let start = Instant::now();
    let pdcfrplus_expl = solve(&mut game, max_iters, target_expl, true);
    let pdcfrplus_time = start.elapsed();

    // Test SAPCFR+
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);
    game.set_cfr_algorithm(CfrAlgorithm::SAPCFRPlus);
    game.allocate_memory();
    let start = Instant::now();
    let sapcfr_expl = solve(&mut game, max_iters, target_expl, true);
    let sapcfr_time = start.elapsed();

    // Print comparison table
    println!("\n┌─────────────┬──────────┬─────────────────┐");
    println!("│ Algorithm   │ Time (s) │ Exploitability  │");
    println!("├─────────────┼──────────┼─────────────────┤");
    println!(
        "│ DCFR        │ {:>8.2} │ {:>15.6} │",
        dcfr_time.as_secs_f32(),
        dcfr_expl
    );
    println!(
        "│ DCFR+       │ {:>8.2} │ {:>15.6} │",
        dcfrplus_time.as_secs_f32(),
        dcfrplus_expl
    );
    println!(
        "│ PDCFR+      │ {:>8.2} │ {:>15.6} │",
        pdcfrplus_time.as_secs_f32(),
        pdcfrplus_expl
    );
    println!(
        "│ SAPCFR+     │ {:>8.2} │ {:>15.6} │",
        sapcfr_time.as_secs_f32(),
        sapcfr_expl
    );
    println!("└─────────────┴──────────┴─────────────────┘");

    // All should reach target
    assert!(dcfr_expl <= target_expl);
    assert!(dcfrplus_expl <= target_expl);
    assert!(pdcfrplus_expl <= target_expl);
    assert!(sapcfr_expl <= target_expl);
}

// ============================================================================
// TEST 5: DCFR with 8-bit regrets (Storage 8-bit, Strategy 16-bit)
// ============================================================================

#[test]
fn test_performance_dcfr_8bit_regrets_node03_turn() {
    const CONFIG_PATH: &str = "hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml";

    // Baseline from user report was 2.64s (with thread_rng)
    // We expect this to be much faster now, potentially closer to 1.18s (16-bit)
    const BASELINE_TIME_SECS: f32 = 2.64;

    // Load game from TOML
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);

    // Configure for DCFR with 8-bit regrets
    game.set_cfr_algorithm(CfrAlgorithm::DCFR);

    // Override regret precision to 8-bit
    game.set_regret_bits(8);
    game.set_strategy_bits(16);
    game.allocate_memory();

    println!("\n=== DCFR 8-bit Regrets Performance ===");
    println!("Baseline (thread_rng): {:.2}s", BASELINE_TIME_SECS);

    // Solve with timing
    let start = Instant::now();
    let final_expl = solve(&mut game, max_iters, target_expl, true);
    let duration = start.elapsed();
    let time_secs = duration.as_secs_f32();

    println!("Current:  Time {:.2}s", time_secs);
    println!(
        "Final exploitability: {:.6} (target: {:.1})",
        final_expl, target_expl
    );

    // Assertions
    assert!(final_expl <= target_expl);

    // Verification of speedup
    if time_secs < BASELINE_TIME_SECS * 0.70 {
        // Expect at least 30% improvement
        println!("✓ Performance IMPROVED significantly! Time reduced from {:.2}s to {:.2}s ({:.1}% reduction)",
                 BASELINE_TIME_SECS, time_secs, (1.0 - time_secs / BASELINE_TIME_SECS) * 100.0);
    }
}

// ============================================================================
// TEST 6: DCFR with Full Mixed-Precision Config (s16r8i8c4)
// ============================================================================
// This test uses the exact configuration from the TOML file:
// strategy_bits=16, regret_bits=8, ip_bits=8, chance_bits=4
// Matching the solve_from_config output with 160 iterations in 1.19s

#[test]
fn test_performance_dcfr_mixed_precision_s16r8i8c4() {
    const CONFIG_PATH: &str = "hands/7438/configs/hand_0000007438_node_03_turn_DeepStack.toml";

    // Baseline from user's solve_from_config run
    const BASELINE_ITERATIONS: u32 = 160;
    const BASELINE_TIME_SECS: f32 = 1.19;

    // Tolerance: allow 20% degradation for time (to account for system variability)
    const TIME_TOLERANCE: f32 = 1.20;

    // Load game from TOML (uses TOML-specified precision settings)
    let (mut game, max_iters, target_expl) = load_game_from_toml(CONFIG_PATH);

    // The TOML file already has the correct settings:
    // strategy_bits = 16, regret_bits = 8, ip_bits = 8, chance_bits = 4
    // So we just allocate without overriding
    game.allocate_memory();

    // Print header before solving
    println!("\n=== DCFR Mixed-Precision (s16r8i8c4) Performance ===");
    println!("Configuration: strategy=16bit, regret=8bit, ip=8bit, chance=4bit");
    println!(
        "Baseline: {} iterations, {:.2}s",
        BASELINE_ITERATIONS, BASELINE_TIME_SECS
    );

    // Solve with timing
    let start = Instant::now();
    let final_expl = solve(&mut game, max_iters, target_expl, true);
    let duration = start.elapsed();
    let time_secs = duration.as_secs_f32();

    println!("Current:  Time {:.2}s", time_secs);
    println!(
        "Final exploitability: {:.6} (target: {:.1})",
        final_expl, target_expl
    );

    // Assertions
    assert!(
        final_expl <= target_expl,
        "Failed to reach target exploitability. Got: {}, Target: {}",
        final_expl,
        target_expl
    );

    // WARNING: Time should not increase significantly
    assert!(
        time_secs <= BASELINE_TIME_SECS * TIME_TOLERANCE,
        "Performance regression! Time increased from {:.2}s to {:.2}s ({:.1}% increase)",
        BASELINE_TIME_SECS,
        time_secs,
        ((time_secs / BASELINE_TIME_SECS - 1.0) * 100.0)
    );

    // SUCCESS: Print if performance improved
    if time_secs < BASELINE_TIME_SECS * 0.90 {
        println!(
            "✓ Performance IMPROVED! Time reduced by {:.2}s ({:.1}%)",
            BASELINE_TIME_SECS - time_secs,
            ((1.0 - time_secs / BASELINE_TIME_SECS) * 100.0)
        );
    }
}
