use postflop_solver::*;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Config {
    ranges: Ranges,
    cards: Cards,
    tree: TreeSettings,
    bet_sizes: BetSizes,
    solver: SolverSettings,
    output: OutputSettings,
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
    #[serde(default)]
    zstd_compression_level: Option<i32>,
    #[serde(default)]
    dry_run: bool,  // Solo stima tree size, non risolve

    // Lazy normalization settings
    #[serde(default)]
    lazy_normalization: bool,
    #[serde(default)]
    lazy_normalization_freq: u32,

    // Granular precision control for each storage component
    /// Strategy precision in bits (8, 16, or 32)
    /// - 8: Ultra-compressed (1 byte per element, ~75% memory saving for strategy)
    /// - 16: Balanced (2 bytes per element, default)
    /// - 32: Full precision (4 bytes per element)
    #[serde(default = "default_strategy_bits")]
    strategy_bits: u8,

    /// Regret precision in bits (16 or 32) - for storage2 and storage4
    /// - 16: Compressed (2 bytes per element, default)
    /// - 32: Full precision (4 bytes per element)
    #[serde(default = "default_regret_bits")]
    regret_bits: u8,

    /// IP counterfactual values precision in bits (16 or 32) - for storage_ip
    /// - 16: Compressed (2 bytes per element, default)
    /// - 32: Full precision (4 bytes per element)
    #[serde(default = "default_ip_bits")]
    ip_bits: u8,

    /// Chance counterfactual values precision in bits (8, 16, or 32)
    /// - 8: Ultra-compressed (1 byte per element)
    /// - 16: Balanced (2 bytes per element, default)
    /// - 32: Full precision (4 bytes per element)
    #[serde(default = "default_chance_bits")]
    chance_bits: u8,

    /// CFR algorithm variant: "dcfr", "dcfr+", "pdcfr+", "sapcfr+"
    /// - "dcfr": Original Discounted CFR (uses separate α and β discount factors)
    /// - "dcfr+": DCFR+ (uses single α discount factor with regret clipping)
    /// - "pdcfr+": Predictive DCFR+ (requires 50% more memory for storage4)
    /// - "sapcfr+": Self-Adaptive Predictive CFR+ (requires 50% more memory for storage4)
    ///
    /// If not specified, defaults to "dcfr" for backward compatibility.
    #[serde(default = "default_algorithm")]
    algorithm: String,

    /// Enable dynamic regret-based pruning (branch skipping)
    ///
    /// When enabled, actions with sufficiently negative regret are temporarily skipped
    /// to reduce computational requirements. The pruning threshold is dynamically
    /// calculated based on:
    /// - Maximum payoff range (Delta)
    /// - Current iteration number
    /// - Accumulated regret values
    ///
    /// This feature is particularly effective with DCFR (beta=0.5), as negative regrets
    /// decay towards -infinity, making pruning safe and effective.
    ///
    /// Default: false (disabled for backward compatibility)
    #[serde(default = "default_enable_pruning")]
    enable_pruning: bool,

    // DEPRECATED/EXPERIMENTAL: Legacy features
    /// DEPRECATED: Use granular *_bits parameters instead
    #[serde(default)]
    #[allow(dead_code)]
    quantization: Option<String>,
    /// DEPRECATED: Use granular *_bits parameters instead
    #[serde(default)]
    #[allow(dead_code)]
    use_compression: bool,
    /// EXPERIMENTAL: Logarithmic encoding (not in active use)
    #[serde(default)]
    #[allow(dead_code)]
    log_encoding: bool,
}

impl SolverSettings {
    /// DEPRECATED: This method is no longer needed with granular *_bits parameters.
    #[allow(dead_code)]
    fn get_quantization_mode(&self) -> Result<QuantizationMode, String> {
        if let Some(ref quant_str) = self.quantization {
            // New format: parse from string
            match quant_str.to_lowercase().as_str() {
                "32bit" | "float32" | "f32" => Ok(QuantizationMode::Float32),
                "16bit" | "int16" | "i16" => Ok(QuantizationMode::Int16),
                "16bit-log" | "int16-log" | "i16-log" => Ok(QuantizationMode::Int16Log),
                _ => Err(format!(
                    "Invalid quantization mode: '{}'. Valid options: 32bit, 16bit, 16bit-log",
                    quant_str
                )),
            }
        } else {
            // Legacy format: use boolean flag
            Ok(QuantizationMode::from_compression_flag(self.use_compression))
        }
    }

    /// Get the CFR algorithm variant.
    fn get_algorithm(&self) -> Result<postflop_solver::CfrAlgorithm, String> {
        match self.algorithm.to_lowercase().as_str() {
            "dcfr" => Ok(postflop_solver::CfrAlgorithm::DCFR),
            "dcfr+" | "dcrfplus" => Ok(postflop_solver::CfrAlgorithm::DCFRPlus),
            "pdcfr+" | "pdcfrplus" => Ok(postflop_solver::CfrAlgorithm::PDCFRPlus),
            "sapcfr+" | "sapcfrplus" => Ok(postflop_solver::CfrAlgorithm::SAPCFRPlus),
            _ => Err(format!(
                "Invalid algorithm: '{}'. Valid options: dcfr, dcfr+, pdcfr+, sapcfr+",
                self.algorithm
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
struct OutputSettings {
    #[serde(default = "default_output_dir")]
    directory: String,
}

// Default functions
fn default_add_allin_threshold() -> f64 { 1.5 }
fn default_force_allin_threshold() -> f64 { 0.01 }  // 1% - permette sizing vicini allo stack
fn default_merging_threshold() -> f64 { 0.05 }  // 5% - buon compromesso precisione/performance
fn default_output_dir() -> String { "solved_games".to_string() }
fn default_strategy_bits() -> u8 { 16 }  // Default: 16-bit strategy (balanced)
fn default_regret_bits() -> u8 { 16 }    // Default: 16-bit regrets (balanced)
fn default_ip_bits() -> u8 { 16 }        // Default: 16-bit IP cfvalues (balanced)
fn default_chance_bits() -> u8 { 16 }    // Default: 16-bit chance cfvalues (balanced)
fn default_algorithm() -> String { "dcfr".to_string() }  // Default: DCFR for backward compatibility
fn default_enable_pruning() -> bool { false }  // Default: disabled for backward compatibility

fn parse_bet_sizes(
    street: &StreetBetSizes,
    pot: i32,
    stack: i32,
    street_name: &str,
) -> Result<[BetSizeOptions; 2], String> {
    parse_bet_sizes_with_preset(
        &street.oop_bet,
        &street.oop_raise,
        &street.ip_bet,
        &street.ip_raise,
        pot,
        stack,
        street_name,
    )
}

fn main() {
    // Read config file
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());

    println!("Loading configuration from: {}", config_path);
    let config_content = fs::read_to_string(&config_path)
        .expect("Failed to read config file");

    let config: Config = toml::from_str(&config_content)
        .expect("Failed to parse config file");

    println!("Configuration loaded successfully!");
    println!("  OOP range: {}", config.ranges.oop);
    println!("  IP range: {}", config.ranges.ip);
    println!("  Board: {} {} {}", config.cards.flop, config.cards.turn, config.cards.river);

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

    // Determine initial state based on cards
    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    println!("  Initial state: {:?}", initial_state);

    // Parse ranges
    let oop_range = config.ranges.oop.parse().expect("Invalid OOP range");
    let ip_range = config.ranges.ip.parse().expect("Invalid IP range");

    // Create card config
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    // Parse bet sizes for each street
    let flop_bet_sizes = parse_bet_sizes(
        &config.bet_sizes.flop,
        config.tree.starting_pot,
        config.tree.effective_stack,
        "flop",
    ).expect("Invalid flop bet sizes");
    let turn_bet_sizes = parse_bet_sizes(
        &config.bet_sizes.turn,
        config.tree.starting_pot,
        config.tree.effective_stack,
        "turn",
    ).expect("Invalid turn bet sizes");
    let river_bet_sizes = parse_bet_sizes(
        &config.bet_sizes.river,
        config.tree.starting_pot,
        config.tree.effective_stack,
        "river",
    ).expect("Invalid river bet sizes");

    // Create tree config
    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.tree.starting_pot,
        effective_stack: config.tree.effective_stack,
        rake_rate: config.tree.rake_rate,
        rake_cap: config.tree.rake_cap,
        flop_bet_sizes,
        turn_bet_sizes,
        river_bet_sizes,
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: config.tree.add_allin_threshold,
        force_allin_threshold: config.tree.force_allin_threshold,
        merging_threshold: config.tree.merging_threshold,
    };

    // Build game tree
    println!("\nBuilding game tree...");
    let action_tree = ActionTree::new(tree_config).expect("Failed to build action tree");
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .expect("Failed to create game");

    // Check memory usage for all quantization modes
    let (mem_usage_32bit, mem_usage_16bit, mem_usage_current) = game.memory_usage();

    // Helper to format size with appropriate unit
    let format_size = |bytes: u64| {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        let gb = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if gb >= 1.0 {
            format!("{:.2} GB ({:.1} MB)", gb, mb)
        } else {
            format!("{:.2} MB", mb)
        }
    };

    println!("\nMemory usage comparison:");
    println!("  32-bit float:    {:>20} (baseline)", format_size(mem_usage_32bit));
    
    let savings_16bit = 100.0 * (1.0 - mem_usage_16bit as f64 / mem_usage_32bit as f64);
    println!("  16-bit integer:  {:>20} ({:+.1}%)", format_size(mem_usage_16bit), -savings_16bit);
    
    let savings_current = 100.0 * (1.0 - mem_usage_current as f64 / mem_usage_32bit as f64);
    println!("  Current config:  {:>20} ({:+.1}%)", format_size(mem_usage_current), -savings_current);

    // Se dry_run è attivo, fermiamo qui
    if config.solver.dry_run {
        println!("\n✓ Dry run completato - albero NON risolto");
        println!("Per risolvere, imposta dry_run = false nel config");
        return;
    }

    // Configure lazy normalization BEFORE memory allocation
    if config.solver.lazy_normalization {
        game.set_lazy_normalization(true, config.solver.lazy_normalization_freq);
        println!("Lazy normalization: enabled (freq: {})",
                 if config.solver.lazy_normalization_freq == 0 {
                     "finalization only".to_string()
                 } else {
                     format!("every {} iterations", config.solver.lazy_normalization_freq)
                 });
    }

    // Configure CFR algorithm variant
    let algorithm = config.solver.get_algorithm()
        .expect("Invalid algorithm configuration");
    game.set_cfr_algorithm(algorithm);
    let algorithm_name = match algorithm {
        postflop_solver::CfrAlgorithm::DCFR => "DCFR (dual discount factors)",
        postflop_solver::CfrAlgorithm::DCFRPlus => "DCFR+ (single discount + clipping)",
        postflop_solver::CfrAlgorithm::PDCFRPlus => "PDCFR+ (predictive discounted CFR+)",
        postflop_solver::CfrAlgorithm::SAPCFRPlus => "SAPCFR+ (asymmetric predictive CFR+)",
    };
    println!("Using algorithm: {}", algorithm_name);

    // Configure pruning
    game.set_enable_pruning(config.solver.enable_pruning);
    if config.solver.enable_pruning {
        // Warn if pruning is enabled with non-DCFR algorithms
        if algorithm != postflop_solver::CfrAlgorithm::DCFR {
            eprintln!("⚠️  Warning: Pruning is only effective with DCFR algorithm (beta=0.5).");
            eprintln!("            Other algorithms clip negative regrets, making pruning ineffective.");
        }
        println!("Regret-based pruning: ENABLED (dynamic threshold)");
    }

    // Configure granular precision for each storage component BEFORE memory allocation
    game.set_strategy_bits(config.solver.strategy_bits);
    game.set_regret_bits(config.solver.regret_bits);
    game.set_ip_bits(config.solver.ip_bits);
    game.set_chance_bits(config.solver.chance_bits);

    println!("\nMemory precision configuration (estimated):");
    let detailed = game.estimated_memory_usage_detailed();
    let total = detailed.total() as f64;
    let to_mb = |bytes: u64| bytes as f64 / 1_048_576.0;

    println!("  Strategy (storage1):      {:>2}-bit  ({:>8.2} MB, {:>5.1}%)",
             config.solver.strategy_bits, to_mb(detailed.strategy), 100.0 * detailed.strategy as f64 / total);
    println!("  Regrets (storage2/4):     {:>2}-bit  ({:>8.2} MB, {:>5.1}%)",
             config.solver.regret_bits, to_mb(detailed.regrets + detailed.storage4), 100.0 * (detailed.regrets + detailed.storage4) as f64 / total);
    println!("  IP CFValues (storage_ip): {:>2}-bit  ({:>8.2} MB, {:>5.1}%)",
             config.solver.ip_bits, to_mb(detailed.ip_cfvalues), 100.0 * detailed.ip_cfvalues as f64 / total);
    println!("  Chance CFValues:          {:>2}-bit  ({:>8.2} MB, {:>5.1}%)",
             config.solver.chance_bits, to_mb(detailed.chance_cfvalues), 100.0 * detailed.chance_cfvalues as f64 / total);
    println!("  Misc (node arena, etc):          ({:>8.2} MB, {:>5.1}%)",
             to_mb(detailed.misc), 100.0 * detailed.misc as f64 / total);

    // Allocate memory using configured precision settings
    game.allocate_memory();

    // Solve
    println!("\nSolving (max {} iterations)...", config.solver.max_iterations);
    let target_exploitability = config.tree.starting_pot as f32 * config.solver.target_exploitability_pct / 100.0;

    let start_time = std::time::Instant::now();
    let _exploitability = solve(
        &mut game,
        config.solver.max_iterations as u32,
        target_exploitability,
        true
    );
    let elapsed = start_time.elapsed();

    println!("\n✓ Solving completed in {:.2}s", elapsed.as_secs_f64());
    // println!("Final exploitability: {:.4}", exploitability);
    // println!("Target exploitability: {:.4}", target_exploitability);

    // Calculate results
    game.cache_normalized_weights();
    // let equity_oop = game.equity(0);
    // let equity_ip = game.equity(1);
    // let ev_oop = game.expected_values(0);
    // let ev_ip = game.expected_values(1);
    // let weights_oop = game.normalized_weights(0);
    // let weights_ip = game.normalized_weights(1);

    // let avg_equity_oop = compute_average(&equity_oop, weights_oop);
    // let avg_equity_ip = compute_average(&equity_ip, weights_ip);
    // let avg_ev_oop = compute_average(&ev_oop, weights_oop);
    // let avg_ev_ip = compute_average(&ev_ip, weights_ip);

    // println!("\nResults:");
    // println!("  OOP - Equity: {:.2}%, EV: {:.2}", avg_equity_oop * 100.0, avg_ev_oop);
    // println!("  IP  - Equity: {:.2}%, EV: {:.2}", avg_equity_ip * 100.0, avg_ev_ip);

    // Generate output filename from config file name
    let config_filename = Path::new(&config_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("solved");

    // Generate suffix based on precision configuration
    let precision_suffix = if config.solver.strategy_bits == 16
        && config.solver.regret_bits == 16
        && config.solver.ip_bits == 16
        && config.solver.chance_bits == 16 {
        "16".to_string()  // Standard 16-bit config
    } else if config.solver.strategy_bits == 32
        && config.solver.regret_bits == 32
        && config.solver.ip_bits == 32
        && config.solver.chance_bits == 32 {
        "32".to_string()  // Full 32-bit precision
    } else {
        // Custom mixed precision: encode as s{strategy}r{regret}i{ip}c{chance}
        format!("s{}r{}i{}c{}",
                config.solver.strategy_bits,
                config.solver.regret_bits,
                config.solver.ip_bits,
                config.solver.chance_bits)
    };

    let filename = format!("{}-{}.bin", config_filename, precision_suffix);

    // Create output directory if it doesn't exist
    let output_dir = Path::new(&config.output.directory);
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).expect("Failed to create output directory");
    }

    let output_path = output_dir.join(&filename);

    // Save
    println!("\nSaving to: {}", output_path.display());
    save_data_to_file(
        &game,
        &format!("Config: {}", config_filename),
        &output_path,
        config.solver.zstd_compression_level
    ).expect("Failed to save game");

    // Get actual file size
    let file_size = fs::metadata(&output_path)
        .expect("Failed to read file metadata")
        .len();

    println!("✓ Game saved successfully!");
    println!("\nFile: {}", filename);
    println!("Size: {:.2} MB", file_size as f64 / (1024.0 * 1024.0));
    if let Some(level) = config.solver.zstd_compression_level {
        println!("Compression: zstd level {}", level);
    } else {
        println!("Compression: none");
    }
}
