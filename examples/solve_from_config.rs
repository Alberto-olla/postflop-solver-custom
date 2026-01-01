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
    /// Legacy field: use `quantization` instead. If both are present, `quantization` takes precedence.
    #[serde(default)]
    use_compression: bool,
    /// Quantization mode: "32bit", "16bit"
    ///
    /// Aliases: "float32"/"f32", "int16"/"i16"
    ///
    /// Memory usage:
    /// - 32bit: Full precision (baseline)
    /// - 16bit: 50% memory saving
    ///
    /// If not specified, falls back to `use_compression` for backward compatibility.
    #[serde(default)]
    quantization: Option<String>,
    #[serde(default)]
    zstd_compression_level: Option<i32>,
    #[serde(default)]
    dry_run: bool,  // Solo stima tree size, non risolve

    // Lazy normalization settings
    #[serde(default)]
    lazy_normalization: bool,
    #[serde(default)]
    lazy_normalization_freq: u32,
    /// Logarithmic encoding (signed magnitude biasing) for regrets (16-bit only)
    #[serde(default)]
    log_encoding: bool,

    /// Mixed precision: strategy precision in bits (16, 8, or 4)
    /// Only works when quantization = "16bit"
    /// - 16: Default (same precision as quantization mode)
    /// - 8: Mixed precision (50% less memory for strategy, ~25% overall)
    /// - 4: Future (75% less memory for strategy)
    #[serde(default = "default_strategy_bits")]
    strategy_bits: u8,
}

impl SolverSettings {
    /// Get the quantization mode, handling backward compatibility with `use_compression`.
    fn get_quantization_mode(&self) -> Result<QuantizationMode, String> {
        if let Some(ref quant_str) = self.quantization {
            // New format: parse from string
            match quant_str.to_lowercase().as_str() {
                "32bit" | "float32" | "f32" => Ok(QuantizationMode::Float32),
                "16bit" | "int16" | "i16" => Ok(QuantizationMode::Int16),
                _ => Err(format!(
                    "Invalid quantization mode: '{}'. Valid options: 32bit, 16bit",
                    quant_str
                )),
            }
        } else {
            // Legacy format: use boolean flag
            Ok(QuantizationMode::from_compression_flag(self.use_compression))
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
fn default_strategy_bits() -> u8 { 16 }  // Default: same precision as quantization mode

/// Definizione dichiarativa di un preset di bet sizing per ogni street
struct PresetSizing {
    flop_bet: &'static [i32],
    flop_raise: &'static [i32],
    turn_bet: &'static [i32],
    turn_raise: &'static [i32],
    river_bet: &'static [i32],
    river_raise: &'static [i32],
}

impl PresetSizing {
    fn get_percentages(&self, street: &str, is_bet: bool) -> &[i32] {
        match (street, is_bet) {
            ("flop", true) => self.flop_bet,
            ("flop", false) => self.flop_raise,
            ("turn", true) => self.turn_bet,
            ("turn", false) => self.turn_raise,
            ("river", true) => self.river_bet,
            ("river", false) => self.river_raise,
            _ => self.river_bet,  // Default: river
        }
    }
}

/// FULL: analisi GTO completa, granularità crescente con le street
const FULL_PRESET: PresetSizing = PresetSizing {
    flop_bet: &[25, 50, 100, 200],
    flop_raise: &[80, 85, 91, 100, 112, 120, 133, 144, 156, 165, 186, 200, 300],
    turn_bet: &[25, 50, 75, 100, 150, 200],
    turn_raise: &[53, 82, 89, 100, 103, 109, 114, 130, 150, 158, 167, 174, 324],
    river_bet: &[50, 75, 100, 150, 200, 300],
    river_raise: &[86, 99, 115, 129, 152, 165, 181, 204, 239, 260, 311, 327, 411],
};

/// MEDIUM/STANDARD: azioni bilanciate, complessità crescente
const MEDIUM_PRESET: PresetSizing = PresetSizing {
    flop_bet: &[50, 100],
    flop_raise: &[100, 200],
    turn_bet: &[50, 100, 200],
    turn_raise: &[89, 150, 174],
    river_bet: &[75, 100, 200],
    river_raise: &[152, 239, 311],
//    flop_bet: &[100],
//    flop_raise: &[200],
//    turn_bet: &[100],
//    turn_raise: &[150],
//    river_bet: &[200],
//    river_raise: &[311],
};

/// Applica logica SPR: calcola su min(pot, stack), converte in chips
/// Il merging_threshold gestirà sizing troppo vicini a livello di ActionTree
fn apply_spr_logic(percentages: &[i32], pot: i32, stack: i32) -> String {
    let base = pot.min(stack);

    let mut result: Vec<String> = percentages
        .iter()
        .filter_map(|&pct| {
            let chips = (base as f64 * pct as f64 / 100.0).round() as i32;
            if chips >= stack {
                None  // Se supera stack, salta
            } else {
                Some(format!("{}c", chips))
            }
        })
        .collect();

    // Rimuovi duplicati esatti
    result.dedup();

    // Aggiungi all-in
    result.push("a".to_string());

    result.join(", ")
}

/// Espande i preset di bet sizing in stringhe effettive
/// TUTTI i preset usano logica SPR (calcolo su min(pot, stack))
fn expand_bet_size_preset(input: &str, is_bet: bool, pot: i32, stack: i32, street: &str) -> String {
    let normalized = input.trim().to_uppercase();

    let preset = match normalized.as_str() {
        "FULL" => Some(&FULL_PRESET),
        "STANDARD" | "MEDIUM" => Some(&MEDIUM_PRESET),
        "MINIMAL" => None,
        _ => return input.to_string(),
    };

    if let Some(preset) = preset {
        let percentages = preset.get_percentages(street, is_bet);
        apply_spr_logic(percentages, pot, stack)
    } else {
        // MINIMAL: solo all-in
        apply_spr_logic(&[], pot, stack)
    }
}

fn parse_bet_sizes(
    street: &StreetBetSizes,
    pot: i32,
    stack: i32,
    street_name: &str,
) -> Result<[BetSizeOptions; 2], String> {
    // Se entrambi i campi sono vuoti, ritorna sizing vuote
    if street.oop_bet.is_empty() && street.ip_bet.is_empty() {
        return Ok([
            BetSizeOptions { bet: Vec::new(), raise: Vec::new() },
            BetSizeOptions { bet: Vec::new(), raise: Vec::new() },
        ]);
    }

    // Espande i preset per OOP
    let oop_bet_expanded = expand_bet_size_preset(&street.oop_bet, true, pot, stack, street_name);
    let oop_raise_expanded = expand_bet_size_preset(&street.oop_raise, false, pot, stack, street_name);

    // Espande i preset per IP
    let ip_bet_expanded = expand_bet_size_preset(&street.ip_bet, true, pot, stack, street_name);
    let ip_raise_expanded = expand_bet_size_preset(&street.ip_raise, false, pot, stack, street_name);

    let oop = BetSizeOptions::try_from((
        oop_bet_expanded.as_str(),
        oop_raise_expanded.as_str(),
    ))?;

    let ip = BetSizeOptions::try_from((
        ip_bet_expanded.as_str(),
        ip_raise_expanded.as_str(),
    ))?;

    Ok([oop, ip])
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
    let (mem_usage_32bit, mem_usage_16bit) = game.memory_usage();

    // Calculate for all modes based on bytes per element
    // The memory_usage() function returns values for 32-bit and 16-bit
    let to_gb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!("\nMemory usage comparison:");
    println!("  32-bit float:    {:.2} GB (baseline)", to_gb(mem_usage_32bit));
    println!("  16-bit integer:  {:.2} GB (-50.0%)", to_gb(mem_usage_16bit));

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

    // Configure log encoding BEFORE memory allocation
    if config.solver.log_encoding {
        game.set_log_encoding(true);
        println!("Log encoding (signed magnitude biasing): enabled (16-bit mode only)");
    }

    // Configure mixed precision if requested
    let quantization_mode = config.solver.get_quantization_mode()
        .expect("Invalid quantization configuration");

    if config.solver.strategy_bits != 16 {
        if quantization_mode != QuantizationMode::Int16 {
            eprintln!("Warning: strategy_bits only works with quantization='16bit', ignoring");
        } else {
            game.set_strategy_bits(config.solver.strategy_bits);
            println!("Mixed precision: {}-bit strategy (regrets stay 16-bit)",
                     config.solver.strategy_bits);
        }
    }

    // Allocate memory
    game.allocate_memory_with_mode(quantization_mode);
    let bits = match quantization_mode {
        QuantizationMode::Float32 => 32,
        QuantizationMode::Int16 => 16,
    };
    println!("Using {}-bit precision ({})", bits,
        if quantization_mode.is_compressed() { "compressed" } else { "full precision" });

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

    let filename = format!("{}-{}.bin", config_filename, bits);

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
