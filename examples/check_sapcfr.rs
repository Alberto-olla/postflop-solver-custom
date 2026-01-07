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
    use_compression: bool,
    #[serde(default)]
    quantization: Option<String>,
    #[serde(default)]
    zstd_compression_level: Option<i32>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    lazy_normalization: bool,
    #[serde(default)]
    lazy_normalization_freq: u32,
    #[serde(default)]
    log_encoding: bool,
    #[serde(default = "default_strategy_bits")]
    strategy_bits: u8,
    #[serde(default = "default_chance_bits")]
    chance_bits: u8,
    #[serde(default = "default_algorithm")]
    algorithm: String,
}

impl SolverSettings {
    fn get_quantization_mode(&self) -> Result<QuantizationMode, String> {
        if let Some(ref quant_str) = self.quantization {
            match quant_str.to_lowercase().as_str() {
                "32bit" | "float32" | "f32" => Ok(QuantizationMode::Float32),
                "16bit" | "int16" | "i16" => Ok(QuantizationMode::Int16),
                "16bit-log" | "int16-log" | "i16-log" => Ok(QuantizationMode::Int16Log),
                _ => Err(format!("Invalid quantization mode")),
            }
        } else {
            Ok(QuantizationMode::from_compression_flag(
                self.use_compression,
            ))
        }
    }
}

#[derive(Debug, Deserialize)]
struct OutputSettings {
    #[serde(default = "default_output_dir")]
    directory: String,
}

fn default_add_allin_threshold() -> f64 {
    1.5
}
fn default_force_allin_threshold() -> f64 {
    0.01
}
fn default_merging_threshold() -> f64 {
    0.05
}
fn default_output_dir() -> String {
    "solved_games".to_string()
}
fn default_strategy_bits() -> u8 {
    16
}
fn default_chance_bits() -> u8 {
    16
}
fn default_algorithm() -> String {
    "dcfr".to_string()
}

// --- Bet Size Logic (Truncated for brevity, assuming standard parsing works or is imported) ---
// Actually, I need to copy the helper functions or import them?
// The example uses parse_bet_sizes which is defined in the example locally.
// I must include them.

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
            _ => self.river_bet,
        }
    }
}

const FULL_PRESET: PresetSizing = PresetSizing {
    flop_bet: &[25, 50, 100, 200],
    flop_raise: &[80, 85, 91, 100, 112, 120, 133, 144, 156, 165, 186, 200, 300],
    turn_bet: &[25, 50, 75, 100, 150, 200],
    turn_raise: &[53, 82, 89, 100, 103, 109, 114, 130, 150, 158, 167, 174, 324],
    river_bet: &[50, 75, 100, 150, 200, 300],
    river_raise: &[
        86, 99, 115, 129, 152, 165, 181, 204, 239, 260, 311, 327, 411,
    ],
};

const MEDIUM_PRESET: PresetSizing = PresetSizing {
    flop_bet: &[50, 100],
    flop_raise: &[100, 200],
    turn_bet: &[50, 100, 200],
    turn_raise: &[89, 150, 174],
    river_bet: &[75, 100, 200],
    river_raise: &[152, 239, 311],
};

fn apply_spr_logic(percentages: &[i32], pot: i32, stack: i32) -> String {
    let base = pot.min(stack);
    let mut result: Vec<String> = percentages
        .iter()
        .filter_map(|&pct| {
            let chips = (base as f64 * pct as f64 / 100.0).round() as i32;
            if chips >= stack {
                None
            } else {
                Some(format!("{}c", chips))
            }
        })
        .collect();
    result.dedup();
    result.push("a".to_string());
    result.join(", ")
}

fn expand_bet_size_preset(input: &str, is_bet: bool, pot: i32, stack: i32, street: &str) -> String {
    let normalized = input.trim().to_uppercase();
    let preset = match normalized.as_str() {
        "FULL" => Some(&FULL_PRESET),
        "STANDARD" | "MEDIUM" => Some(&MEDIUM_PRESET),
        "MINIMAL" => None,
        _ => return input.to_string(),
    };
    if let Some(preset) = preset {
        apply_spr_logic(preset.get_percentages(street, is_bet), pot, stack)
    } else {
        apply_spr_logic(&[], pot, stack)
    }
}

fn parse_bet_sizes(
    street: &StreetBetSizes,
    pot: i32,
    stack: i32,
    street_name: &str,
) -> Result<[BetSizeOptions; 2], String> {
    if street.oop_bet.is_empty() && street.ip_bet.is_empty() {
        return Ok([
            BetSizeOptions {
                bet: Vec::new(),
                raise: Vec::new(),
            },
            BetSizeOptions {
                bet: Vec::new(),
                raise: Vec::new(),
            },
        ]);
    }
    let oop_bet_expanded = expand_bet_size_preset(&street.oop_bet, true, pot, stack, street_name);
    let oop_raise_expanded =
        expand_bet_size_preset(&street.oop_raise, false, pot, stack, street_name);
    let ip_bet_expanded = expand_bet_size_preset(&street.ip_bet, true, pot, stack, street_name);
    let ip_raise_expanded =
        expand_bet_size_preset(&street.ip_raise, false, pot, stack, street_name);

    Ok([
        BetSizeOptions::try_from((oop_bet_expanded.as_str(), oop_raise_expanded.as_str()))?,
        BetSizeOptions::try_from((ip_bet_expanded.as_str(), ip_raise_expanded.as_str()))?,
    ])
}

fn main() {
    // Hardcoded config path for testing user request
    let config_path = "hands/7438/configs/hand_0000007438_node_06_river_DeepStack.toml";

    println!("Loading configuration from: {}", config_path);
    let config_content = fs::read_to_string(&config_path).expect("Failed to read config file");
    let config: Config = toml::from_str(&config_content).expect("Failed to parse config file");

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
    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    let card_config = CardConfig {
        range: [
            config.ranges.oop.parse().expect("Invalid OOP range"),
            config.ranges.ip.parse().expect("Invalid IP range"),
        ],
        flop,
        turn,
        river,
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.tree.starting_pot,
        effective_stack: config.tree.effective_stack,
        rake_rate: config.tree.rake_rate,
        rake_cap: config.tree.rake_cap,
        flop_bet_sizes: parse_bet_sizes(
            &config.bet_sizes.flop,
            config.tree.starting_pot,
            config.tree.effective_stack,
            "flop",
        )
        .unwrap(),
        turn_bet_sizes: parse_bet_sizes(
            &config.bet_sizes.turn,
            config.tree.starting_pot,
            config.tree.effective_stack,
            "turn",
        )
        .unwrap(),
        river_bet_sizes: parse_bet_sizes(
            &config.bet_sizes.river,
            config.tree.starting_pot,
            config.tree.effective_stack,
            "river",
        )
        .unwrap(),
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: config.tree.add_allin_threshold,
        force_allin_threshold: config.tree.force_allin_threshold,
        merging_threshold: config.tree.merging_threshold,
    };

    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config).expect("Failed to build action tree");
    let mut game =
        PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game");

    // Force SAPCFR+
    println!("Forcing SAPCFR+ Algorithm...");
    game.set_cfr_algorithm(CfrAlgorithm::SAPCFRPlus);

    let quantization_mode = config
        .solver
        .get_quantization_mode()
        .expect("Invalid quantization");
    game.allocate_memory_with_mode(quantization_mode);

    println!("Starting Solver (SAPCFR+)...");
    let target_exploitability =
        config.tree.starting_pot as f32 * config.solver.target_exploitability_pct / 100.0;

    solve(
        &mut game,
        config.solver.max_iterations as u32,
        target_exploitability,
        true,
    );

    println!("Solver completed.");
}
