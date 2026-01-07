//! Bet size preset expansion utilities
//!
//! This module provides functionality to expand bet size presets (FULL, MEDIUM, MINIMAL)
//! into concrete bet size strings. Used by both examples and tests to ensure consistent
//! preset handling.

use crate::BetSizeOptions;

/// Definizione dichiarativa di un preset di bet sizing per ogni street
pub struct PresetSizing {
    pub flop_bet: &'static [i32],
    pub flop_raise: &'static [i32],
    pub turn_bet: &'static [i32],
    pub turn_raise: &'static [i32],
    pub river_bet: &'static [i32],
    pub river_raise: &'static [i32],
}

impl PresetSizing {
    pub fn get_percentages(&self, street: &str, is_bet: bool) -> &[i32] {
        match (street, is_bet) {
            ("flop", true) => self.flop_bet,
            ("flop", false) => self.flop_raise,
            ("turn", true) => self.turn_bet,
            ("turn", false) => self.turn_raise,
            ("river", true) => self.river_bet,
            ("river", false) => self.river_raise,
            _ => self.river_bet, // Default: river
        }
    }
}

/// FULL: analisi GTO completa, granularità crescente con le street
pub const FULL_PRESET: PresetSizing = PresetSizing {
    flop_bet: &[25, 50, 100, 200],
    flop_raise: &[80, 85, 91, 100, 112, 120, 133, 144, 156, 165, 186, 200, 300],
    turn_bet: &[25, 50, 75, 100, 150, 200],
    turn_raise: &[53, 82, 89, 100, 103, 109, 114, 130, 150, 158, 167, 174, 324],
    river_bet: &[50, 75, 100, 150, 200, 300],
    river_raise: &[
        86, 99, 115, 129, 152, 165, 181, 204, 239, 260, 311, 327, 411,
    ],
};

/// MEDIUM/STANDARD: azioni bilanciate, complessità crescente
pub const MEDIUM_PRESET: PresetSizing = PresetSizing {
    flop_bet: &[50, 100],
    flop_raise: &[100, 200],
    turn_bet: &[50, 100, 200],
    turn_raise: &[89, 150, 174],
    river_bet: &[75, 100, 200],
    river_raise: &[152, 239, 311],
};

/// Applica logica SPR: calcola su min(pot, stack), converte in chips
/// Il merging_threshold gestirà sizing troppo vicini a livello di ActionTree
pub fn apply_spr_logic(percentages: &[i32], pot: i32, stack: i32) -> String {
    let base = pot.min(stack);

    let mut result: Vec<String> = percentages
        .iter()
        .filter_map(|&pct| {
            let chips = (base as f64 * pct as f64 / 100.0).round() as i32;
            if chips >= stack {
                None // Se supera stack, salta
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
pub fn expand_bet_size_preset(
    input: &str,
    is_bet: bool,
    pot: i32,
    stack: i32,
    street: &str,
) -> String {
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

/// Parse bet sizes for a street, expanding presets as needed
pub fn parse_bet_sizes_with_preset(
    oop_bet: &str,
    oop_raise: &str,
    ip_bet: &str,
    ip_raise: &str,
    pot: i32,
    stack: i32,
    street_name: &str,
) -> Result<[BetSizeOptions; 2], String> {
    // Se entrambi i campi sono vuoti, ritorna sizing vuote
    if oop_bet.is_empty() && ip_bet.is_empty() {
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

    // Espande i preset per OOP
    let oop_bet_expanded = expand_bet_size_preset(oop_bet, true, pot, stack, street_name);
    let oop_raise_expanded = expand_bet_size_preset(oop_raise, false, pot, stack, street_name);

    // Espande i preset per IP
    let ip_bet_expanded = expand_bet_size_preset(ip_bet, true, pot, stack, street_name);
    let ip_raise_expanded = expand_bet_size_preset(ip_raise, false, pot, stack, street_name);

    let oop = BetSizeOptions::try_from((oop_bet_expanded.as_str(), oop_raise_expanded.as_str()))?;

    let ip = BetSizeOptions::try_from((ip_bet_expanded.as_str(), ip_raise_expanded.as_str()))?;

    Ok([oop, ip])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_preset_expansion() {
        let result = expand_bet_size_preset("full", true, 100, 1000, "flop");
        assert!(result.contains("25c"));
        assert!(result.contains("50c"));
        assert!(result.contains("100c"));
        assert!(result.contains("a"));
    }

    #[test]
    fn test_medium_preset_expansion() {
        let result = expand_bet_size_preset("medium", true, 100, 1000, "flop");
        assert!(result.contains("50c"));
        assert!(result.contains("100c"));
        assert!(result.contains("a"));
    }

    #[test]
    fn test_minimal_preset_expansion() {
        let result = expand_bet_size_preset("minimal", true, 100, 1000, "flop");
        assert_eq!(result, "a");
    }

    #[test]
    fn test_non_preset_passthrough() {
        let result = expand_bet_size_preset("50%, 100%", true, 100, 1000, "flop");
        assert_eq!(result, "50%, 100%");
    }

    #[test]
    fn test_spr_logic_filters_above_stack() {
        // With pot=100, stack=150, 200% would be 200 chips which exceeds stack
        let result = apply_spr_logic(&[50, 100, 200], 100, 150);
        assert!(result.contains("50c"));
        assert!(result.contains("100c"));
        assert!(!result.contains("200c")); // Should be filtered out
        assert!(result.contains("a"));
    }
}
