Piano di Implementazione: Algoritmi CFR Avanzati (APDCFR+, DCFR+, APCFR+, SAPCFR+)

     📋 Obiettivo

     Implementare varianti avanzate dell'algoritmo CFR nel postflop-solver, mantenendo la compatibilità con l'implementazione DCFR esistente e permettendo la selezione
      dell'algoritmo via configurazione TOML.

     🎯 Algoritmi da Implementare

     1. APDCFR+ (Asymmetric Predictive Discounted CFR+)

     - Combina DCFR con asimmetria di step size
     - Migliore performance per poker secondo il paper
     - Formula: R̂ᵗ = [λt^β/(κ+t^β) * Rᵗ + 1/(1+αᵗ) * rᵗ⁻¹]⁺

     2. DCFR+ (Discounted CFR+)

     - Miglioria di DCFR attuale
     - Floor negativi a 0 (come CFR+)
     - Weighted averaging: iterazione T pesata per T

     3. APCFR+ (Asymmetric Predictive CFR+)

     - Variante senza discounting DCFR
     - Formula: R̂ᵗ = [Rᵗ + 1/(1+αᵗ) * rᵗ⁻¹]⁺

     4. SAPCFR+ (Simple Asymmetric Predictive CFR+)

     - Versione semplificata con α=2 fisso
     - Formula: R̂ᵗ = [Rᵗ + 1/3 * rᵗ⁻¹]⁺

     5. DCFR (attuale)

     - Mantenuto per compatibilità

     📊 Analisi Implementazione Attuale

     File Chiave

     - src/solver.rs: Logica solver principale (DCFR)
       - solve(): Entry point (linea 44)
       - solve_step(): Singola iterazione (linea 110)
       - solve_recursive(): Core CFR logic (linea 137-351)
       - DiscountParams: Parametri discount (linea 12-37)
       - regret_matching(): Regret matching (linea 392-477)
     - src/game/node.rs: Struttura nodi e storage
       - PostFlopNode: storage1 (strategy), storage2 (regrets)
       - Supporta quantization (Float32, Int16, 8-bit, 4-bit)
     - examples/solve_from_config.rs: Config parsing
       - SolverSettings: Parametri solver (linea 66-105)
       - Parsing TOML config

     Storage Attuale

     PostFlopNode {
         storage1: *mut u8,  // Cumulative strategy
         storage2: *mut u8,  // Cumulative regrets
         storage3: *mut u8,  // IP cfvalues (opzionale)
         scale1: f32,        // Quantization scale strategy
         scale2: f32,        // Quantization scale regrets
         scale3: f32,        // Quantization scale IP
     }

     🏗️ Architettura Proposta

     1. Nuovo Enum per Algoritmi

     File: src/solver.rs (inizio file, dopo imports)

     #[derive(Debug, Clone, Copy, PartialEq, Eq)]
     pub enum SolverAlgorithm {
         DCFR,       // Algoritmo attuale
         DCFRPlus,   // DCFR con floor a 0 sui negativi
         APCFRPlus,  // Asymmetric Predictive CFR+
         SAPCFRPlus, // Simple APCFR+ (α=2)
         APDCFRPlus, // Asymmetric Predictive Discounted CFR+
     }

     impl Default for SolverAlgorithm {
         fn default() -> Self {
             SolverAlgorithm::DCFR
         }
     }

     impl std::str::FromStr for SolverAlgorithm {
         type Err = String;

         fn from_str(s: &str) -> Result<Self, Self::Err> {
             match s.to_lowercase().as_str() {
                 "dcfr" => Ok(SolverAlgorithm::DCFR),
                 "dcfr+" | "dcfr_plus" => Ok(SolverAlgorithm::DCFRPlus),
                 "apcfr+" | "apcfr_plus" => Ok(SolverAlgorithm::APCFRPlus),
                 "sapcfr+" | "sapcfr_plus" => Ok(SolverAlgorithm::SAPCFRPlus),
                 "apdcfr+" | "apdcfr_plus" => Ok(SolverAlgorithm::APDCFRPlus),
                 _ => Err(format!("Unknown algorithm: {}", s)),
             }
         }
     }

     2. Parametri Algoritmo Estesi

     File: src/solver.rs (sostituire/estendere DiscountParams)

     pub struct AlgorithmParams {
         // DCFR parameters (esistenti)
         pub alpha_t: f32,   // Positive regret discount
         pub beta_t: f32,    // Negative regret discount
         pub gamma_t: f32,   // Strategy discount

         // APDCFR+ additional parameters
         pub lambda: f32,    // DCFR scaling (default: 20.0 per HUNL)
         pub kappa: f32,     // DCFR offset (default: 500.0 per HUNL)
         pub dcfr_beta: f32, // DCFR exponent (default: 1.5)

         // Asymmetry parameters
         pub asymmetry_alpha: Option<f32>,  // Fixed alpha (None = auto-learning)
         pub alpha_max: f32,                // Max alpha for auto-learning (default: 9.0)

         // Algorithm selection
         pub algorithm: SolverAlgorithm,
     }

     impl AlgorithmParams {
         pub fn new(iteration: usize, algorithm: SolverAlgorithm) -> Self {
             let t = iteration as f64;

             // Compute DCFR discounts (for DCFR, DCFRPlus, APDCFRPlus)
             let (alpha_t, beta_t, gamma_t) = if matches!(
                 algorithm,
                 SolverAlgorithm::DCFR | SolverAlgorithm::DCFRPlus | SolverAlgorithm::APDCFRPlus
             ) {
                 // Alpha: t^1.5 / (t^1.5 + 1)
                 let t_alpha = (t - 1.0).max(0.0);
                 let pow_alpha = t_alpha * t_alpha.sqrt();
                 let alpha = (pow_alpha / (pow_alpha + 1.0)) as f32;

                 // Beta
                 let beta = if algorithm == SolverAlgorithm::DCFRPlus {
                     0.0  // Floor negativi a 0
                 } else {
                     0.5  // DCFR standard
                 };

                 // Gamma: reset at powers of 4
                 let nearest_power_of_4 = if iteration == 0 {
                     0
                 } else {
                     1 << ((iteration.leading_zeros() ^ 31) & !1)
                 };
                 let t_gamma = (iteration - nearest_power_of_4) as f64;
                 let pow_gamma = t_gamma / (t_gamma + 1.0);
                 let gamma = (pow_gamma * pow_gamma * pow_gamma) as f32;

                 (alpha, beta, gamma)
             } else {
                 // No discounting for APCFR+, SAPCFR+
                 (1.0, 1.0, 1.0)
             };

             AlgorithmParams {
                 alpha_t,
                 beta_t,
                 gamma_t,
                 lambda: 20.0,
                 kappa: 500.0,
                 dcfr_beta: 1.5,
                 asymmetry_alpha: None,  // Auto-learning by default
                 alpha_max: 9.0,
                 algorithm,
             }
         }
     }

     3. Storage per Prediction e Alpha Tracking

     Opzione Scelta: Usare HashMap temporanei durante solve (Option C)
     - Pro: No modifiche a PostFlopNode, compatibilità mantenu ta
     - Pro: Memory efficient (solo nodi visitati)
     - Con: Overhead HashMap lookup (minimo con FxHashMap)

     File: src/solver.rs

     use rustc_hash::FxHashMap;

     // Storage per previous regrets (per prediction)
     type PrevRegretStorage = FxHashMap<usize, Vec<f32>>;  // node_index -> prev_regret

     // Storage per alpha auto-learning tracking
     struct AlphaTracker {
         sum_regret_diff_sq: FxHashMap<usize, f32>,   // Σ||rᵗ - rᵗ⁻¹||²
         sum_cum_regret_diff_sq: FxHashMap<usize, f32>, // Σ||Rᵗ⁺¹ - Rᵗ||²
     }

     impl AlphaTracker {
         fn new() -> Self {
             AlphaTracker {
                 sum_regret_diff_sq: FxHashMap::default(),
                 sum_cum_regret_diff_sq: FxHashMap::default(),
             }
         }

         fn compute_alpha(&self, node_index: usize, alpha_max: f32) -> f32 {
             let sum_r_diff = self.sum_regret_diff_sq.get(&node_index).copied().unwrap_or(0.0);
             let sum_R_diff = self.sum_cum_regret_diff_sq.get(&node_index).copied().unwrap_or(0.0);

             if sum_R_diff > 0.0 {
                 (sum_r_diff / sum_R_diff).sqrt().min(alpha_max)
             } else {
                 2.0  // Default fallback
             }
         }

         fn update(&mut self, node_index: usize, regret_diff_sq: f32, cum_regret_diff_sq: f32) {
             *self.sum_regret_diff_sq.entry(node_index).or_insert(0.0) += regret_diff_sq;
             *self.sum_cum_regret_diff_sq.entry(node_index).or_insert(0.0) += cum_regret_diff_sq;
         }
     }

     📝 Modifiche File per File

     1. src/solver.rs

     Modifiche alle Signature delle Funzioni

     // BEFORE
     pub fn solve<T: Game>(
         game: &mut T,
         max_num_iterations: u32,
         target_exploitability: f32,
         print_progress: bool,
     ) -> f32

     // AFTER
     pub fn solve<T: Game>(
         game: &mut T,
         max_num_iterations: u32,
         target_exploitability: f32,
         print_progress: bool,
         algorithm: SolverAlgorithm,  // NEW
         asymmetry_mode: AsymmetryMode, // NEW: Auto vs Fixed(2.0)
     ) -> f32

     // Nuovo enum per alpha mode
     #[derive(Debug, Clone, Copy)]
     pub enum AsymmetryMode {
         AutoLearning,  // Calcola alpha automaticamente
         Fixed(f32),    // Usa valore fisso (es. 2.0 per SAPCFR+)
     }

     impl Default for AsymmetryMode {
         fn default() -> Self {
             AsymmetryMode::AutoLearning
         }
     }

     Modifiche a solve() Function (linea 44-108)

     pub fn solve<T: Game>(
         game: &mut T,
         max_num_iterations: u32,
         target_exploitability: f32,
         print_progress: bool,
         algorithm: SolverAlgorithm,
         asymmetry_mode: AsymmetryMode,
     ) -> f32 {
         // Initialize prediction storage for APCFR+ variants
         let mut prev_regrets: PrevRegretStorage = FxHashMap::default();
         let mut alpha_tracker = if matches!(asymmetry_mode, AsymmetryMode::AutoLearning) {
             Some(AlphaTracker::new())
         } else {
             None
         };

         // Existing solve loop...
         for iteration in 1..=max_num_iterations {
             let params = AlgorithmParams::new(iteration as usize, algorithm);

             // Set asymmetry alpha based on mode
             let params = if needs_asymmetry(algorithm) {
                 let alpha = match asymmetry_mode {
                     AsymmetryMode::AutoLearning => {
                         alpha_tracker.as_ref()
                             .map(|t| t.compute_alpha(0, params.alpha_max))
                             .unwrap_or(2.0)
                     }
                     AsymmetryMode::Fixed(val) => val,
                 };
                 AlgorithmParams { asymmetry_alpha: Some(alpha), ..params }
             } else {
                 params
             };

             solve_step(
                 game,
                 iteration as usize,
                 &params,
                 &mut prev_regrets,
                 alpha_tracker.as_mut(),
             );

             // Rest of solve loop (exploitability check, etc.)...
         }

         // Return final exploitability
     }

     fn needs_asymmetry(algorithm: SolverAlgorithm) -> bool {
         matches!(
             algorithm,
             SolverAlgorithm::APCFRPlus | SolverAlgorithm::SAPCFRPlus | SolverAlgorithm::APDCFRPlus
         )
     }

     Modifiche a solve_step() (linea 110-135)

     fn solve_step<T: Game>(
         game: &mut T,
         iteration: usize,
         params: &AlgorithmParams,
         prev_regrets: &mut PrevRegretStorage,  // NEW
         alpha_tracker: Option<&mut AlphaTracker>, // NEW
     ) {
         // Alternate between players
         let player = ((iteration - 1) & 1) as usize;

         // Solve for this player
         solve_recursive(
             game,
             game.root(),
             player,
             params,
             1.0,
             prev_regrets,
             alpha_tracker,
         );
     }

     Modifiche a solve_recursive() (linea 137-351) - CORE CHANGES

     Questa è la modifica più significativa. Dividiamo in sezioni:

     A. Signature aggiornata

     fn solve_recursive<T: Game>(
         game: &T,
         node: &MutexLike<T::Node>,
         player: usize,
         params: &AlgorithmParams,
         cfreach: f32,
         prev_regrets: &mut PrevRegretStorage,  // NEW
         alpha_tracker: Option<&mut AlphaTracker>, // NEW
     ) -> f32

     B. Current player node - calcolo strategia con prediction

     // Linea ~256-351: quando node.player() == player
     let node_index = /* calcola index univoco del nodo */;
     let num_actions = node.num_actions();
     let num_elements = node.num_elements() as usize;

     // Get cumulative regrets
     let cum_regrets = node.regrets();  // R^t

     // COMPUTE EXPLICIT REGRETS (R̂^t) based on algorithm
     let explicit_regrets = match params.algorithm {
         SolverAlgorithm::APCFRPlus | SolverAlgorithm::SAPCFRPlus | SolverAlgorithm::APDCFRPlus => {
             // Get previous instantaneous regret (r^{t-1})
             let prev_regret = prev_regrets.get(&node_index);

             // Get alpha
             let alpha = params.asymmetry_alpha.unwrap_or(2.0);
             let prediction_weight = 1.0 / (1.0 + alpha);

             // R̂^t = [R^t + 1/(1+α) * r^{t-1}]^+
             let mut explicit = cum_regrets.to_vec();
             if let Some(prev_r) = prev_regret {
                 for i in 0..num_elements {
                     explicit[i] += prediction_weight * prev_r[i];
                 }
             }

             // Apply positive projection [·]^+
             for val in explicit.iter_mut() {
                 *val = val.max(0.0);
             }

             explicit
         }
         SolverAlgorithm::DCFR | SolverAlgorithm::DCFRPlus => {
             // Standard: R̂^t = [R^t]^+ (positive projection only)
             cum_regrets.iter().map(|&r| r.max(0.0)).collect()
         }
     };

     // Compute strategy from explicit regrets
     let strategy = regret_matching_from_values(&explicit_regrets, num_actions);

     // Recurse to children and collect CFVs...
     let child_cfvalues = /* recursive calls */;

     // Compute expected CFV
     let expected_cfv = compute_expected_value(&strategy, &child_cfvalues, num_actions);

     // COMPUTE INSTANTANEOUS REGRETS (r^t)
     let instantaneous_regrets: Vec<f32> = child_cfvalues.iter()
         .map(|&cfv| cfv - expected_cfv)
         .collect();

     // UPDATE CUMULATIVE REGRETS (R^{t+1}) based on algorithm
     match params.algorithm {
         SolverAlgorithm::APDCFRPlus => {
             // APDCFR+: R^{t+1} = [R^t + λt^β/(κ+t^β) * r^t]^+
             let t = iteration as f32;
             let t_pow_beta = t.powf(params.dcfr_beta);
             let dcfr_weight = params.lambda * t_pow_beta / (params.kappa + t_pow_beta);

             update_regrets_apdcfr(node, &instantaneous_regrets, dcfr_weight, params.beta_t);
         }
         SolverAlgorithm::APCFRPlus | SolverAlgorithm::SAPCFRPlus => {
             // APCFR+/SAPCFR+: R^{t+1} = [R^t + r^t]^+
             update_regrets_simple(node, &instantaneous_regrets);
         }
         SolverAlgorithm::DCFR | SolverAlgorithm::DCFRPlus => {
             // DCFR: R^{t+1} = [α_t * R^t_pos + β_t * R^t_neg + r^t]^+
             update_regrets_dcfr(node, &instantaneous_regrets, params);
         }
     }

     // STORE INSTANTANEOUS REGRETS for next iteration (for prediction)
     if matches!(params.algorithm,
         SolverAlgorithm::APCFRPlus | SolverAlgorithm::SAPCFRPlus | SolverAlgorithm::APDCFRPlus
     ) {
         // Compute ||r^t - r^{t-1}||² for alpha tracking
         if let Some(tracker) = alpha_tracker {
             let regret_diff_sq = if let Some(prev_r) = prev_regrets.get(&node_index) {
                 instantaneous_regrets.iter()
                     .zip(prev_r.iter())
                     .map(|(r, r_prev)| (r - r_prev).powi(2))
                     .sum::<f32>()
             } else {
                 instantaneous_regrets.iter().map(|r| r.powi(2)).sum()
             };

             // Compute ||R^{t+1} - R^t||²
             let new_cum_regrets = node.regrets();
             let cum_regret_diff_sq: f32 = new_cum_regrets.iter()
                 .zip(cum_regrets.iter())
                 .map(|(r_new, r_old)| (r_new - r_old).powi(2))
                 .sum();

             tracker.update(node_index, regret_diff_sq, cum_regret_diff_sq);
         }

         prev_regrets.insert(node_index, instantaneous_regrets);
     }

     // UPDATE CUMULATIVE STRATEGY
     update_strategy(node, &strategy, params.gamma_t);

     return expected_cfv;

     C. Nuove funzioni helper per regret update

     fn update_regrets_simple(node: &MutexLike<PostFlopNode>, instantaneous_regrets: &[f32]) {
         // R^{t+1} = [R^t + r^t]^+
         let mut cum_regrets = node.regrets_mut();
         for (cum_r, &inst_r) in cum_regrets.iter_mut().zip(instantaneous_regrets) {
             *cum_r = (*cum_r + inst_r).max(0.0);
         }
     }

     fn update_regrets_apdcfr(
         node: &MutexLike<PostFlopNode>,
         instantaneous_regrets: &[f32],
         dcfr_weight: f32,
         beta_t: f32,
     ) {
         // R^{t+1} = [R^t + λt^β/(κ+t^β) * r^t]^+
         // With separate treatment for positive/negative (via beta_t if needed)
         let mut cum_regrets = node.regrets_mut();
         for (cum_r, &inst_r) in cum_regrets.iter_mut().zip(instantaneous_regrets) {
             let weighted_inst = dcfr_weight * inst_r;
             *cum_r = (*cum_r + weighted_inst).max(0.0);
         }
     }

     fn update_regrets_dcfr(
         node: &MutexLike<PostFlopNode>,
         instantaneous_regrets: &[f32],
         params: &AlgorithmParams,
     ) {
         // R^{t+1} = [α_t * R^t_pos + β_t * R^t_neg + r^t]^+
         let mut cum_regrets = node.regrets_mut();
         for (cum_r, &inst_r) in cum_regrets.iter_mut().zip(instantaneous_regrets) {
             let discount = if *cum_r >= 0.0 { params.alpha_t } else { params.beta_t };
             *cum_r = (discount * *cum_r + inst_r).max(0.0);
         }
     }

     fn regret_matching_from_values(regrets: &[f32], num_actions: usize) -> Vec<f32> {
         // Same as existing regret_matching but takes pre-computed regret values
         let row_size = regrets.len() / num_actions;
         let mut strategy = vec![0.0; regrets.len()];

         // Sum positive regrets per hand
         let denom = sum_by_rows(regrets, num_actions);

         // Normalize
         for (hand_idx, &sum) in denom.iter().enumerate() {
             for action_idx in 0..num_actions {
                 let idx = action_idx * row_size + hand_idx;
                 strategy[idx] = if sum > 0.0 {
                     regrets[idx] / sum
                 } else {
                     1.0 / num_actions as f32
                 };
             }
         }

         strategy
     }

     2. examples/solve_from_config.rs

     Modifiche a SolverSettings Struct

     #[derive(Debug, Deserialize)]
     struct SolverSettings {
         // Existing fields...
         max_iterations: usize,
         target_exploitability_pct: f32,
         quantization: Option<String>,
         lazy_normalization: bool,
         lazy_normalization_freq: u32,
         log_encoding: bool,
         strategy_bits: u8,

         // NEW FIELDS
         #[serde(default)]
         algorithm: String,  // "dcfr", "dcfr+", "apcfr+", "sapcfr+", "apdcfr+"

         #[serde(default)]
         asymmetry_mode: String,  // "auto", "fixed"

         #[serde(default = "default_asymmetry_alpha")]
         asymmetry_alpha: f32,  // Used when asymmetry_mode = "fixed", default 2.0
     }

     fn default_asymmetry_alpha() -> f32 { 2.0 }

     impl SolverSettings {
         fn get_algorithm(&self) -> Result<SolverAlgorithm, String> {
             if self.algorithm.is_empty() {
                 return Ok(SolverAlgorithm::DCFR);  // Default
             }
             self.algorithm.parse()
         }

         fn get_asymmetry_mode(&self) -> AsymmetryMode {
             match self.asymmetry_mode.to_lowercase().as_str() {
                 "fixed" => AsymmetryMode::Fixed(self.asymmetry_alpha),
                 _ => AsymmetryMode::AutoLearning,  // Default
             }
         }
     }

     Modifiche alla chiamata solve()

     // In main() function
     let algorithm = config.solver.get_algorithm()?;
     let asymmetry_mode = config.solver.get_asymmetry_mode();

     let final_exploitability = solve(
         &mut game,
         config.solver.max_iterations as u32,
         config.solver.target_exploitability_pct / 100.0,
         true,  // print_progress
         algorithm,      // NEW
         asymmetry_mode, // NEW
     );

     3. Config TOML Schema (Esempi)

     config.toml - Esempio con APDCFR+ auto-learning

     [solver]
     max_iterations = 5000
     target_exploitability_pct = 0.1
     algorithm = "apdcfr+"  # NEW: dcfr, dcfr+, apcfr+, sapcfr+, apdcfr+
     asymmetry_mode = "auto"  # NEW: auto, fixed
     # asymmetry_alpha = 2.0  # Only used if asymmetry_mode = "fixed"

     config_sapcfr.toml - Esempio con SAPCFR+ (α=2 fisso)

     [solver]
     max_iterations = 5000
     target_exploitability_pct = 0.1
     algorithm = "sapcfr+"
     asymmetry_mode = "fixed"
     asymmetry_alpha = 2.0

     config_dcfr_legacy.toml - Compatibilità con DCFR attuale

     [solver]
     max_iterations = 5000
     target_exploitability_pct = 0.1
     # algorithm non specificato = default DCFR

     4. src/lib.rs

     // Update documentation
     //! Postflop GTO solver using advanced CFR algorithms:
     //! - DCFR (Discounted CFR)
     //! - DCFR+ (Discounted CFR+)
     //! - APCFR+ (Asymmetric Predictive CFR+)
     //! - SAPCFR+ (Simple Asymmetric Predictive CFR+)
     //! - APDCFR+ (Asymmetric Predictive Discounted CFR+)

     // Re-export new types
     pub use solver::{SolverAlgorithm, AsymmetryMode};

     ✅ Checklist Implementazione

     Fase 1: Strutture Base

     - Creare SolverAlgorithm enum in src/solver.rs
     - Creare AsymmetryMode enum in src/solver.rs
     - Estendere DiscountParams a AlgorithmParams
     - Implementare FromStr per SolverAlgorithm
     - Creare AlphaTracker struct per auto-learning

     Fase 2: Storage Prediction

     - Aggiungere PrevRegretStorage type alias
     - Implementare logica storage/retrieval prev_regrets
     - Implementare AlphaTracker::update()
     - Implementare AlphaTracker::compute_alpha()

     Fase 3: Core Solver Logic

     - Modificare signature solve() (aggiungere algorithm, asymmetry_mode)
     - Modificare signature solve_step() (aggiungere prev_regrets, alpha_tracker)
     - Modificare signature solve_recursive() (aggiungere prev_regrets, alpha_tracker)
     - Implementare calcolo explicit regrets (R̂^t) con prediction
     - Implementare update_regrets_simple() per APCFR+/SAPCFR+
     - Implementare update_regrets_apdcfr() per APDCFR+
     - Implementare update_regrets_dcfr() per DCFR/DCFR+
     - Implementare regret_matching_from_values()
     - Aggiungere tracking ||r^t - r^{t-1}||² e ||R^{t+1} - R^t||²
     - Salvare instantaneous regrets in prev_regrets

     Fase 4: Configurazione

     - Aggiungere campi algorithm, asymmetry_mode, asymmetry_alpha a SolverSettings
     - Implementare SolverSettings::get_algorithm()
     - Implementare SolverSettings::get_asymmetry_mode()
     - Aggiornare chiamata solve() in examples/solve_from_config.rs
     - Creare config esempio per ogni algoritmo

     Fase 5: Testing

     - Test DCFR (regressione - deve dare stessi risultati di prima)
     - Test DCFR+ (confronto con DCFR)
     - Test APCFR+ con auto-learning
     - Test SAPCFR+ con α=2 fisso
     - Test APDCFR+ con auto-learning
     - Test APDCFR+ con α fisso
     - Test convergenza su Leduc Poker
     - Test convergenza su HUNL Subgames
     - Test compatibilità quantization modes (32-bit, 16-bit)
     - Test compatibilità mixed precision (8-bit strategy)

     Fase 6: Documentazione

     - Aggiornare README con nuovi algoritmi
     - Aggiungere esempi config per ogni algoritmo
     - Documentare parametri TOML
     - Aggiungere commenti al codice
     - Creare guide su quando usare quale algoritmo

     🧪 Strategia di Testing

     Test Unitari

     #[cfg(test)]
     mod tests {
         use super::*;

         #[test]
         fn test_algorithm_from_str() {
             assert_eq!("dcfr".parse::<SolverAlgorithm>().unwrap(), SolverAlgorithm::DCFR);
             assert_eq!("apdcfr+".parse::<SolverAlgorithm>().unwrap(), SolverAlgorithm::APDCFRPlus);
         }

         #[test]
         fn test_alpha_tracker_compute() {
             let mut tracker = AlphaTracker::new();
             tracker.update(0, 10.0, 2.0);  // ||r^t - r^{t-1}||² = 10, ||R^{t+1} - R^t||² = 2
             let alpha = tracker.compute_alpha(0, 9.0);
             assert_eq!(alpha, (10.0 / 2.0).sqrt());  // sqrt(5) ≈ 2.236
         }

         #[test]
         fn test_dcfr_backward_compatibility() {
             // Verificare che DCFR con nuova implementazione dia stessi risultati
             // di versione precedente
         }
     }

     Test Integrazione

     1. Confronto convergenza algoritmi
       - Eseguire tutti gli algoritmi su Leduc Poker per 1000 iterazioni
       - Verificare che APDCFR+ converga più veloce di DCFR
       - Verificare che SAPCFR+ dia risultati simili a APCFR+ con α=2
     2. Regressione DCFR
       - Eseguire DCFR su hand 7438 node 06 (già testato)
       - Confrontare exploitability finale con baseline
       - Verificare che sia identico (entro tolleranza numerica)
     3. Performance test
       - Misurare tempo di esecuzione per ogni algoritmo
       - APDCFR+ con auto-learning dovrebbe essere ~10-20% più lento (HashMap overhead)
       - SAPCFR+ dovrebbe essere velocità simile a DCFR (no auto-learning)

     Test Configurazione

     # Test DCFR (default)
     cargo run --release --example solve_from_config -- config.toml

     # Test APDCFR+ auto
     cargo run --release --example solve_from_config -- config_apdcfr_auto.toml

     # Test SAPCFR+ fixed
     cargo run --release --example solve_from_config -- config_sapcfr.toml

     ⚠️ Considerazioni Backward Compatibility

     1. Default algorithm = DCFR
       - Se algorithm non specificato in TOML → DCFR
       - Garantisce funzionamento config esistenti
     2. Signature solve() con defaults
     // Helper function per backward compatibility
     pub fn solve_default<T: Game>(
         game: &mut T,
         max_iterations: u32,
         target_exploitability: f32,
         print_progress: bool,
     ) -> f32 {
         solve(
             game,
             max_iterations,
             target_exploitability,
             print_progress,
             SolverAlgorithm::DCFR,  // Default
             AsymmetryMode::AutoLearning,
         )
     }
     3. No breaking changes a PostFlopNode
       - Storage prediction in HashMap temporanei
       - No nuovi field a struct
       - Compatibilità con serialization esistente
     4. Quantization support
       - Tutti algoritmi funzionano con 32-bit e 16-bit
       - Mixed precision (8-bit strategy) supportato
       - Log encoding compatibile

     📊 Performance Attese

     Basandosi sui paper:

     | Algoritmo | Convergenza vs DCFR | Overhead Computazionale |
     |-----------|---------------------|-------------------------|
     | DCFR      | Baseline            | Baseline                |
     | DCFR+     | 1.0-1.2x            | ~0%                     |
     | APCFR+    | 2-3x in poker       | ~10% (prediction)       |
     | SAPCFR+   | 2-3x in poker       | ~5% (fixed α)           |
     | APDCFR+   | 3-5x in poker       | ~15% (DCFR+pred+alpha)  |

     🎯 Priorità Implementazione

     Must Have (MVP)

     1. ✅ SolverAlgorithm enum
     2. ✅ APDCFR+ con α fisso (SAPDCFR+)
     3. ✅ Config TOML support
     4. ✅ Backward compatibility DCFR

     Should Have

     1. ✅ APDCFR+ con auto-learning
     2. ✅ APCFR+/SAPCFR+ (senza DCFR)
     3. ✅ DCFR+
     4. ✅ Alpha tracking completo

     Nice to Have

     1. Performance benchmarks dettagliati
     2. Visualizzazione dinamica α durante solve
     3. Export metriche (||r^t - r^{t-1}||², α_t, ecc.)
     4. Adaptive alpha_max per auto-learning

     🔍 Note Implementative

     Gestione Node Index

     Per identificare univocamente un nodo (per HashMap):
     // In PostFlopGame: ogni nodo ha offset nell'arena
     let node_index = node.as_ptr() as usize - game.nodes[0].as_ptr() as usize;

     Ottimizzazioni Future

     1. Pre-allocare prev_regrets con capacità stimata
     2. Usare FxHashMap invece di HashMap standard (già importato)
     3. Clear prev_regrets dopo ogni iterazione completa (memory saving)
     4. Parallelizzazione alpha tracking (attualmente HashMap non thread-safe)

     Compatibilità Quantization

     Per funzionare con 16-bit:
     // Quando node.is_compressed():
     let cum_regrets_i16 = node.regrets_compressed();
     let cum_regrets_f32: Vec<f32> = decode_regrets(cum_regrets_i16, node.scale2);
     // ... logica con f32 ...
     let new_cum_regrets_i16 = encode_regrets(&cum_regrets_f32, node.scale2);

     ---
     📋 Summary

     Questo piano implementa 4 nuovi algoritmi CFR (DCFR+, APCFR+, SAPCFR+, APDCFR+) mantenendo compatibilità con DCFR esistente. La soluzione usa HashMap temporanei 
     per prediction storage, evitando modifiche a PostFlopNode. La configurazione via TOML permette selezione algoritmo user-friendly. Testing completo garantisce 
     correttezza e performance.

     Files modificati:
     - src/solver.rs (core logic)
     - examples/solve_from_config.rs (config parsing)
     - src/lib.rs (exports)

     Files aggiunti:
     - Config examples per ogni algoritmo
