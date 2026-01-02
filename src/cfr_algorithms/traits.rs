/// Parametri di discounting calcolati per un'iterazione
#[derive(Debug, Clone, Copy)]
pub struct DiscountParams {
    /// Discount factor per regrets positivi
    pub alpha_t: f32,
    /// Discount factor per regrets negativi (usato solo in DCFR)
    pub beta_t: f32,
    /// Discount factor per strategy accumulation
    pub gamma_t: f32,
}

/// Interfaccia comune per tutti gli algoritmi CFR
///
/// Questo trait definisce le operazioni che ogni variante CFR deve implementare:
/// - Calcolo parametri di discount
/// - Update dei regrets (compressed e uncompressed)
/// - Requisiti di storage (storage4 per prev_regrets)
/// - Calcolo della strategy (opzionale override per SAPCFR+ explicit regrets)
pub trait CfrAlgorithmTrait: Send + Sync + std::fmt::Debug {
    /// Nome leggibile dell'algoritmo (per logging/debug)
    fn name(&self) -> &'static str;

    /// Calcola i parametri di discount per l'iterazione corrente
    ///
    /// # Arguments
    /// * `iteration` - Numero iterazione corrente (0-indexed)
    ///
    /// # Returns
    /// Struct con alpha_t, beta_t, gamma_t
    fn compute_discounts(&self, iteration: u32) -> DiscountParams;

    /// Indica se l'algoritmo richiede storage4 (previous instantaneous regrets)
    ///
    /// Default: false (solo SAPCFR+ richiede storage4)
    fn requires_storage4(&self) -> bool {
        false
    }

    /// Clone boxed per dynamic dispatch (quando necessario)
    fn clone_box(&self) -> Box<dyn CfrAlgorithmTrait>;
}

// Supporto per Clone su Box<dyn CfrAlgorithmTrait>
impl Clone for Box<dyn CfrAlgorithmTrait> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
