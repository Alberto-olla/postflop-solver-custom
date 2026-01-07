pub mod algorithm;
pub mod algorithms;
pub mod experimental;
/// Nuova architettura trait-based per algoritmi CFR
///
/// Questo modulo implementa un sistema modulare per gli algoritmi CFR che:
/// - Separa la logica algoritmica in implementazioni indipendenti
/// - Usa trait per definire interfacce comuni
/// - Supporta zero-cost abstraction tramite enum dispatch
/// - Facilita testing e estensibilità
///
/// # Struttura
/// - `traits`: Definizione trait CfrAlgorithmTrait e DiscountParams
/// - `algorithms`: Implementazioni concrete principali (DCFR, DCFR+) - raccomandati per produzione
/// - `experimental`: Algoritmi sperimentali (PDCFR+, SAPCFR+) - meno supportati
/// - `algorithm`: Enum wrapper per dispatch statico
///
/// # Algoritmi Raccomandati
/// - **DCFR**: Original Discounted CFR
/// - **DCFR+**: DCFR con regret floor clipping (raccomandato)
///
/// # Algoritmi Sperimentali
/// Per algoritmi sperimentali (PDCFR+, SAPCFR+), vedi il modulo `experimental`.
///
/// # Esempio
/// ```ignore
/// use crate::solver::{CfrAlgorithmEnum, CfrAlgorithmTrait};
///
/// let algo = CfrAlgorithmEnum::from_name("dcfr+").unwrap();
/// let params = algo.compute_discounts(10);
/// println!("Alpha: {}, Gamma: {}", params.alpha_t, params.gamma_t);
/// ```
pub mod traits;

// Re-exports pubblici
pub use algorithm::CfrAlgorithmEnum;
pub use traits::{CfrAlgorithmTrait, DiscountParams};

// Algoritmi principali (raccomandati)
pub use algorithms::{DcfrAlgorithm, DcfrPlusAlgorithm};

// Nota: Gli algoritmi sperimentali (PdcfrPlusAlgorithm, SapcfrPlusAlgorithm)
// sono disponibili nel modulo `experimental` e richiedono import esplicito
