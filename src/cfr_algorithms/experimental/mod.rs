//! Algoritmi CFR sperimentali
//!
//! Questi algoritmi sono in fase sperimentale e meno supportati.
//! Per uso in produzione, preferire DCFR o DCFR+.
//!
//! ## Algoritmi Sperimentali
//!
//! - **PDCFR+**: Predictive Discounted CFR Plus - richiede storage4 per i regrets previsti
//! - **SAPCFR+**: Sampling Average Predictive CFR Plus - richiede storage4 per i regrets istantanei precedenti

mod pdcfr_plus;
mod sapcfr_plus;

pub use pdcfr_plus::PdcfrPlusAlgorithm;
pub use sapcfr_plus::SapcfrPlusAlgorithm;
