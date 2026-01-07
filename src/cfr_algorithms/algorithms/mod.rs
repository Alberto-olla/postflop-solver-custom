/// Implementazioni concrete degli algoritmi CFR principali (raccomandati per produzione)
pub mod dcfr;
pub mod dcfr_plus;

// Re-export per comodità
pub use dcfr::DcfrAlgorithm;
pub use dcfr_plus::DcfrPlusAlgorithm;
