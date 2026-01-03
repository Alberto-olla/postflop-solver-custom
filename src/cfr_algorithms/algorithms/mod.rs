/// Implementazioni concrete degli algoritmi CFR
pub mod dcfr;
pub mod dcfr_plus;
pub mod pdcfr_plus;
pub mod sapcfr_plus;

// Re-export per comodità
pub use dcfr::DcfrAlgorithm;
pub use dcfr_plus::DcfrPlusAlgorithm;
pub use pdcfr_plus::PdcfrPlusAlgorithm;
pub use sapcfr_plus::SapcfrPlusAlgorithm;
