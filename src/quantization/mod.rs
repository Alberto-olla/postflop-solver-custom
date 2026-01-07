//! Sistema modulare di quantization per storage efficiente dei dati
//!
//! Questo modulo fornisce:
//! - `QuantizationMode`: Enum per definire il tipo di quantizzazione (Float32, Int16, Int8, Int4)
//! - Trait-based abstraction per eliminare duplicazione tra diversi bit-width
//! - Funzioni di encoding/decoding ottimizzate per ogni tipo
//!
//! # Esempio
//! ```ignore
//! use postflop_solver::quantization::{QuantizationMode, QuantizationType};
//!
//! let mode = QuantizationMode::Int16;
//! let data = vec![1.5, -2.3, 0.8];
//! // ... encoding/decoding logic
//! ```

pub mod encoding;
pub mod mode;
pub mod traits;
pub mod types;

// Re-export principali
pub use mode::{QuantizationConfig, QuantizationMode};
pub use traits::QuantizationType;
// types viene esportato per uso interno, non in public API
