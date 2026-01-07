/// Trait che definisce le operazioni di quantization per un tipo specifico
///
/// Questo trait permette di astrarre le operazioni di encoding/decoding
/// per diversi bit-width (32, 16, 8, 4 bit), eliminando la duplicazione
/// di codice nel solver.
///
/// # Implementazioni
/// - `Float32Quant`: 32-bit floating point (no compression)
/// - `Int16Quant`: 16-bit signed integer
/// - `Int16LogQuant`: 16-bit with logarithmic encoding
/// - `Int8Quant`: 8-bit signed integer
/// - `Int4PackedQuant`: 4-bit packed (2 values per byte)
///
/// # Esempio
/// ```ignore
/// use postflop_solver::quantization::{QuantizationType, Int16Quant};
///
/// let data = vec![1.5, -2.3, 0.8];
/// let mut storage = vec![0i16; data.len()];
/// let scale = Int16Quant::encode(&mut storage, &data, 0);
///
/// let mut decoded = vec![0.0; data.len()];
/// Int16Quant::decode_slice(&storage, scale, &mut decoded);
/// ```
pub trait QuantizationType: Send + Sync + 'static {
    /// Tipo di storage utilizzato (u8, i8, u16, i16, f32)
    type Storage: Copy + Send + Sync;

    /// Bit width (32, 16, 8, 4)
    const BITS: u8;

    /// Valore massimo per scaling (es. i16::MAX per Int16)
    const MAX_VALUE: f32;

    /// Se il tipo è signed
    const IS_SIGNED: bool;

    /// Encode f32 slice to quantized storage.
    ///
    /// # Arguments
    /// - `dst`: Destination storage buffer
    /// - `src`: Source f32 data
    /// - `seed`: Random seed for stochastic rounding (if applicable)
    ///
    /// # Returns
    /// Scale factor used for quantization
    fn encode(dst: &mut [Self::Storage], src: &[f32], seed: u32) -> f32;

    /// Decode single element from storage.
    ///
    /// # Arguments
    /// - `value`: Quantized value
    /// - `scale`: Scale factor from encoding
    ///
    /// # Returns
    /// Decoded f32 value
    fn decode_element(value: Self::Storage, scale: f32) -> f32;

    /// Decode entire slice from storage.
    ///
    /// # Arguments
    /// - `src`: Source quantized storage
    /// - `scale`: Scale factor from encoding
    /// - `dst`: Destination f32 buffer
    fn decode_slice(src: &[Self::Storage], scale: f32, dst: &mut [f32]);

    /// Regret matching ottimizzato per questo tipo.
    ///
    /// Implementazione specializzata del regret matching che opera
    /// direttamente sui dati quantizzati quando possibile, evitando
    /// decode/encode superflui.
    ///
    /// # Arguments
    /// - `data`: Quantized regrets storage
    /// - `scale`: Scale factor
    /// - `num_actions`: Number of actions per hand
    ///
    /// # Returns
    /// Strategy vector (f32)
    fn regret_matching(data: &[Self::Storage], scale: f32, num_actions: usize) -> Vec<f32>;
}

/// Wrapper per slice quantizzata (immutable)
///
/// Fornisce operazioni di alto livello su dati quantizzati.
pub struct QuantizedSlice<'a, Q: QuantizationType> {
    pub data: &'a [Q::Storage],
    pub scale: f32,
}

impl<'a, Q: QuantizationType> QuantizedSlice<'a, Q> {
    pub fn new(data: &'a [Q::Storage], scale: f32) -> Self {
        Self { data, scale }
    }

    /// Decode to f32 vector
    pub fn to_f32(&self) -> Vec<f32> {
        let mut result = vec![0.0; self.data.len()];
        Q::decode_slice(self.data, self.scale, &mut result);
        result
    }

    /// Perform regret matching
    pub fn regret_matching(&self, num_actions: usize) -> Vec<f32> {
        Q::regret_matching(self.data, self.scale, num_actions)
    }
}

/// Wrapper per slice quantizzata (mutable)
///
/// Permette encoding di nuovi dati.
pub struct QuantizedSliceMut<'a, Q: QuantizationType> {
    pub data: &'a mut [Q::Storage],
    pub scale: &'a mut f32,
}

impl<'a, Q: QuantizationType> QuantizedSliceMut<'a, Q> {
    pub fn new(data: &'a mut [Q::Storage], scale: &'a mut f32) -> Self {
        Self { data, scale }
    }

    /// Encode from f32 slice
    ///
    /// # Returns
    /// The scale factor used
    pub fn from_f32(&mut self, src: &[f32], seed: u32) -> f32 {
        let new_scale = Q::encode(self.data, src, seed);
        *self.scale = new_scale;
        new_scale
    }
}
