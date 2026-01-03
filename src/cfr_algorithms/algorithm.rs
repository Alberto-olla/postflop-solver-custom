use super::algorithms::*;
use super::traits::*;

/// Enum wrapper per dispatch statico (zero-cost abstraction)
///
/// Questo enum permette di usare gli algoritmi CFR con dispatch statico,
/// evitando l'overhead di virtual calls mantenendo i benefici dell'astrazione trait-based.
#[derive(Debug, Clone)]
pub enum CfrAlgorithmEnum {
    DCFR(DcfrAlgorithm),
    DCFRPlus(DcfrPlusAlgorithm),
    PDCFRPlus(PdcfrPlusAlgorithm),
    SAPCFRPlus(SapcfrPlusAlgorithm),
}

impl CfrAlgorithmEnum {
    /// Factory method per nome
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "dcfr" => Some(Self::DCFR(DcfrAlgorithm)),
            "dcfr+" | "dcfrplus" => Some(Self::DCFRPlus(DcfrPlusAlgorithm)),
            "pdcfr+" | "pdcfrplus" => Some(Self::PDCFRPlus(PdcfrPlusAlgorithm)),
            "sapcfr+" | "sapcfrplus" => Some(Self::SAPCFRPlus(SapcfrPlusAlgorithm)),
            _ => None,
        }
    }

    /// Dispatch ai trait methods (inline per zero overhead)
    #[inline]
    pub fn compute_discounts(&self, iteration: u32) -> DiscountParams {
        match self {
            Self::DCFR(algo) => algo.compute_discounts(iteration),
            Self::DCFRPlus(algo) => algo.compute_discounts(iteration),
            Self::PDCFRPlus(algo) => algo.compute_discounts(iteration),
            Self::SAPCFRPlus(algo) => algo.compute_discounts(iteration),
        }
    }

    #[inline]
    pub fn requires_storage4(&self) -> bool {
        match self {
            Self::DCFR(algo) => algo.requires_storage4(),
            Self::DCFRPlus(algo) => algo.requires_storage4(),
            Self::PDCFRPlus(algo) => algo.requires_storage4(),
            Self::SAPCFRPlus(algo) => algo.requires_storage4(),
        }
    }

    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Self::DCFR(algo) => algo.name(),
            Self::DCFRPlus(algo) => algo.name(),
            Self::PDCFRPlus(algo) => algo.name(),
            Self::SAPCFRPlus(algo) => algo.name(),
        }
    }
}

impl Default for CfrAlgorithmEnum {
    fn default() -> Self {
        Self::DCFR(DcfrAlgorithm)
    }
}

// Implement CfrAlgorithmTrait per l'enum (delegation)
impl CfrAlgorithmTrait for CfrAlgorithmEnum {
    fn name(&self) -> &'static str {
        self.name()
    }

    #[inline]
    fn compute_discounts(&self, iteration: u32) -> DiscountParams {
        self.compute_discounts(iteration)
    }

    #[inline]
    fn requires_storage4(&self) -> bool {
        self.requires_storage4()
    }

    fn clone_box(&self) -> Box<dyn CfrAlgorithmTrait> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_name() {
        assert!(matches!(
            CfrAlgorithmEnum::from_name("dcfr"),
            Some(CfrAlgorithmEnum::DCFR(_))
        ));
        assert!(matches!(
            CfrAlgorithmEnum::from_name("DCFR"),
            Some(CfrAlgorithmEnum::DCFR(_))
        ));
        assert!(matches!(
            CfrAlgorithmEnum::from_name("dcfr+"),
            Some(CfrAlgorithmEnum::DCFRPlus(_))
        ));
        assert!(matches!(
            CfrAlgorithmEnum::from_name("dcfrplus"),
            Some(CfrAlgorithmEnum::DCFRPlus(_))
        ));
        assert!(matches!(
            CfrAlgorithmEnum::from_name("pdcfr+"),
            Some(CfrAlgorithmEnum::PDCFRPlus(_))
        ));
        assert!(matches!(
            CfrAlgorithmEnum::from_name("pdcfrplus"),
            Some(CfrAlgorithmEnum::PDCFRPlus(_))
        ));
        assert!(matches!(
            CfrAlgorithmEnum::from_name("sapcfr+"),
            Some(CfrAlgorithmEnum::SAPCFRPlus(_))
        ));
        assert!(matches!(
            CfrAlgorithmEnum::from_name("sapcfrplus"),
            Some(CfrAlgorithmEnum::SAPCFRPlus(_))
        ));
        assert!(CfrAlgorithmEnum::from_name("invalid").is_none());
    }

    #[test]
    fn test_default() {
        let algo = CfrAlgorithmEnum::default();
        assert!(matches!(algo, CfrAlgorithmEnum::DCFR(_)));
    }

    #[test]
    fn test_dispatch_name() {
        let dcfr = CfrAlgorithmEnum::DCFR(DcfrAlgorithm);
        let dcfr_plus = CfrAlgorithmEnum::DCFRPlus(DcfrPlusAlgorithm);
        let pdcfr_plus = CfrAlgorithmEnum::PDCFRPlus(PdcfrPlusAlgorithm);
        let sapcfr_plus = CfrAlgorithmEnum::SAPCFRPlus(SapcfrPlusAlgorithm);

        assert_eq!(dcfr.name(), "DCFR");
        assert_eq!(dcfr_plus.name(), "DCFR+");
        assert_eq!(pdcfr_plus.name(), "PDCFR+");
        assert_eq!(sapcfr_plus.name(), "SAPCFR+");
    }

    #[test]
    fn test_dispatch_requires_storage4() {
        let dcfr = CfrAlgorithmEnum::DCFR(DcfrAlgorithm);
        let dcfr_plus = CfrAlgorithmEnum::DCFRPlus(DcfrPlusAlgorithm);
        let pdcfr_plus = CfrAlgorithmEnum::PDCFRPlus(PdcfrPlusAlgorithm);
        let sapcfr_plus = CfrAlgorithmEnum::SAPCFRPlus(SapcfrPlusAlgorithm);

        assert!(!dcfr.requires_storage4());
        assert!(!dcfr_plus.requires_storage4());
        assert!(pdcfr_plus.requires_storage4());
        assert!(sapcfr_plus.requires_storage4());
    }

    #[test]
    fn test_dispatch_compute_discounts() {
        let dcfr = CfrAlgorithmEnum::DCFR(DcfrAlgorithm);
        let params = dcfr.compute_discounts(10);

        // Verifica che il dispatch funzioni
        assert_eq!(params.beta_t, 0.5);
    }

    #[test]
    fn test_trait_implementation() {
        let algo: Box<dyn CfrAlgorithmTrait> = Box::new(CfrAlgorithmEnum::DCFR(DcfrAlgorithm));
        assert_eq!(algo.name(), "DCFR");
        assert!(!algo.requires_storage4());

        let params = algo.compute_discounts(0);
        assert_eq!(params.alpha_t, 0.0);
    }
}
