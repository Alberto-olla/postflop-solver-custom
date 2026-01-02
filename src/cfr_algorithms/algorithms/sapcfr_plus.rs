use super::super::traits::*;

/// Sampling Average Predictive CFR Plus Algorithm
///
/// SAPCFR+ è una variante avanzata che:
/// - Usa Implicit Regrets (simple sum, no discounting: alpha=1.0)
/// - Calcola Explicit Regrets = Implicit + 1/3 * Previous Instantaneous
/// - Richiede storage4 per memorizzare previous instantaneous regrets
/// - Usa linear averaging per strategy (gamma discounting)
#[derive(Debug, Clone, Copy, Default)]
pub struct SapcfrPlusAlgorithm;

impl CfrAlgorithmTrait for SapcfrPlusAlgorithm {
    fn name(&self) -> &'static str {
        "SAPCFR+"
    }

    fn compute_discounts(&self, iteration: u32) -> DiscountParams {
        // SAPCFR+:
        // - Implicit Regret: Simple Sum (alpha=1.0)
        // - Strategy: Linear Avg (gamma=1.0) -> requires discounting previous sum by t/(t+1)
        // - Explicit Regret: Implicit + 1/3 * Prev (Handled separately in strategy computation)

        let t = iteration as f64;
        let gamma_t = if iteration == 0 {
            0.0
        } else {
            (t / (t + 1.0)).powi(3)
        } as f32;

        DiscountParams {
            alpha_t: 1.0, // No discount for implicit regrets
            beta_t: 1.0,  // Not used (no sign-based discounting)
            gamma_t,
        }
    }

    fn requires_storage4(&self) -> bool {
        true // SAPCFR+ necessita di storage4 per previous instantaneous regrets
    }

    fn clone_box(&self) -> Box<dyn CfrAlgorithmTrait> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sapcfr_plus_discount_iteration_0() {
        let algo = SapcfrPlusAlgorithm;
        let params = algo.compute_discounts(0);
        assert_eq!(params.alpha_t, 1.0);
        assert_eq!(params.beta_t, 1.0);
        assert_eq!(params.gamma_t, 0.0);
    }

    #[test]
    fn test_sapcfr_plus_discount_iteration_1() {
        let algo = SapcfrPlusAlgorithm;
        let params = algo.compute_discounts(1);
        assert_eq!(params.alpha_t, 1.0);
        assert_eq!(params.beta_t, 1.0);

        // gamma_t = (1 / 2)^3 = 1/8 = 0.125
        let expected_gamma = (0.5_f64.powi(3)) as f32;
        assert!((params.gamma_t - expected_gamma).abs() < 0.0001);
    }

    #[test]
    fn test_sapcfr_plus_discount_iteration_10() {
        let algo = SapcfrPlusAlgorithm;
        let params = algo.compute_discounts(10);
        assert_eq!(params.alpha_t, 1.0);
        assert_eq!(params.beta_t, 1.0);

        // gamma_t = (10/11)^3
        let expected_gamma = ((10.0 / 11.0_f64).powi(3)) as f32;
        assert!((params.gamma_t - expected_gamma).abs() < 0.0001);
    }

    #[test]
    fn test_sapcfr_plus_requires_storage4() {
        let algo = SapcfrPlusAlgorithm;
        assert!(algo.requires_storage4());
    }

    #[test]
    fn test_sapcfr_plus_name() {
        let algo = SapcfrPlusAlgorithm;
        assert_eq!(algo.name(), "SAPCFR+");
    }

    #[test]
    fn test_alpha_always_one() {
        let algo = SapcfrPlusAlgorithm;

        for i in 0..1000 {
            let params = algo.compute_discounts(i);
            assert_eq!(params.alpha_t, 1.0,
                "Alpha should always be 1.0 for SAPCFR+ (iteration {})", i);
            assert_eq!(params.beta_t, 1.0,
                "Beta should always be 1.0 for SAPCFR+ (iteration {})", i);
        }
    }

    #[test]
    fn test_gamma_increases_with_iteration() {
        let algo = SapcfrPlusAlgorithm;
        let mut prev_gamma = 0.0;

        for i in 1..100 {
            let params = algo.compute_discounts(i);
            assert!(params.gamma_t >= prev_gamma,
                "Gamma should increase monotonically (iteration {})", i);
            prev_gamma = params.gamma_t;
        }
    }

    #[test]
    fn test_gamma_bounded() {
        let algo = SapcfrPlusAlgorithm;

        for i in 0..10000 {
            let params = algo.compute_discounts(i);
            assert!(params.gamma_t >= 0.0 && params.gamma_t < 1.0,
                "Gamma should be in [0, 1) (iteration {})", i);
        }
    }

    #[test]
    fn test_gamma_approaches_one() {
        let algo = SapcfrPlusAlgorithm;

        // Per iterazioni molto grandi, gamma_t -> 1
        let params = algo.compute_discounts(100000);
        assert!(params.gamma_t > 0.99,
            "Gamma should approach 1.0 for large iterations");
    }
}
