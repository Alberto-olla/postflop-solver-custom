use super::super::traits::*;

/// Predictive Discounted CFR Plus Algorithm
///
/// PDCFR+ combina:
/// - Discounting differenziato di DCFR (alpha_t per positivi, beta_t per negativi)
/// - Meccanismo predittivo che richiede storage4 per predicted regrets
/// - Strategy calcolata da predicted regrets invece che da cumulative regrets
///
/// Formule chiave:
/// - Cumulative: Rt[j] = [Rt-1[j] * ((t-1)^α / ((t-1)^α+1)) + rt[j]]+
/// - Predicted: R̃t+1[j] = [Rt[j] * (t^α / (t^α+1)) + vt+1[j]]+
/// - Strategy: derivata da R̃t (non da Rt)
#[derive(Debug, Clone, Copy, Default)]
pub struct PdcfrPlusAlgorithm;

impl CfrAlgorithmTrait for PdcfrPlusAlgorithm {
    fn name(&self) -> &'static str {
        "PDCFR+"
    }

    fn compute_discounts(&self, iteration: u32) -> DiscountParams {
        // PDCFR+ usa gli stessi discount parameters di DCFR:
        // - alpha_t per regrets positivi: (t-1)^1.5 / ((t-1)^1.5 + 1)
        // - beta_t = 0.5 per regrets negativi (costante)
        // - gamma_t per strategy averaging con power-of-4 reset heuristic

        // Strategy reset heuristic: 0, 1, 4, 16, 64, 256, ...
        let nearest_lower_power_of_4 = match iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };

        // Use (t-1) for alpha (come DCFR)
        let t_alpha = (iteration as i32 - 1).max(0) as f64;
        let t_gamma = (iteration - nearest_lower_power_of_4) as f64;

        // Regret discount: (t-1)^1.5 / ((t-1)^1.5 + 1)
        let pow_alpha = t_alpha * t_alpha.sqrt(); // (t-1)^1.5
        let alpha_t = (pow_alpha / (pow_alpha + 1.0)) as f32;

        // Strategy discount: (t_gamma / (t_gamma + 1))^3
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);

        DiscountParams {
            alpha_t,
            beta_t: 0.5, // Constant discount for negative regrets (come DCFR)
            gamma_t: pow_gamma as f32,
        }
    }

    fn requires_storage4(&self) -> bool {
        true // PDCFR+ necessita di storage4 per predicted regrets
    }

    fn clone_box(&self) -> Box<dyn CfrAlgorithmTrait> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdcfr_plus_discount_iteration_0() {
        let algo = PdcfrPlusAlgorithm;
        let params = algo.compute_discounts(0);
        assert_eq!(params.alpha_t, 0.0);
        assert_eq!(params.beta_t, 0.5);
        assert_eq!(params.gamma_t, 0.0);
    }

    #[test]
    fn test_pdcfr_plus_discount_iteration_1() {
        let algo = PdcfrPlusAlgorithm;
        let params = algo.compute_discounts(1);
        assert_eq!(params.alpha_t, 0.0); // (0)^1.5 / (0 + 1) = 0
        assert_eq!(params.beta_t, 0.5);
    }

    #[test]
    fn test_pdcfr_plus_discount_iteration_10() {
        let algo = PdcfrPlusAlgorithm;
        let params = algo.compute_discounts(10);

        // t-1 = 9, so (9^1.5) / (9^1.5 + 1)
        let expected_alpha = (27.0_f64 / 28.0) as f32;
        assert!((params.alpha_t - expected_alpha).abs() < 0.001);
        assert_eq!(params.beta_t, 0.5);
    }

    #[test]
    fn test_pdcfr_plus_requires_storage4() {
        let algo = PdcfrPlusAlgorithm;
        assert!(algo.requires_storage4());
    }

    #[test]
    fn test_pdcfr_plus_name() {
        let algo = PdcfrPlusAlgorithm;
        assert_eq!(algo.name(), "PDCFR+");
    }

    #[test]
    fn test_alpha_increases_with_iteration() {
        let algo = PdcfrPlusAlgorithm;
        let mut prev_alpha = 0.0;

        for i in 1..100 {
            let params = algo.compute_discounts(i);
            assert!(params.alpha_t >= prev_alpha,
                "Alpha should increase monotonically (iteration {})", i);
            prev_alpha = params.alpha_t;
        }
    }

    #[test]
    fn test_alpha_bounded() {
        let algo = PdcfrPlusAlgorithm;

        for i in 0..10000 {
            let params = algo.compute_discounts(i);
            assert!(params.alpha_t >= 0.0 && params.alpha_t < 1.0,
                "Alpha should be in [0, 1) (iteration {})", i);
        }
    }

    #[test]
    fn test_discounts_match_dcfr() {
        // PDCFR+ dovrebbe avere gli stessi discount parameters di DCFR
        use super::super::DcfrAlgorithm;

        let pdcfr_plus = PdcfrPlusAlgorithm;
        let dcfr = DcfrAlgorithm;

        for i in 0..100 {
            let params_pdcfr = pdcfr_plus.compute_discounts(i);
            let params_dcfr = dcfr.compute_discounts(i);

            assert_eq!(params_pdcfr.alpha_t, params_dcfr.alpha_t,
                "Alpha should match DCFR (iteration {})", i);
            assert_eq!(params_pdcfr.beta_t, params_dcfr.beta_t,
                "Beta should match DCFR (iteration {})", i);
            assert_eq!(params_pdcfr.gamma_t, params_dcfr.gamma_t,
                "Gamma should match DCFR (iteration {})", i);
        }
    }
}
