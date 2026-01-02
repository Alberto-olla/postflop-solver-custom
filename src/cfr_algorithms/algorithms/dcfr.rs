use super::super::traits::*;

/// Discounted CFR Algorithm
///
/// DCFR usa discounting differenziato per regrets positivi/negativi:
/// - alpha_t per regrets positivi (crescita con t^1.5)
/// - beta_t = 0.5 per regrets negativi (costante)
/// - gamma_t per strategy averaging (reset periodico su potenze di 4)
#[derive(Debug, Clone, Copy, Default)]
pub struct DcfrAlgorithm;

impl CfrAlgorithmTrait for DcfrAlgorithm {
    fn name(&self) -> &'static str {
        "DCFR"
    }

    fn compute_discounts(&self, iteration: u32) -> DiscountParams {
        // DCFR: Uses (t-1) for alpha discounting
        // Strategy reset heuristic: 0, 1, 4, 16, 64, 256, ...
        let nearest_lower_power_of_4 = match iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };

        // Use (t-1) for alpha
        let t_alpha = (iteration as i32 - 1).max(0) as f64;
        let t_gamma = (iteration - nearest_lower_power_of_4) as f64;

        // Regret discount: (t-1)^1.5 / ((t-1)^1.5 + 1)
        let pow_alpha = t_alpha * t_alpha.sqrt(); // (t-1)^1.5
        let alpha_t = (pow_alpha / (pow_alpha + 1.0)) as f32;

        // Strategy discount: (t_gamma / (t_gamma + 1))^3
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);

        DiscountParams {
            alpha_t,
            beta_t: 0.5, // Constant discount for negative regrets
            gamma_t: pow_gamma as f32,
        }
    }

    fn requires_storage4(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn CfrAlgorithmTrait> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcfr_discount_iteration_0() {
        let algo = DcfrAlgorithm;
        let params = algo.compute_discounts(0);
        assert_eq!(params.alpha_t, 0.0);
        assert_eq!(params.beta_t, 0.5);
        assert_eq!(params.gamma_t, 0.0);
    }

    #[test]
    fn test_dcfr_discount_iteration_1() {
        let algo = DcfrAlgorithm;
        let params = algo.compute_discounts(1);
        assert_eq!(params.alpha_t, 0.0); // (0)^1.5 / (0 + 1) = 0
        assert_eq!(params.beta_t, 0.5);
    }

    #[test]
    fn test_dcfr_discount_iteration_10() {
        let algo = DcfrAlgorithm;
        let params = algo.compute_discounts(10);

        // t-1 = 9, so (9^1.5) / (9^1.5 + 1)
        let expected_alpha = (27.0_f64 / 28.0) as f32;
        assert!((params.alpha_t - expected_alpha).abs() < 0.001);
        assert_eq!(params.beta_t, 0.5);
    }

    #[test]
    fn test_dcfr_no_storage4() {
        let algo = DcfrAlgorithm;
        assert!(!algo.requires_storage4());
    }

    #[test]
    fn test_dcfr_name() {
        let algo = DcfrAlgorithm;
        assert_eq!(algo.name(), "DCFR");
    }

    #[test]
    fn test_alpha_increases_with_iteration() {
        let algo = DcfrAlgorithm;
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
        let algo = DcfrAlgorithm;

        for i in 0..10000 {
            let params = algo.compute_discounts(i);
            assert!(params.alpha_t >= 0.0 && params.alpha_t < 1.0,
                "Alpha should be in [0, 1) (iteration {})", i);
        }
    }
}
