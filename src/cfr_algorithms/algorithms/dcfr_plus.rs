use super::super::traits::*;

/// Discounted CFR Plus Algorithm
///
/// DCFR+ è una variante di DCFR che:
/// - Usa discounting uniforme (solo alpha_t, no beta)
/// - Applica regret floor clipping a zero dopo l'update
/// - Mantiene solo regrets non-negativi
#[derive(Debug, Clone, Copy, Default)]
pub struct DcfrPlusAlgorithm;

impl CfrAlgorithmTrait for DcfrPlusAlgorithm {
    fn name(&self) -> &'static str {
        "DCFR+"
    }

    fn compute_discounts(&self, iteration: u32) -> DiscountParams {
        // DCFR+: Using (t-1) as per the paper
        // Strategy reset heuristic: 0, 1, 4, 16, 64, 256, ...
        let nearest_lower_power_of_4 = match iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };

        // Align t_alpha with DCFR: use (t-1)
        let t_alpha = (iteration as i32 - 1).max(0) as f64;
        let t_gamma = (iteration - nearest_lower_power_of_4) as f64;

        // Regret discount: (t-1)^1.5 / ((t-1)^1.5 + 1)
        let pow_alpha = t_alpha.powf(1.5);
        let alpha_t = (pow_alpha / (pow_alpha + 1.0)) as f32;

        // Strategy discount: (t_gamma / (t_gamma + 1))^3
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);

        DiscountParams {
            alpha_t,
            beta_t: 0.5, // Not used in DCFR+ (uniform discounting)
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
    fn test_dcfr_plus_discount_iteration_0() {
        let algo = DcfrPlusAlgorithm;
        let params = algo.compute_discounts(0);
        assert_eq!(params.alpha_t, 0.0);
        assert_eq!(params.gamma_t, 0.0);
    }

    #[test]
    fn test_dcfr_plus_discount_iteration_1() {
        let algo = DcfrPlusAlgorithm;
        let params = algo.compute_discounts(1);
        assert_eq!(params.alpha_t, 0.0); // (0)^1.5 / (0 + 1) = 0
    }

    #[test]
    fn test_dcfr_plus_discount_iteration_10() {
        let algo = DcfrPlusAlgorithm;
        let params = algo.compute_discounts(10);

        // t-1 = 9, so (9^1.5) / (9^1.5 + 1)
        let expected_alpha = (27.0_f64 / 28.0) as f32;
        assert!((params.alpha_t - expected_alpha).abs() < 0.001);
    }

    #[test]
    fn test_dcfr_plus_no_storage4() {
        let algo = DcfrPlusAlgorithm;
        assert!(!algo.requires_storage4());
    }

    #[test]
    fn test_dcfr_plus_name() {
        let algo = DcfrPlusAlgorithm;
        assert_eq!(algo.name(), "DCFR+");
    }

    #[test]
    fn test_alpha_increases_with_iteration() {
        let algo = DcfrPlusAlgorithm;
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
        let algo = DcfrPlusAlgorithm;

        for i in 0..10000 {
            let params = algo.compute_discounts(i);
            assert!(params.alpha_t >= 0.0 && params.alpha_t < 1.0,
                "Alpha should be in [0, 1) (iteration {})", i);
        }
    }

    #[test]
    fn test_dcfr_plus_matches_dcfr_discounts() {
        // DCFR+ e DCFR dovrebbero avere gli stessi parametri di discount
        let dcfr = super::super::dcfr::DcfrAlgorithm;
        let dcfr_plus = DcfrPlusAlgorithm;

        for i in 0..100 {
            let params_dcfr = dcfr.compute_discounts(i);
            let params_plus = dcfr_plus.compute_discounts(i);

            assert!((params_dcfr.alpha_t - params_plus.alpha_t).abs() < 0.0001,
                "Alpha should match at iteration {}", i);
            assert!((params_dcfr.gamma_t - params_plus.gamma_t).abs() < 0.0001,
                "Gamma should match at iteration {}", i);
        }
    }
}
