/// Esempio: Come estendere il sistema CFR con un algoritmo custom
///
/// Questo esempio mostra come implementare Linear CFR (CFR senza discounting)
/// usando il trait CfrAlgorithmTrait. Dimostra l'estensibilità del sistema
/// senza dover modificare il codice core.
///
/// Linear CFR è la variante più semplice di CFR:
/// - No discounting sui regrets (alpha = 1.0)
/// - Linear averaging sulla strategy (gamma cresce linearmente)
/// - Convergenza più lenta ma implementazione semplice

use postflop_solver::cfr_algorithms::{CfrAlgorithmTrait, DiscountParams};

/// Linear CFR - Variante base senza discounting
///
/// Caratteristiche:
/// - Alpha = 1.0 (nessun discount sui regrets)
/// - Beta = 1.0 (non usato, no sign-based discounting)
/// - Gamma = t / (t+1) (linear averaging, no cubic power)
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearCfrAlgorithm;

impl CfrAlgorithmTrait for LinearCfrAlgorithm {
    fn name(&self) -> &'static str {
        "Linear CFR"
    }

    fn compute_discounts(&self, iteration: u32) -> DiscountParams {
        let t = iteration as f32;

        // Linear CFR: no regret discounting, simple linear averaging
        DiscountParams {
            alpha_t: 1.0,  // No regret discounting
            beta_t: 1.0,   // No sign-based discounting
            gamma_t: if iteration == 0 {
                0.0
            } else {
                t / (t + 1.0)  // Linear averaging (no cubic power)
            },
        }
    }

    fn requires_storage4(&self) -> bool {
        false  // Linear CFR non richiede storage4
    }

    fn clone_box(&self) -> Box<dyn CfrAlgorithmTrait> {
        Box::new(*self)
    }
}

fn main() {
    println!("=== Custom CFR Algorithm Example ===\n");

    // Crea un'istanza del nuovo algoritmo
    let linear_cfr = LinearCfrAlgorithm;

    println!("Algoritmo: {}", linear_cfr.name());
    println!("Richiede storage4: {}\n", linear_cfr.requires_storage4());

    // Mostra i discount params per diverse iterazioni
    println!("Discount parameters per iterazione:\n");
    println!("{:>10} {:>10} {:>10} {:>10}", "Iteration", "Alpha", "Beta", "Gamma");
    println!("{:-<44}", "");

    for iteration in [0, 1, 5, 10, 50, 100, 500, 1000] {
        let params = linear_cfr.compute_discounts(iteration);
        println!(
            "{:>10} {:>10.4} {:>10.4} {:>10.4}",
            iteration,
            params.alpha_t,
            params.beta_t,
            params.gamma_t
        );
    }

    println!("\n=== Confronto con altri algoritmi ===\n");

    // Importa algoritmi standard per confronto
    use postflop_solver::cfr_algorithms::{
        DcfrAlgorithm, DcfrPlusAlgorithm, SapcfrPlusAlgorithm
    };

    let algorithms: Vec<(&str, Box<dyn CfrAlgorithmTrait>)> = vec![
        ("Linear CFR", Box::new(LinearCfrAlgorithm)),
        ("DCFR", Box::new(DcfrAlgorithm)),
        ("DCFR+", Box::new(DcfrPlusAlgorithm)),
        ("SAPCFR+", Box::new(SapcfrPlusAlgorithm)),
    ];

    println!("Gamma values @ iteration 100:\n");
    for (name, algo) in &algorithms {
        let params = algo.compute_discounts(100);
        println!("  {:<15} gamma = {:.6}", name, params.gamma_t);
    }

    println!("\nAlpha values @ iteration 100:\n");
    for (name, algo) in &algorithms {
        let params = algo.compute_discounts(100);
        println!("  {:<15} alpha = {:.6}", name, params.alpha_t);
    }

    println!("\nStorage4 requirements:\n");
    for (name, algo) in &algorithms {
        let requires = if algo.requires_storage4() { "YES" } else { "NO" };
        println!("  {:<15} {}", name, requires);
    }

    println!("\n=== Note ===");
    println!("Linear CFR converge più lentamente di DCFR/DCFR+ ma è più semplice.");
    println!("Per usarlo in produzione, aggiungerlo a CfrAlgorithm enum in solver.rs");
    println!("e implementare la logica di update regrets (come DCFR ma senza discount).\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_cfr_no_discount() {
        let algo = LinearCfrAlgorithm;

        // Alpha e Beta devono sempre essere 1.0 (no discounting)
        for i in 0..1000 {
            let params = algo.compute_discounts(i);
            assert_eq!(params.alpha_t, 1.0);
            assert_eq!(params.beta_t, 1.0);
        }
    }

    #[test]
    fn test_linear_cfr_gamma_increases() {
        let algo = LinearCfrAlgorithm;
        let mut prev_gamma = 0.0;

        for i in 1..100 {
            let params = algo.compute_discounts(i);
            assert!(params.gamma_t > prev_gamma, "Gamma should increase");
            prev_gamma = params.gamma_t;
        }
    }

    #[test]
    fn test_linear_cfr_gamma_approaches_one() {
        let algo = LinearCfrAlgorithm;
        let params = algo.compute_discounts(10000);

        // Per t molto grande, t/(t+1) -> 1
        // t=10000 => 10000/10001 ≈ 0.9999
        assert!(params.gamma_t > 0.999);
    }

    #[test]
    fn test_linear_cfr_no_storage4() {
        let algo = LinearCfrAlgorithm;
        assert!(!algo.requires_storage4());
    }
}
