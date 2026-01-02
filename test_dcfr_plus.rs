use postflop_solver::{Game, CfrAlgorithm, solve};

fn main() {
    // Create a simple game configuration
    let mut game = create_simple_game();
    
    // Set the CFR algorithm to DCFR+
    game.set_cfr_algorithm(CfrAlgorithm::DCRFPlus);
    
    // Solve the game
    println!("Solving with DCFR+...");
    let exploitability = solve(&mut game, 100, 0.001, true);
    
    println!("Final exploitability: {}", exploitability);
}

fn create_simple_game() -> impl Game {
    // This is a placeholder - you would need to create an actual game instance
    // based on the available game implementations in the project
    unimplemented!("Create a simple game instance")
}