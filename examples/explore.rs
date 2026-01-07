use postflop_solver::*;
use std::io::{self, Write};

struct ActionHistory {
    actions: Vec<usize>,
}

impl ActionHistory {
    fn new() -> Self {
        Self {
            actions: Vec::new(),
        }
    }

    fn push(&mut self, action_idx: usize) {
        self.actions.push(action_idx);
    }

    fn pop(&mut self) -> bool {
        if !self.actions.is_empty() {
            self.actions.pop();
            true
        } else {
            false
        }
    }

    fn replay(&self, game: &mut PostFlopGame) {
        for &action_idx in &self.actions {
            game.play(action_idx);
        }
    }

    fn depth(&self) -> usize {
        self.actions.len()
    }
}

fn print_help() {
    println!("\n=== Commands ===");
    println!("  actions         - Show available actions");
    println!("  strategy        - Show GTO strategy for current player");
    println!("  play <n>        - Execute action n (0-indexed)");
    println!("  back            - Undo last move");
    println!("  info            - Show current game state");
    println!("  help            - Show this help");
    println!("  quit            - Exit explorer");
    println!();
    println!(
        "Note: When at a chance node (player = CHANCE), use 'play <n>' to deal a specific card."
    );
    println!();
}

fn print_game_info(game: &PostFlopGame, history: &ActionHistory) {
    println!("\n=== Game State ===");

    let player = game.current_player();
    let player_name = match player {
        0 => "OOP",
        1 => "IP",
        4 => "CHANCE (card dealing)",
        _ => "Unknown",
    };

    println!("Current player: {} ({})", player_name, player);
    println!("Depth: {} moves", history.depth());

    if player == 4 {
        println!("\nThis is a chance node - a card will be dealt.");
        println!("Use 'actions' to see which cards can be dealt.");
    }

    println!();
}

fn print_actions(game: &PostFlopGame) {
    let actions = game.available_actions();
    let player = game.current_player();

    println!("\n=== Available Actions ===");

    if player == 4 {
        // Chance node - show card names
        println!("Cards that can be dealt:");
        for (i, action) in actions.iter().enumerate() {
            if let Action::Chance(card_idx) = action {
                let card_name = card_to_string(*card_idx as u8)
                    .unwrap_or_else(|_| format!("Card({})", card_idx));
                println!("  [{}] {}", i, card_name);
            }
        }
    } else {
        // Player node - show actions normally
        for (i, action) in actions.iter().enumerate() {
            println!("  [{}] {:?}", i, action);
        }
    }

    println!();
}

fn print_strategy(game: &PostFlopGame) {
    let player = game.current_player();

    println!("\n=== GTO Strategy ===");

    // Check if this is a chance node
    if player == 4 {
        println!("Strategy not available at chance nodes.");
        println!("Chance nodes represent card dealing - each card has equal probability.");
        println!();
        return;
    }

    let actions = game.available_actions();
    let strategy = game.strategy();

    let player_name = if player == 0 { "OOP" } else { "IP" };

    println!("Player: {}", player_name);

    // Get current player's cards
    let cards = game.private_cards(player);

    // For simplicity, show strategy for first few hands
    if !cards.is_empty() {
        let hand_strings = holes_to_strings(cards).unwrap_or_default();

        println!("Number of hands: {}", hand_strings.len());
        println!();

        // Show strategy for first hand as example
        if !hand_strings.is_empty() {
            println!("Example hand: {}", hand_strings[0]);

            for (i, action) in actions.iter().enumerate() {
                let prob = strategy[i * cards.len()];
                println!("  {:?}: {:.2}%", action, prob * 100.0);
            }
        }
    } else {
        println!("No cards available for current player");
    }

    println!();
}

fn reload_game(file_path: &str, history: &ActionHistory) -> PostFlopGame {
    let (mut game, _memo): (PostFlopGame, String) =
        load_data_from_file(file_path, None).expect("Failed to load game file");

    game.cache_normalized_weights();

    // Replay all actions to restore state
    history.replay(&mut game);

    game
}

fn main() {
    // Get file path from command line
    let file_path = std::env::args().nth(1).unwrap_or_else(|| {
        println!("Usage: explore <solved_game.bin>");
        std::process::exit(1);
    });

    println!("Loading: {}", file_path);

    let (mut game, memo): (PostFlopGame, String) =
        load_data_from_file(&file_path, None).expect("Failed to load game file");

    game.cache_normalized_weights();

    println!("Game loaded successfully!");
    if !memo.is_empty() {
        println!("Memo: {}", memo);
    }

    // Initialize action history
    let mut history = ActionHistory::new();

    println!("\nType 'help' for commands");

    // Main interactive loop
    loop {
        print!("\nexplore> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let command = parts[0].to_lowercase();

        match command.as_str() {
            "help" | "h" => {
                print_help();
            }

            "quit" | "q" | "exit" => {
                println!("Goodbye!");
                break;
            }

            "info" | "i" => {
                print_game_info(&game, &history);
            }

            "actions" | "a" => {
                print_actions(&game);
            }

            "strategy" | "s" => {
                print_strategy(&game);
            }

            "back" | "b" | "undo" | "u" => {
                if history.pop() {
                    println!("Moved back to previous state");
                    // Reload game and replay history
                    game = reload_game(&file_path, &history);
                } else {
                    println!("Already at root - cannot go back");
                }
            }

            "play" | "p" => {
                if parts.len() < 2 {
                    println!("Usage: play <action_index>");
                    continue;
                }

                let action_idx: usize = match parts[1].parse() {
                    Ok(n) => n,
                    Err(_) => {
                        println!("Invalid action index");
                        continue;
                    }
                };

                let actions = game.available_actions();

                if action_idx >= actions.len() {
                    println!("Action index out of range (max: {})", actions.len() - 1);
                    continue;
                }

                // Execute the action
                let action = &actions[action_idx];

                // Print action with card name if it's a chance action
                match action {
                    Action::Chance(card_idx) => {
                        let card_name = card_to_string(*card_idx as u8)
                            .unwrap_or_else(|_| format!("Card({})", card_idx));
                        println!("Dealt: {}", card_name);
                    }
                    _ => {
                        println!("Executed: {:?}", action);
                    }
                }

                game.play(action_idx);
                history.push(action_idx);
            }

            _ => {
                println!("Unknown command: {}", command);
                println!("Type 'help' for available commands");
            }
        }
    }
}
