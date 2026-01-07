use postflop_solver::*;
use std::fs::File;
use std::io::Write;

fn export_node_recursive(
    node: &MutexGuardLike<PostFlopNode>,
    depth: usize,
    path: &str,
    output: &mut File,
) -> std::io::Result<()> {
    let indent = "  ".repeat(depth);

    // Export node info
    writeln!(output, "{}Node: {}", indent, path)?;
    writeln!(output, "{}  Player: {}", indent, node.player())?;
    writeln!(output, "{}  Actions: {}", indent, node.num_actions())?;

    // Export strategy if not terminal/chance
    if !node.is_terminal() && !node.is_chance() {
        let strategy = node.strategy();
        writeln!(
            output,
            "{}  Strategy (first 10): {:?}",
            indent,
            &strategy[..strategy.len().min(10)]
        )?;
    }

    writeln!(output, "")?;

    // Recurse into children
    for i in 0..node.num_actions() {
        let child = node.play(i);
        let child_path = format!("{}/{}", path, i);
        export_node_recursive(&child, depth + 1, &child_path, output)?;
    }

    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input.bin> <output.txt>", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!("Loading game from: {}", input_path);

    let result = load_data_from_file::<PostFlopGame, _>(input_path, None);

    match result {
        Ok((game, memo)) => {
            println!("Loaded successfully!");
            println!("Memo: {}", memo);

            let mut output = File::create(output_path)?;

            writeln!(output, "=== GAME EXPORT ===")?;
            writeln!(output, "Memo: {}", memo)?;
            writeln!(output, "Quantization: {:?}", game.quantization_mode())?;
            writeln!(output, "Strategy bits: {}", game.strategy_bits())?;
            writeln!(output, "")?;

            println!("Exporting strategies...");

            let root = game.root();
            export_node_recursive(&root, 0, "root", &mut output)?;

            println!("Export complete! Saved to: {}", output_path);
            Ok(())
        }
        Err(e) => {
            eprintln!("Error loading file: {}", e);
            std::process::exit(1);
        }
    }
}
