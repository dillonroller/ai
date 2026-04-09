use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "Load a real model's tokenizer and inspect how it splits text.")]
struct Args {
    #[arg(short, long, default_value = "Qwen/Qwen2-0.5B-Instruct")]
    model: String,

    #[arg(short, long)]
    compare: Vec<String>,

    text: String,
}

fn load_tokenizer(repo_id: &str) -> Result<Tokenizer> {
    let path = Api::new()?
        .model(repo_id.to_string())
        .get("tokenizer.json")?;
    Tokenizer::from_file(path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("loading tokenizer for {repo_id}"))
}

fn inspect(repo_id: &str, tokenizer: &Tokenizer, text: &str) -> Result<()> {
    let encoded = tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
    let ids = encoded.get_ids();
    let pieces: Vec<String> = ids
        .iter()
        .map(|id| {
            tokenizer
                .id_to_token(*id)
                .unwrap_or_else(|| format!("<{id}>"))
        })
        .collect();

    println!("== {repo_id} ==");
    println!("tokens ({}): ", ids.len());
    for (id, piece) in ids.iter().zip(pieces.iter()) {
        println!("  {id:>6}  {}", piece.replace('Ġ', "·").replace('Ċ', "⏎"));
    }
    let roundtrip = tokenizer.decode(ids, true).map_err(anyhow::Error::msg)?;
    println!("decoded: {roundtrip:?}");
    println!();
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let primary = load_tokenizer(&args.model)?;
    inspect(&args.model, &primary, &args.text)?;

    for other in &args.compare {
        match load_tokenizer(other) {
            Ok(t) => inspect(other, &t, &args.text)?,
            Err(e) => eprintln!("skip {other}: {e}"),
        }
    }

    Ok(())
}
