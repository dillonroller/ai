use std::io::Write;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "Stream tokens from a small LLM running locally via candle.")]
struct Args {
    #[arg(short, long, default_value = "Qwen/Qwen2-0.5B-Instruct")]
    model: String,

    #[arg(short, long, default_value_t = 256)]
    max_tokens: usize,

    #[arg(short, long, default_value_t = 0.7)]
    temperature: f64,

    #[arg(short, long, default_value_t = 299792458)]
    seed: u64,

    prompt: String,
}

fn pick_device() -> Device {
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn load_model(repo_id: &str, device: &Device) -> Result<(ModelForCausalLM, Tokenizer, Config)> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;

    let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };
    let model = ModelForCausalLM::new(&config, vb)?;
    Ok((model, tokenizer, config))
}

fn format_prompt(user: &str) -> String {
    format!(
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        user = user
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device();
    eprintln!("device: {device:?}");

    let t0 = Instant::now();
    let (mut model, tokenizer, _config) = load_model(&args.model, &device)?;
    eprintln!("loaded {} in {:.1?}", args.model, t0.elapsed());

    let prompt = format_prompt(&args.prompt);
    let encoded = tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
    let mut tokens: Vec<u32> = encoded.get_ids().to_vec();

    let eos_id = tokenizer
        .token_to_id("<|im_end|>")
        .context("model tokenizer missing <|im_end|>")?;

    let mut logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), None);
    let mut generated = 0usize;
    let start = Instant::now();
    let mut stdout = std::io::stdout().lock();

    for index in 0..args.max_tokens {
        let (context, pos) = if index == 0 {
            (tokens.as_slice(), 0)
        } else {
            (&tokens[tokens.len() - 1..], tokens.len() - 1)
        };

        let input = Tensor::new(context, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = logits.i((logits.dim(0)? - 1, ..))?;

        let next = logits_processor.sample(&logits)?;
        if next == eos_id {
            break;
        }
        tokens.push(next);
        generated += 1;

        if let Ok(piece) = tokenizer.decode(&[next], true) {
            write!(stdout, "{piece}")?;
            stdout.flush()?;
        }
    }
    writeln!(stdout)?;

    let dt = start.elapsed().as_secs_f64();
    eprintln!("\n[{generated} tokens, {:.1} tok/s]", generated as f64 / dt);
    Ok(())
}

// Tensor indexing trait glue
use candle_core::IndexOp;
