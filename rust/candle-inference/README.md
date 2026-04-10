# candle-inference

Stream tokens from a small LLM running **entirely on your machine** — no API, no server, no Python. Uses HuggingFace's [`candle`](https://github.com/huggingface/candle) framework.

## Build

CPU:
```bash
cargo build --release
```

Apple Silicon (Metal):
```bash
cargo build --release --features metal
```

NVIDIA (CUDA):
```bash
cargo build --release --features cuda
```

## Run

```bash
./target/release/candle-inference "Write a haiku about Rust"
```

First run downloads ~1GB of model weights from HuggingFace Hub into `~/.cache/huggingface/`. Subsequent runs are instant-start.

Flags:
- `--model` — HF repo id, default `Qwen/Qwen2-0.5B-Instruct`
- `--max-tokens` — generation budget, default `256`
- `--temperature` — sampling temp, default `0.7`
- `--seed` — RNG seed for reproducibility

## What it's actually doing

1. Downloads `config.json`, `tokenizer.json`, `model.safetensors` from the Hub
2. Memory-maps the weights into candle tensors
3. Tokenizes your prompt using the model's own tokenizer
4. Runs a generation loop: forward pass → logits → sample → append → repeat
5. Decodes each new token and streams it to stdout as it's generated

That loop is the core of every LLM in the world. Python's `model.generate()` hides it. Here it's ~30 lines.

## Theory

### The model is a function from tokens to logits

A decoder-only transformer like Qwen2 is, at heart, a function:

```
forward(token_ids: [seq_len]) -> logits: [seq_len, vocab_size]
```

For each position in the input, it produces an unnormalized score ("logit") for every possible next token in the vocabulary (~150k tokens for Qwen2). The *last* row of that output — the logits at the final position — is the model's prediction for "what comes next." That's it. An LLM is a next-token predictor.

### Autoregressive generation

To generate more than one token, you just run the model, pick a token, append it to the input, and run again:

```
tokens = tokenize(prompt)
loop:
    logits = model(tokens)[-1]       # logits for the next position
    next   = sample(logits)           # pick one token
    tokens.append(next)
    if next == eos: break
```

"Autoregressive" = each step's output becomes the next step's input. This is why generation is inherently sequential and can't be parallelized across tokens (only within a single forward pass).

### The KV cache — why step 0 and step N are different

Naive generation would re-run the entire sequence through the model every step: O(N²) in tokens. Transformers avoid this with a **KV cache**: every self-attention layer stores the Key and Value tensors for tokens it has already seen. On subsequent steps you only need to compute K/V for the *new* token and attend against the cached ones.

That's why the code does:

```rust
let (context, pos) = if index == 0 {
    (tokens.as_slice(), 0)               // step 0: feed the whole prompt
} else {
    (&tokens[tokens.len()-1..], tokens.len()-1)  // step N: feed only the new token
};
```

The `pos` argument tells the model "this new token goes at position N" so rotary position embeddings land correctly. Without this, you'd either get wrong results or kill throughput by a factor of hundreds.

### Sampling — temperature, logits, and randomness

`logits` are raw scores. To turn them into probabilities, apply softmax:

```
p[i] = exp(logits[i] / T) / sum(exp(logits[j] / T))
```

Where `T` is **temperature**:
- `T → 0`: distribution collapses onto the single highest-scoring token. Deterministic, repetitive, "safe."
- `T = 1`: use the model's raw distribution.
- `T > 1`: flatten the distribution, more surprising / creative / incoherent.

`LogitsProcessor::sample` does this (plus optional top-k / top-p filtering) and returns one sampled token ID.

### The chat template

The model was fine-tuned on conversations wrapped in specific sentinel tokens:

```
<|im_start|>user
Write a haiku about Rust<|im_end|>
<|im_start|>assistant
```

Without this wrapping, the base transformer has no reason to behave like an assistant — it'll just continue text. This is why `format_prompt` exists and why every chat model has its own template (Llama, Mistral, Qwen, ChatML all differ).

### Memory-mapped weights

`VarBuilder::from_mmaped_safetensors` uses `mmap` to map the ~1GB weights file into virtual memory without reading it. The OS pages tensors in on demand. This is why startup after the first run feels instant — you're not reading gigabytes, you're just handing the kernel a promise to read them lazily.

### `safetensors` vs `.bin`

`safetensors` is a simple binary format: a JSON header describing tensor names, dtypes, and shapes, followed by a flat buffer of tensor data. Unlike PyTorch's `.bin` (which is pickle), it's not executable and can be safely mmap'd. Every modern HF model ships it.

## Why this matters

Ollama, llama.cpp, vLLM, TGI — they're all variations on this loop wrapped in different servers. Understanding it is the difference between using LLMs and understanding them.
