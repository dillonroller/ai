# tokenizer-playground

Load a real model's tokenizer and see exactly how it chops your text into tokens. Uses HuggingFace's [`tokenizers`](https://github.com/huggingface/tokenizers) crate — the same one that powers every Python `transformers` tokenizer under the hood.

## Build

```bash
cargo build --release
```

## Run

```bash
./target/release/tokenizer-playground "The quick brown fox jumps over the lazy dog"
```

Compare how different models tokenize the same string:

```bash
./target/release/tokenizer-playground \
  --compare meta-llama/Llama-3.2-1B \
  --compare gpt2 \
  "tokenization is surprisingly weird"
```

Each byte-pair boundary is rendered with `·` for leading space and `⏎` for newline so you can see the chunking clearly.

## Theory

### Why tokenize at all?

A language model can't operate on raw characters or raw bytes directly — the vocabulary would be too small (inefficient) or too large (expensive). Tokenization is a compression step: turn a string into a sequence of ~1 token per ~4 characters of English, where each token is an integer ID into a fixed vocabulary (typically 30k–200k).

The model's input and output layers are sized to that vocabulary. Every embedding lookup, every softmax, every logit refers to a specific token ID. Change the tokenizer and you've changed the model's entire interface.

### Byte-Pair Encoding (BPE)

Modern LLMs (GPT, Llama, Qwen, Mistral) use a variant of **BPE**. Training it works like this:

1. Start with a vocabulary of single bytes (256 tokens).
2. Count all adjacent token pairs in a big corpus.
3. Merge the most frequent pair into a new single token. Add it to the vocab.
4. Repeat until you hit the target vocab size.

The result is a learned "merge table." Common sequences (`the`, `ing`, `tion`, `def `, `\n    `) become single tokens. Rare sequences stay as multiple tokens. This is why:

- `"hello"` → 1 token (learned as a frequent word)
- `"antidisestablishmentarianism"` → 5+ tokens (rare, falls back to sub-pieces)
- `"こんにちは"` → many tokens (underrepresented in training)

**Encoding** at inference time is the reverse: start with bytes, greedily apply merges in the order they were learned, emit the resulting token IDs. This is what `tokenizer.encode()` does internally.

### The weird marker characters

BPE tokenizers preserve whitespace by *encoding* it into the token itself, not by tracking it separately. GPT-2-family tokenizers use:

- `Ġ` (U+0120) = a leading space
- `Ċ` (U+010A) = a newline

So `" hello"` becomes the single token `Ġhello`, distinct from `hello`. This is how `"tokenization"` and `" tokenization"` are different tokens with different IDs.

This playground replaces those with `·` and `⏎` so you can actually see them. Without that substitution, a raw token dump looks like cryptic Unicode garbage.

### Different tokenizers → different models

Every model ships its own `tokenizer.json` because the vocab was learned on that model's specific training corpus. Consequences:

- **Token counts differ.** GPT-4's tokenizer is much more efficient on code than GPT-2's. Same sentence, different bill.
- **Context windows differ in *meaning*.** A 128k-context Llama model and a 128k-context GPT model hold different amounts of actual *text*.
- **You cannot mix tokenizers across models.** Feeding Llama tokens into a Qwen model produces garbage — the IDs index into different embedding tables.
- **Failure modes are tokenizer-shaped.** The classic "model can't count letters in a word" bug is because the model sees `strawberry` as 2–3 tokens, not 10 letters.

### Why this matters in practice

- **Cost:** LLM APIs bill by token. A 2x worse tokenizer is a 2x worse bill.
- **Truncation bugs:** Context limits are measured in tokens, not characters. Most "my RAG is randomly dropping content" bugs are actually "I counted characters and overflowed the token limit."
- **Prompt sensitivity:** Adding or removing a single space can change how the rest of the prompt tokenizes. This is why `"Answer:"` and `"Answer: "` can produce different outputs.
- **Fine-tuning gotchas:** If your training data has different whitespace conventions than your inference prompts, the model sees different tokens and performs worse.

Five minutes with this tool beats an hour reading about BPE.

## Why this is worth knowing

- Token counts drive cost and context limits — being off by 2x means your bills are off by 2x.
- Model behavior is shaped by its vocab: Llama, GPT-4, and Qwen tokenize the same sentence into very different pieces.
- Most bugs in RAG / prompt engineering are actually tokenization bugs in disguise (truncation, overflow, weird whitespace).
