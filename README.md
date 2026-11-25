# MiniGPT

A small GPT-style (decoder-only) transformer built from scratch in PyTorch. Train it on any text file, then generate new text in the style of that data.

>Goal: build, train, and talk to your own transformer while understanding every line.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Training](#training)
- [Generating Text](#generating-text)
- [Model Configurations](#model-configurations)
- [Results](#results)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

## Features

- Causal multi-head self-attention, MLP, residual connections, and pre-LayerNorm, all written by hand
- Character-level tokenizer (BPE planned)
- Config-driven model sizes: `tiny`, `small`, `medium`
- Training loop with AdamW, warmup + cosine LR schedule, gradient clipping, mixed precision, and gradient accumulation
- Checkpointing and resume, so it works on free Colab/Kaggle sessions
- Text generation with temperature, top-k, and top-p sampling
- Test suite covering shapes, causality, tokenizer, and overfitting a single batch

## How It Works

```
text -> tokenizer -> token ids
     -> token embedding + positional embedding
     -> N x [ LayerNorm -> Causal Self-Attention -> + residual
              LayerNorm -> MLP                   -> + residual ]
     -> LayerNorm -> linear head -> next-token probabilities
```

The model is trained to predict the next token at every position. At generation time it repeatedly samples a token, appends it to the context, and continues.

See [`architecture.md`](architecture.md) for the full design.

## Project Structure

```
minigpt/
├── README.md
├── architecture.md
├── TRD.md
├── PRT.md
├── requirements.txt
├── configs/            # tiny.yaml, small.yaml, medium.yaml
├── data/
│   ├── raw/            # put your .txt here
│   └── processed/      # train.bin, val.bin, tokenizer.json
├── minigpt/
│   ├── config.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── model/          # attention.py, mlp.py, block.py, gpt.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── utils.py
├── scripts/
│   └── prepare_data.py
├── tests/
├── notebooks/
└── checkpoints/        # saved models (git-ignored)
```

## Quickstart

### 1. Requirements

- Python 3.10+
- PyTorch 2.2+
- Optional: an NVIDIA GPU (the `tiny` config runs on CPU)

### 2. Install

```bash
git clone https://github.com/<your-username>/minigpt.git
cd minigpt

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Get data

Any plain UTF-8 text file works. For a quick start, use TinyShakespeare:

```bash
mkdir -p data/raw
curl -o data/raw/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 4. Prepare data

```bash
python scripts/prepare_data.py --input data/raw/input.txt --tokenizer char
```

This splits the text 90/10 into train and validation sets, builds the vocabulary, and writes `data/processed/`.

### 5. Train

```bash
python -m minigpt.train --config configs/small.yaml
```

### 6. Generate

```bash
python -m minigpt.generate \
  --checkpoint checkpoints/best.pt \
  --prompt "ROMEO:" \
  --max_new_tokens 300 \
  --temperature 0.8 \
  --top_k 40
```

## Training

Training is controlled by a YAML config. Key settings:

| Setting | Meaning | Default (`small`) |
| `block_size` | Context length in tokens | 256 |
| `n_layer` | Number of transformer blocks | 6 |
| `n_head` | Attention heads per block | 6 |
| `n_embd` | Embedding dimension | 384 |
| `dropout` | Dropout rate | 0.2 |
| `batch_size` | Sequences per step | 64 |
| `learning_rate` | Peak learning rate | 3e-4 |
| `max_steps` | Total training steps | 5000 |
| `dtype` | `float32`, `float16`, or `bfloat16` | `bfloat16` |

**Resume from a checkpoint:**

```bash
python -m minigpt.train --config configs/small.yaml --resume checkpoints/last.pt
```

**Low on GPU memory?** Lower `batch_size` and raise `grad_accum_steps` to keep the same effective batch size.

During training you will see train and validation loss, plus a short generated sample at each evaluation.

### Using Google Colab

1. Upload the repo or clone it in a notebook cell.
2. Switch the runtime to GPU (Runtime > Change runtime type).
3. Run the Quickstart commands in cells prefixed with `!`.
4. Save checkpoints to Google Drive so a disconnect does not lose progress.

## Generating Text

| Flag | Description | Typical value |
| `--prompt` | Starting text | `"ROMEO:"` |
| `--max_new_tokens` | Number of tokens to generate | 300 |
| `--temperature` | Lower = safer, higher = more random (0 = greedy) | 0.7 to 1.0 |
| `--top_k` | Sample only from the k most likely tokens | 40 |
| `--top_p` | Nucleus sampling threshold | 0.9 to 1.0 |
| `--seed` | For reproducible output | 1337 |

## Model Configurations

| Config | Layers | Heads | d_model | Context | Params | Hardware |
| `tiny` | 4 | 4 | 128 | 128 | ~0.8M | CPU or laptop |
| `small` | 6 | 6 | 384 | 256 | ~10.7M | Colab T4 / consumer GPU |
| `medium` | 12 | 12 | 768 | 512 | ~85M | 16 to 24 GB GPU |

## Results

> Fill this section in after your own training run.

| Config | Dataset | Steps | Train loss | Val loss | Perplexity |
| small | TinyShakespeare (char) | 5000 | _TBD_ | _TBD_ | _TBD_ |

**Loss curve:**

```
![loss curve](docs/loss_curve.png)
```

**Sample output** (`temperature=0.8`, `top_k=40`):

```
<paste a generated sample here>
```

## Testing

```bash
pytest -q
```

The suite checks:

- Tokenizer round-trip (`decode(encode(s)) == s`)
- Output shapes for different batch and sequence sizes
- **Causality:** changing a future token does not change earlier outputs
- Manual attention matches PyTorch's `scaled_dot_product_attention`
- Initial loss is close to `ln(vocab_size)`
- The model can overfit a single small batch
- Checkpoint save and resume

Run these before any long training job. They catch most wiring bugs early.

## Roadmap

- [x] Project design docs
- [ ] Data pipeline and character tokenizer
- [ ] Bigram baseline
- [ ] Transformer model (attention, MLP, blocks, GPT)
- [ ] Training loop, checkpointing, evaluation
- [ ] Generation CLI
- [ ] Test suite and results report
- [ ] BPE tokenizer
- [ ] Mixed precision and fused attention (SDPA)
- [ ] Ablation experiments (layers, heads, context, dropout)
- [ ] Gradio web demo
- [ ] KV cache for fast inference
- [ ] RoPE and RMSNorm
- [ ] LoRA fine-tuning of a pretrained model
- [ ] Hugging Face export

## Documentation

| File | Contents |
| [`architecture.md`](architecture.md) | Model design, data flow, components, repo layout |
| [`TRD.md`](TRD.md) | Technical requirements, hyperparameters, testing, acceptance criteria |
| [`PRT.md`](PRT.md) | Product requirements, user stories, scope, metrics |



## Troubleshooting

| Problem | Likely fix |

| `CUDA out of memory` | Lower `batch_size` or `block_size`; raise `grad_accum_steps`; use the `tiny` config |
| Loss is `nan` | Lower `learning_rate`, keep gradient clipping on, try `float32` to isolate the issue |
| Loss stuck near `ln(vocab_size)` (about 4.17 for 65 characters) | Learning rate too low or a bug in the data/targets; run the overfit-one-batch test |
| Validation loss rises while train loss falls | Overfitting: raise dropout, add data, or stop earlier |
| Very slow on CPU | Use `tiny`, shrink `block_size`, or use a free Colab GPU |
| `n_embd must be divisible by n_head` | Change one of the two values in your config |
| Garbled output | Train longer, or lower `temperature` and set `top_k` |


## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out* (video) and the [nanoGPT](https://github.com/karpathy/nanoGPT) repository
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)


## License

MIT License. See `LICENSE` for details. 

## Acknowledgments

Inspired by Andrej Karpathy's educational work on nanoGPT and the many open-source transformer tutorials that make projects like this possible.
