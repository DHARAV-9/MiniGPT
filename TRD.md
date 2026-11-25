# Technical Requirements Document (TRD): MiniGPT

**Version:** 1.0
**Status:** Draft
**Companion docs:** `architecture.md` (design), `PRT.md` (product requirements)

---

## 1. Purpose

This document defines the technical requirements, tooling, standards, and acceptance criteria for building MiniGPT, a decoder-only transformer language model implemented from scratch in PyTorch. It answers **how** the project is built and **how we verify** it works.

---

## 2. Technology Stack

| Area | Choice | Version / Notes |
|---|---|---|
| Language | Python | 3.10+ |
| Deep learning | PyTorch | 2.2+ (needs SDPA, `torch.compile`) |
| Numerics | NumPy | 1.26+ |
| Config | PyYAML + dataclasses | |
| Tokenization | Custom char tokenizer; BPE via `tokenizers` (Hugging Face) or `tiktoken` | Phase 2 |
| Datasets | `datasets` (Hugging Face) or plain `.txt` | |
| Logging | `tqdm`, CSV, optional TensorBoard / Weights & Biases | |
| Testing | `pytest` | |
| Lint / format | `ruff`, `black` | |
| Type checking | `mypy` (light, optional) | |
| Version control | Git + GitHub | |
| Demo UI (optional) | Gradio | |
| Fine-tuning (optional) | `transformers`, `peft`, `accelerate` | |

---

## 3. Environment and Hardware

### 3.1 Minimum and recommended

| Tier | Hardware | What it can run |
|---|---|---|
| Minimum | CPU only, 8 GB RAM | `tiny` config (slow, a few hours) |
| Recommended | NVIDIA GPU with 8 GB+ VRAM, or Colab T4 / Kaggle P100 | `small` config in ~20 to 60 min |
| Stretch | 16 to 24 GB GPU | `medium` config |

### 3.2 Setup

```bash
git clone <repo>
cd minigpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # sanity check
```

### 3.3 Reproducibility
- Fixed global seed (`random`, `numpy`, `torch`, `torch.cuda`) set from config.
- Pinned versions in `requirements.txt`.
- Every run stores its resolved config and git commit hash with the checkpoint.

---

## 4. Functional Technical Requirements

### 4.1 Data

| ID | Requirement |
|---|---|
| D-1 | System shall load a UTF-8 text file and split it 90/10 into train and validation sets, with the split done before any tokenizer fitting. |
| D-2 | System shall provide a character-level tokenizer with `encode`, `decode`, `save`, `load`. |
| D-3 | System shall provide a BPE tokenizer trained on the training split only (Phase 2). |
| D-4 | Encoded data shall be stored as memory-mapped `uint16` binary files (`train.bin`, `val.bin`). |
| D-5 | Batch sampler shall return `(x, y)` tensors of shape `[B, T]` where `y` is `x` shifted left by one token. |
| D-6 | `decode(encode(s)) == s` for all strings composed of in-vocabulary characters. |

### 4.2 Model

| ID | Requirement |
|---|---|
| M-1 | Implement `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `GPT` as separate `nn.Module`s. |
| M-2 | Model hyperparameters shall be driven by `GPTConfig` (vocab size, block size, layers, heads, embedding dim, dropout, bias flag). |
| M-3 | `n_embd` must be divisible by `n_head`; config validation shall raise a clear error otherwise. |
| M-4 | Attention must be strictly causal: output at position `t` must not depend on tokens at positions `> t`. |
| M-5 | LM head shall share weights with token embedding (configurable). |
| M-6 | Model shall expose `generate()` supporting temperature, top-k, top-p, and max new tokens. |
| M-7 | Forward pass shall return `(logits, loss)` when targets are provided, and `(logits, None)` otherwise. |
| M-8 | Provide `count_parameters()` that reports total and non-embedding parameters. |
| M-9 | Context longer than `block_size` shall be cropped to the last `block_size` tokens during generation. |

### 4.3 Training

| ID | Requirement |
|---|---|
| T-1 | Training loop shall use AdamW with decoupled weight decay applied only to parameters with `ndim >= 2`. |
| T-2 | Learning rate shall follow linear warmup then cosine decay. |
| T-3 | Gradient clipping at global norm 1.0 (configurable). |
| T-4 | Support mixed precision (bf16 preferred, fp16 with GradScaler) and gradient accumulation. |
| T-5 | Evaluate on validation set every `eval_interval` steps over `eval_iters` batches; log train and val loss. |
| T-6 | Save a checkpoint at every `eval_interval` and keep a separate `best.pt` by validation loss. |
| T-7 | Checkpoint shall contain model state, optimizer state, scheduler/step, config, tokenizer path, RNG state. |
| T-8 | Training shall be resumable from a checkpoint with loss curve continuity. |
| T-9 | Training shall print a short generated sample at each evaluation for qualitative monitoring. |
| T-10 | Support CPU, single GPU, and (stretch) multi-GPU via DDP. |

### 4.4 Inference and Evaluation

| ID | Requirement |
|---|---|
| I-1 | CLI `generate.py` shall load a checkpoint and generate text from a prompt with configurable sampling parameters. |
| I-2 | Greedy decoding (temperature 0) shall be deterministic. |
| I-3 | Sampling with a fixed seed shall be reproducible. |
| I-4 | `evaluate.py` shall report validation loss and perplexity (`exp(loss)`). |
| I-5 | (Phase 3) KV cache shall produce token-for-token identical greedy outputs to the non-cached path. |

---

## 5. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| N-1 | Performance | `small` model on a T4 GPU shall sustain at least 20k tokens/second training throughput with bf16/fp16 (target; measured and reported). |
| N-2 | Memory | `small` config shall train in under 8 GB VRAM at batch size 64, block size 256, with accumulation available for lower memory. |
| N-3 | Quality | See acceptance metrics in Section 8. |
| N-4 | Reproducibility | Same seed, config, and data produce the same loss curve within small numerical tolerance on the same hardware. |
| N-5 | Readability | Core model code (`model/`) shall stay under ~400 lines total, with docstrings and shape comments. |
| N-6 | Portability | Runs on Linux, macOS, Windows; GPU optional. |
| N-7 | Test coverage | At least 80% line coverage on `minigpt/model`, `tokenizer.py`, `dataset.py`. |
| N-8 | Documentation | README quickstart works end-to-end from a clean clone in under 15 minutes (excluding training time). |

---

## 6. Hyperparameter Specifications

### 6.1 Default `small` config (`configs/small.yaml`)

```yaml
model:
  vocab_size: 65        # set automatically from tokenizer
  block_size: 256
  n_layer: 6
  n_head: 6
  n_embd: 384
  dropout: 0.2
  bias: false
  tie_weights: true

train:
  batch_size: 64
  grad_accum_steps: 1
  max_steps: 5000
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  warmup_steps: 100
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  dtype: bfloat16       # or float16 / float32
  compile: false
  eval_interval: 250
  eval_iters: 100
  seed: 1337

data:
  path: data/processed
  tokenizer: char       # char | bpe

sample:
  max_new_tokens: 300
  temperature: 0.8
  top_k: 40
  top_p: 1.0
```

### 6.2 Scaling guidance

- Tokens per step = `batch_size * block_size * grad_accum_steps`.
- For tiny datasets (about 1 MB of text), expect overfitting past a few thousand steps; use dropout 0.2 and early stopping on val loss.
- For larger datasets, reduce dropout toward 0 and raise `max_steps`.

---

## 7. Testing Strategy

### 7.1 Unit tests

| Test | What it checks |
|---|---|
| `test_tokenizer.py` | Round-trip encode/decode, unknown character handling, save/load equality. |
| `test_model_shapes.py` | Output shapes `[B, T, V]` for various `B`, `T <= block_size`; error when `T > block_size` in forward. |
| `test_causality.py` | Change a future token; confirm logits at earlier positions are unchanged (tolerance 1e-6). |
| `test_attention.py` | Manual attention matches `scaled_dot_product_attention` output within tolerance. |
| `test_config.py` | Divisibility validation, YAML loading, defaults. |
| `test_dataset.py` | `y` is `x` shifted by one; batches stay within bounds. |
| `test_generate.py` | Greedy determinism; seeded sampling reproducibility; context cropping. |

### 7.2 Integration tests

| Test | What it checks |
|---|---|
| `test_overfit_batch.py` | Model can drive loss near zero on a single small batch within a few hundred steps (catches wiring bugs). |
| `test_checkpoint_resume.py` | Train N steps, save, resume, train M steps equals training N+M steps straight (within tolerance). |
| `test_initial_loss.py` | At init, loss ≈ `ln(vocab_size)` (for V = 65, about 4.17). |

### 7.3 Sanity checklist before long training runs
1. Initial loss ≈ `ln(V)`.
2. Overfit a single batch to near zero.
3. Causality test passes.
4. Gradient norms are finite and not exploding.
5. Validation loss decreases during the first few hundred steps.

---

## 8. Acceptance Criteria and Metrics

### 8.1 Model quality (TinyShakespeare, `small` config, char-level)

| Metric | Minimum | Target |
|---|---|---|
| Validation loss | < 1.65 | < 1.50 |
| Perplexity (char-level) | < 5.2 | < 4.5 |
| Qualitative | Generated text uses correct character names, line structure, and mostly real words | Coherent short phrases with Shakespeare-like cadence |

(Values are typical for this setup and should be recorded from the actual run; adjust if a different dataset is used.)

### 8.2 Engineering

| Criterion | Pass condition |
|---|---|
| All unit and integration tests | Pass in CI |
| Lint and format | Clean |
| Fresh-clone quickstart | Succeeds |
| Resume from checkpoint | Verified |
| Documentation | README, architecture, TRD, PRT complete |

---

## 9. Development Plan and Milestones

| Phase | Deliverable | Est. time |
|---|---|---|
| 0. Setup | Repo, env, CI, config system | 0.5 day |
| 1. Data | Download data, char tokenizer, `.bin` pipeline, batch sampler, tests | 1 day |
| 2. Baseline | Bigram model, training loop, loss logging | 1 day |
| 3. Transformer core | Self-attention, multi-head, MLP, block, GPT, tests | 2 to 3 days |
| 4. Training polish | AdamW groups, LR schedule, AMP, checkpoint/resume, eval + sampling | 1 to 2 days |
| 5. Full run | Train `small`, tune, record curves and samples | 1 to 2 days |
| 6. Upgrades (pick any) | BPE, RoPE/RMSNorm, KV cache, SDPA | 2 to 4 days |
| 7. Demo + docs | Gradio app, README, final report, loss plots | 1 to 2 days |
| 8. Stretch | LoRA fine-tune of a pretrained HF model; HF export | optional |

Total: roughly 2 to 3 weeks part-time.

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Subtle causal-mask bug leaks future tokens (loss looks too good) | Medium | High | Dedicated causality test |
| Training instability (loss spikes, NaN) | Medium | Medium | Pre-LN, warmup, grad clipping, bf16, lower LR |
| Overfitting on small dataset | High | Medium | Dropout, weight decay, early stop, more data |
| Not enough compute | Medium | Medium | Use `tiny`, Colab/Kaggle, gradient accumulation, shorter block size |
| Scope creep | High | Medium | Phase gates; upgrades are optional |
| Tokenizer/data leakage between splits | Low | Medium | Fit tokenizer on train only; split first |
| Irreproducible results | Medium | Low | Seeds, pinned deps, saved configs |

---

## 11. Coding Standards

- PEP 8, formatted with `black`, linted with `ruff`.
- Type hints on public functions; docstrings with tensor shapes, e.g. `x: [B, T, C]`.
- No global mutable state; pass config explicitly.
- Keep model code dependency-light (only `torch`).
- Commit messages: imperative mood, one logical change each.
- CI (GitHub Actions): install, lint, run `pytest` on CPU with the `tiny` config.

---

## 12. Deliverables

1. Source code repository as in `architecture.md`, Section 7.
2. Trained checkpoint(s) and tokenizer file.
3. Training logs and loss curve plot (train vs val).
4. Sample generations at several temperatures.
5. Documentation: `README.md`, `architecture.md`, `TRD.md`, `PRT.md`.
6. (Optional) Demo app and Hugging Face model card.

---

## 13. Out of Scope

- Pretraining at billions-of-tokens scale
- RLHF / instruction alignment
- Production serving infrastructure
- Multimodal inputs
- Safety/content moderation systems

---

## 14. Open Questions

1. Which dataset: TinyShakespeare (default), a custom domain corpus, or a regional-language corpus?
2. Is a downstream fine-tuning task required, or is a from-scratch language model enough?
3. Compute available: local GPU, Colab, or CPU only?
4. Are there course or submission format requirements (report length, presentation, demo)?
