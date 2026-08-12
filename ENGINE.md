# Kronos Core Engine

How the Kronos foundation model actually works, end to end. This covers the two files that
make up the engine — `model/module.py` (building blocks) and `model/kronos.py`
(tokenizer, base model, sampling, autoregressive inference, predictor) — and how they
fit together to turn daily OHLCV bars into a multi-day forecast.

Kronos is a **causal Transformer trained over discretized market data**. It treats the
market like a language: a stock's price history is encoded into a sequence of discrete
tokens, the transformer learns to predict the next token, and forecasting means
autoregressively sampling new tokens and decoding them back into prices.

```
        normalized OHLCV bars
                │
        ┌───────▼────────┐
        │ KronosTokenizer │   encode → (s1_id, s2_id) per bar
        │  (BSQ VQ-VAE)   │   ── coarse token, fine token ──▶
        └───────┬────────┘
                │  token streams
        ┌───────▼────────┐
        │   Kronos base  │   autoregressive next-token prediction
        │   (causal XL)  │   s1 first, then s2 conditioned on s1
        └───────┬────────┘
                │  sampled tokens
        ┌───────▼────────┐
        │ KronosTokenizer │   decode → normalized OHLCV
        └───────┬────────┘
                │  denormalize with window mean/std
                ▼
        forecast DataFrame (open, high, low, close, volume, amount)
```

---

## 1. Input representation

The predictor accepts a `pandas.DataFrame` with per-day rows. The engine distinguishes:

| Column   | Meaning                              |
|----------|--------------------------------------|
| `open`   | day open price                       |
| `high`   | day high price                       |
| `low`    | day low price                        |
| `close`  | day close price                      |
| `volume` | shares traded (defaults to 0)        |
| `amount` | dollar volume; if absent, computed as `volume * mean(open..close)` |

### Normalization

Before the tokenizer sees anything, `KronosPredictor.predict` z-scores every feature
**per context window** (`model/kronos.py`):

```python
x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
x = (x - x_mean) / (x_std + 1e-5)
x = np.clip(x, -self.clip, self.clip)      # clip = 5
```

The mean/std are computed over the window you feed in. This is important:

- The model **never sees absolute prices** — only "how far is today from the average of
  the last N days, in standard deviations".
- The same window statistics are used to denormalize the predictions afterwards:
  `preds * (x_std + 1e-5) + x_mean`.

Consequence: the model is normalized by **local** context, so a $20 stock and a $2000
stock are represented the same way.

### Time features

Each bar also carries a 5-dim time stamp (`calc_time_stamps`): `minute`, `hour`,
`weekday`, `day`, `month`. For daily bars `minute` and `hour` are always 0, so the
effective time signal is weekday / day-of-month / month.

---

## 2. Tokenizer — `KronosTokenizer` (`model/kronos.py`)

The tokenizer is a **Binary Spherical Quantization (BSQ) VQ-VAE**: a Transformer
autoencoder that compresses each bar's 6 normalized channels into discrete codes.

### Encoder

```
normalized bar (6 channels)
        │
   Linear embed (d_in → d_model)
        │
   encoder TransformerBlocks  (n_enc_layers - 1)
        │
   quant_embed (d_model → codebook_dim)
        │
   BSQuantizer
```

### BSQ quantization (`model/module.py`)

`BSQuantizer(s1_bits, s2_bits, ...)` wraps `BinarySphericalQuantizer` with
`codebook_dim = s1_bits + s2_bits`. Quantization is a **sign operation**:

```python
zhat = +1 if z > 0 else -1        # straight-through estimator, scale 1/sqrt(codebook_dim)
```

So each of the `codebook_dim` latent dimensions becomes a single bit, and each bar is
represented by a `codebook_dim`-bit binary code — i.e. one of `2^20` discrete codes.
The bit code is then read as two separate integer token IDs:

- `s1_id` — from the first `s1_bits` bits (the **coarse / pre** token)
- `s2_id` — from the last `s2_bits` bits (the **fine / post** token)

All released variants use `s1_bits = s2_bits = 10`, giving two vocabularies of 1024 each.
`encode(x, half=True)` returns the pair `(s1_ids, s2_ids)`; `decode(tokens, half=True)`
maps the integer codes back to bit vectors, scales them, and reconstructs normalized
OHLCV through the decoder.

### Decoder

Two parallel branches, both `n_dec_layers - 1` TransformerBlocks + a `head`:

- a **pre branch** fed only the `s1` bits (reconstructs from the coarse code),
- a **full branch** fed the whole codebook (reconstructs from the fine code).

### Training losses

The tokenizer is trained with the VQ-style objective returned by `forward`:

- **reconstruction loss** between the decoded output and the input (implicit in the
  encoder/decoder training),
- **commit loss**: `beta * mean((zq.detach() - z)^2)` — keeps encoder outputs near the
  discrete codes,
- **entropy penalty** `zeta * (gamma0 * persample_entropy - gamma * codebook_entropy)`
  — encourages the codebook to be used evenly instead of collapsing onto a few codes
  (BSQ paper: arXiv:2406.07548).

---

## 3. Base model — `Kronos` (`model/kronos.py`)

The base model is a causal decoder-only Transformer that consumes the *tokenized* market
history and predicts the next token.

### Embedding

`HierarchicalEmbedding` embeds `s1_id` and `s2_id` in two **separate** 1024-vocab
embeddings (each `d_model`), concatenates them, and projects to `d_model`:

```python
s1_emb = emb_s1(s1_ids) * sqrt(d_model)
s2_emb = emb_s2(s2_ids) * sqrt(d_model)
out    = fusion_proj(cat([s1_emb, s2_emb]))
```

`TemporalEmbedding` adds the time features — either fixed sinusoidal embeddings or
learnable embeddings (all released checkpoints use `learn_te=True`). The token
embeddings are summed with the time embeddings before the transformer.

### Transformer stack

Each `TransformerBlock` (`model/module.py`) is a pre-norm block:

```
x → RMSNorm → MultiHeadSelfAttention(RoPE, causal) ──add──▶
x → RMSNorm → SwiGLU FeedForward                   ──add──▶
```

Details that differ from a vanilla transformer:

- **RMSNorm** instead of LayerNorm (`x * rsqrt(mean(x^2) + eps)`).
- **Rotary Positional Embeddings (RoPE)** inside attention — rotation applied to Q/K,
  so relative position is encoded rotationally (no learned position table).
- **SwiGLU feedforward**: `silu(w1(x)) * w3(x)` projected by `w2`.
- **Causal self-attention** via `is_causal=True`.

### Two-stage token prediction (the key idea)

Kronos predicts tokens in a **hierarchy**: the coarse `s1` token first, then the fine
`s2` token *conditioned on the predicted `s1`*. This is why there is a dependency-aware
layer and a dual head.

1. `decode_s1(context)` — one forward pass through the transformer stack, then a head
   projection → `s1_logits` (distribution over the 1024 coarse tokens).
2. Sample (or teacher-force) `s1`, embed it → *sibling embedding*.
3. `DependencyAwareLayer(context, sibling_embed)` — a **cross-attention** block whose
   query is the sibling embedding and whose key/value are the transformer context
   representations:

   ```python
   attn = cross_attn(query=sibling_embed, key=hidden, value=hidden)   # non-causal in eval
   out  = RMSNorm(hidden + attn)
   ```

4. `decode_s2` → `s2_logits` (distribution over the fine tokens) via `DualHead.cond_forward`.

So the model never predicts a bar's 20 bits in one shot — it predicts the coarse 10-bit
prefix, then refines it with the fine 10-bit suffix that is compatible with it.

### Model variants

| Checkpoint        | s1/s2 bits | layers | d_model | heads | ff_dim |
|-------------------|------------|--------|---------|-------|--------|
| `Kronos-mini`     | 10 / 10    | 4      | 256     | 4     | 512    |
| `Kronos-small`    | 10 / 10    | 8      | 512     | 8     | 1024   |
| `Kronos-base`     | 10 / 10    | 12     | 832     | 16    | 2048   |

`KronosPredictor` defaults to `max_context=512` (the window size the small/base
checkpoints were trained with).

---

## 4. Autoregressive inference — `auto_regressive_inference` (`model/kronos.py`)

The generation loop (called from `KronosPredictor.generate`):

1. **Clip** the normalized input to `[-clip, clip]` (`clip=5`).
2. **Replicate** the batch `sample_count` times so several forecast paths are sampled in
   parallel and averaged at the end.
3. **Encode** the context into `(pre_ids, post_ids)` and fill rolling token buffers that
   hold at most `max_context` tokens (older tokens are shifted out once the window is
   full).
4. Loop `pred_len` steps:
   - `decode_s1` on the current window → `s1_logits`
   - `sample_from_logits(s1_logits, T, top_k, top_p)` → `sample_pre`
   - `decode_s2(context, sample_pre)` → `s2_logits`
   - `sample_from_logits(s2_logits, ...)` → `sample_post`
   - append both tokens to the buffers.
5. **Decode** the trailing `max_context` window of tokens back to normalized OHLCV
   through the tokenizer decoder.
6. **Average** the `sample_count` paths and take the last `pred_len` bars.

### Sampling (`sample_from_logits` / `top_k_top_p_filtering`)

- Logits are divided by **temperature** `T`.
- **top-k**: keep the top `k` logits, zero everything else.
- **top-p (nucleus)**: keep the smallest set of tokens whose cumulative probability
  reaches `p`.
- If `sample_logits=False` it takes the argmax instead of `torch.multinomial`.

Because sampling is stochastic, `sample_count` paths are drawn and **averaged** for the
final output — this is the `--samples` knob in `test_predict_stocks.py`. Results are
deterministic given a fixed seed.

---

## 5. The predictor — `KronosPredictor`

### `predict(df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count)`

The public entry point (`model/kronos.py`) used by `test_predict_stocks.py`:

1. Validate the price columns; fill `volume` (0) and `amount`
   (`volume * mean(open..close)`) if missing.
2. Compute time stamps for context (`x_timestamp`) and horizon (`y_timestamp`).
3. Normalize the context window (Section 1), `np.newaxis` → batch 1.
4. `generate(...)` → autoregressive sampling (Section 4).
5. **Denormalize**: `preds * (x_std + 1e-5) + x_mean` — note it uses the *context
   window's* mean/std.
6. Return a `DataFrame` of `open, high, low, close, volume, amount` indexed by
   `y_timestamp`.

### `predict_batch(...)`

The same pipeline vectorized over several series: all series must share the same
context length and `pred_len`; they are normalized with **their own** mean/std, stacked
into one batch, and each comes back with its own normalization inverted.

---

## 6. Practical gotchas (learned the hard way)

- **The first predicted point is anchored to the context window's last bar, not to the
  chart's last price.** The model's day-1 forecast is essentially a 1-step-ahead
  prediction from the last bar it *saw*. If you slice the context to end before the
  latest bar (for example to hold out a test window) but then also forecast into the
  future, the projection will start at the *stale* window's level and appear to gap
  down/up from spot on the chart. **When forecasting beyond the last bar, the context
  must end at the last bar.** (`test_predict_stocks.py --future` was fixed for this.)

- **Predictions regress toward the normalization-window mean.** Because the model is
  trained on z-scored OHLCV, its output is pulled toward "average" behavior. After a
  strong run, the first predicted point will typically sit a few percent below (or
  above, after a sell-off) the last close even with correct context. This is model
  behavior, not a bug; the day-over-day *shape* of the forecast is the useful signal.

- **OHLC consistency is not guaranteed.** Each channel is decoded independently, so a
  predicted bar can show e.g. `open > high`. Plot only what you need (usually `close`).

- **Sampling variance is real.** Small `sample_count` (or 1) produces noisy paths. Use
  `--samples` ≥ 5 for stable charts, and fix `--seed` for reproducibility.

- **CPU inference is slow** (the checkpoints are float32 and run one forward pass per
  generated step). Budget ~1 min per ticker at `lookback=512`/`pred_len=30` on a
  workstation CPU, or pass `--device cuda` when a GPU is available.

---

## File map

| File                | Contents                                                     |
|---------------------|--------------------------------------------------------------|
| `model/kronos.py`   | `KronosTokenizer`, `Kronos`, sampling utilities, `auto_regressive_inference`, `KronosPredictor`, `predict_batch` |
| `model/module.py`   | `BSQuantizer`/`BinarySphericalQuantizer`, `RMSNorm`, `FeedForward`, `RotaryPositionalEmbedding`, `MultiHeadAttentionWithRoPE`, `MultiHeadCrossAttentionWithRoPE`, `HierarchicalEmbedding`, `DependencyAwareLayer`, `TransformerBlock`, `DualHead`, `TemporalEmbedding` |
| `test_predict_stocks.py` | Minimal chart + CSV driver: loads Yahoo CSVs, calls `KronosPredictor.predict`, plots history vs forecast |
