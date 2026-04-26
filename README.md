# Handwritten Digit Generator — Conditional GAN

A small **Conditional GAN** (PyTorch) that generates MNIST-style handwritten digits on demand. Pick a digit 0–9 and the model produces 5 fresh samples in the requested class.

A trained checkpoint (`generator.pth`) is included, and the demo runs as a Streamlit app.

---

## What it does

You pass a class label (`0` through `9`) into the generator alongside random noise, and it returns synthetic 28×28 grayscale images that resemble handwritten digits of that class.

- **Conditional generation** — class label is concatenated with the noise vector via an embedding layer
- **Same shape as MNIST** (28×28, 1 channel)
- **Pre-trained checkpoint included**, no training needed to try it
- **Streamlit UI** for interactive sampling

## How it works

The generator is a four-layer MLP with `LeakyReLU` activations and a `Tanh` output:

```
input:  [noise (100-d)] ⊕ [label_embedding (10-d)]      shape: (B, 110)
       │
       ├── Linear  →  256   → LeakyReLU(0.2)
       ├── Linear  →  512   → LeakyReLU(0.2)
       ├── Linear  → 1024   → LeakyReLU(0.2)
       └── Linear  → 28·28  → Tanh
output: image                                             shape: (B, 1, 28, 28)
```

| Hyperparameter | Value |
| --- | --- |
| Latent dim | 100 |
| Num classes | 10 |
| Image size | 28 × 28 |
| Channels | 1 |
| Output activation | Tanh (range `[-1, 1]`, un-normalized to `[0, 1]` for display) |

## Run locally

```bash
git clone https://github.com/Harihara04sudhan/digit-generator-app.git
cd digit-generator-app
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`), pick a digit, and click **Generate Images**.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit app + Generator architecture + sampling loop |
| `generator.pth` | Trained generator weights (loaded with `map_location=cpu`) |
| `requirements.txt` | `streamlit`, `torch`, `torchvision` |
| `.devcontainer/` | VS Code dev container config |

## Notes

- The architecture in `app.py` must match the architecture used at training time — if you retrain with different layer sizes, update `app.py` accordingly before loading the checkpoint.
- CPU-only by default (`map_location=cpu`). Move to GPU by changing the `torch.load` call and the `noise` / `labels` tensor placement.
- The display step un-normalizes from `[-1, 1]` to `[0, 1]` (`x * 0.5 + 0.5`) before passing tensors to `st.image`.

## License

MIT — see `LICENSE`.
