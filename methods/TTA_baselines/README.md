# TTA Baselines (Tent, EATA, SAR)

This folder contains reusable test-time adaptation (TTA) baselines for image classification:

- **Tent**
- **EATA**
- **SAR**

The implementations are designed to work with:

- Standard classifiers that return `logits`
- Models that return tuples/lists where `logits` are at index `0`

---

## 1) Quick Integration

Import setup helpers:

```python
from test_time.TTA_baselines import setup_tent, setup_eata, setup_sar
```

Wrap your model with a TTA method:

```python
tta_model, adapted_param_names, arch = setup_tent(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    architecture="auto",      # auto | cnn | transformer | hybrid
    layer_selection="auto",   # auto | all | late
)
```

Use it during inference:

```python
# Online adaptation on each forward pass
output = tta_model(images)

# Optional: inference without adaptation
output_no_adapt = tta_model.forward_no_adapt(images)
```

---

## 2) Method-Specific Setup

### Tent

```python
from test_time.TTA_baselines import setup_tent

tent_model, names, arch = setup_tent(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
)
```

### EATA

```python
from test_time.TTA_baselines import setup_eata, compute_fishers

# Optional Fisher regularizer (recommended for anti-forgetting)
fishers = compute_fishers(model, fisher_loader, device)

eata_model, names, arch = setup_eata(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    fishers=fishers,
    fisher_alpha=2000.0,
    e_margin=2.45,   # adjust to num classes / confidence
    d_margin=0.05,
)
```

### SAR

```python
from test_time.TTA_baselines import setup_sar

sar_model, names, arch = setup_sar(
    model,
    lr=1e-3,
    steps=1,
    episodic=False,
    margin_e0=0.4 * 6.9,  # default follows SAR-style setup for ~1000 classes
    reset_constant_em=0.2,
    rho=0.05,
    sam_adaptive=False,
)
```

---

## 3) Architecture-Aware Layer Adaptation

The baselines automatically adapt normalization parameters according to architecture:

- **CNN**: BN/GN/IN affine parameters
- **Transformer**: LayerNorm affine parameters
- **Hybrid**: both sets

Control this behavior with:

- `architecture`: `auto`, `cnn`, `transformer`, `hybrid`
- `layer_selection`: `all`, `late`, `auto`

`layer_selection="auto"` uses:

- `late` for transformer-like models (usually more stable)
- `all` for CNN-like models

---

## 4) Output Handling

These wrappers preserve model output format.

- If your model returns `logits`, wrapper returns `logits`
- If your model returns `(logits, extra1, extra2, ...)`, wrapper returns the full tuple

Internally, adaptation losses always use the extracted `logits`.

---

## 5) Minimal End-to-End Pattern

```python
model.eval()
model.to(device)

tta_model, _, _ = setup_tent(model, lr=1e-3, steps=1)
tta_model.to(device)

for images, _ in test_loader:
    images = images.to(device)
    output = tta_model(images)   # adapts online + predicts
```

---

## 6) Notes

- Use small learning rates (`1e-4` to `1e-3`) for stability.
- `steps > 1` can help adaptation but increases compute.
- `episodic=True` resets model each batch/episode; `False` enables continual adaptation.
- For fair baseline comparisons, keep augmentations and dataloaders identical across methods.
