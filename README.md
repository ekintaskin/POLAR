# dmri-protocol-opt

Diffusion MRI protocol design for reliable parameter recovery: a unified framework across biophysical models

## How it works

```
       ┌───────────────── CMA-ES proposes candidate protocols ◄──────────────┐
       ▼                                                                     │
  for each substrate and noise realization:                                  │
    simulate signal ──► add Rician noise ──► fit the model ──► squared error │
       │                                                                     │
       └───────────────── mean error over all of them = loss ────────────────┘
```

## Installation

Python >= 3.10

```bash
git clone https://github.com/ekintaskin/dmri-protocol-opt.git
cd dmri-protocol-opt
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick start

### From the command line

Copy a configuration from [examples/configs](examples/configs), and set your scanner
limits, noise level and tissue parameters. Then run:

```bash
# A small run that checks everything works
dmri-protocol-opt optimize examples/configs/cexi_quick.yaml --output-dir runs/quick

# Evaluate the result; repeat --protocol to compare several under matched noise
dmri-protocol-opt evaluate examples/configs/cexi_quick.yaml \
    --protocol optimized=runs/quick/best_protocol.csv
```

A protocol CSV needs the columns `delta_ms`, `Delta_ms` and `G_mT_per_m`, with one row
per shell.

### From Python

```python
import dmri_protocol_opt as dpo

def main():
    model = dpo.CEXI(T2=60e-3)
    substrates = dpo.make_substrate_grid(
        {"R": [3, 8], "Di": [1e-9], "De": [2e-9], "f": [0.5], "kappa": [5, 30]}
    )

    result = dpo.optimize_protocol(
        model,
        substrates,
        n_shells=8,
        constraints=dpo.ScannerConstraints(G_bounds=(0.010, 0.080), te_max=100e-3),
        noise=dpo.RicianNoise(snr=20, te_ref=70e-3, t2=60e-3),
        loss_parameters=["R", "Di", "De", "f", "kappa"],
        seed=1,
        output_dir="runs/my_protocol",
    )
    print(result.protocol.as_dict())

if __name__ == "__main__": 
    main()
```

All five CEXI parameters are always fitted. `loss_parameters` sets which of them count
in the loss, so you can list all of them, as here, or only the ones you care about.

## What you configure

| Setting | Meaning | Default |
|---|---|---|
| `delta_bounds`, `big_delta_bounds` | Range of δ and Δ | 5–40 ms, 5–50 ms |
| `G_bounds` | Gradient strength range | 10–180 mT/m |
| `min_gap` | Minimum Δ − δ | 3 ms |
| `te_overhead` | TE = δ + Δ + overhead | 10 ms |
| `te_max` | Longest TE | 120 ms |
| `b_max` | Largest b-value | none |
| `snr`, `te_ref`, `t2` | Noise level: σ = exp(−TE_ref / T2) / SNR | 20, 70 ms, 60 ms |
| `n_directions` | Directions averaged per shell | 20 |


## CEXI parameters

| Parameter | Meaning | Unit | Default fit range |
|---|---|---|---|
| `R` | Sphere radius | µm | 1–15 |
| `Di` | Intracellular diffusivity | m²/s | 0.3–2 × 10⁻⁹ |
| `De` | Extracellular diffusivity | m²/s | 1–3.5 × 10⁻⁹ |
| `f` | Intracellular volume fraction | | 0.3–0.85 |
| `kappa` | Membrane permeability | µm/s | 0.001–100 |


## Outputs

An optimization's output directory contains:

| File | Contents |
|---|---|
| `best_protocol.csv` | The final protocol |
| `result.json` | Final loss, stop reason and seed |
| `metadata.json` | All settings and package versions |
| `trace.jsonl` | Best protocol so far, per generation; survives an interrupted run |
| `population.jsonl` | Every candidate and its loss |


## Citation

If you use this software, please cite:

> Taskin E, Rafael-Patiño J, Canales-Rodríguez EJ, Thiran JP. Diffusion MRI protocol
> design for reliable parameter recovery: a unified framework across biophysical models.
> *arXiv preprint* arXiv:2610.28099. 2026. https://arxiv.org/abs/2610.28099


## License

Copyright 2026 Ekin Taskin. Apache License 2.0; see [LICENSE](LICENSE). The CEXI
implementation derives from code under the MIT License.
