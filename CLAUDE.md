# CLAUDE.md / AGENTS.md

This file provides guidance when working with code in this repository.

## Project Overview

NeMo Speech — toolkit for training/deploying speech models (ASR, TTS, Speech LLM). Active collections: `asr`, `tts`, `audio`, `speechlm2`, `common`. No Megatron / Megatron Core / Transformer Engine — parallelism is PyTorch-native (DDP, FSDP2, TP/SP via DTensor).

## Build & Install

```bash
pip install -e '.[all]'       # Full dev install
pip install -e '.[asr]'       # ASR only
pip install -e '.[test]'      # With test deps
```

Requires Python 3.10+, PyTorch 2.6+.

## Code Style

- **Line length: 119** (not default 88) — consistent across black, isort, flake8
- Black with `skip_string_normalization = true`
- isort with `profile = black`
- Check: `python setup.py style --scope <path>`
- Fix: `python setup.py style --scope <path> --fix`
- **Incremental reformatting**: most collections are excluded from black (see `extend-exclude` in pyproject.toml). The files are reformatted when somebody makes changes to avoid a single big reformatting PR. Do not reformat files outside your changes.

## Testing

```bash
pytest tests/collections/asr -m "not pleasefixme" -v     # ASR tests, skip broken
pytest tests/collections/tts -m unit -v                  # TTS unit tests
pytest -k "test_name" tests/                             # Single test by name
```

Markers: `unit`, `integration`, `system`, `pleasefixme` (broken — skip), `skipduringci`.

## CI & PRs

- NVIDIA developers: feature branches off `main`; community: fork-based workflow
- CI triggered by adding **"Run CICD"** label to the PR
- E2E nightly tests: only when really needed. Add both **"Run e2e nightly"** and **"Run CICD"** labels
- `skip-linting` / `skip-docs` labels bypass those checks
- Formatting CI auto-commits black/isort fixes back to the PR branch
- CI: GitHub Actions in `.github/workflows/`

## Documentation

Sphinx-based docs live in `docs/source/`. Build with:

```bash
pip install -r requirements/requirements_docs.txt   # one-time setup
make -C docs clean html                              # full rebuild
make -C docs html                                    # incremental rebuild
```

Output goes to `docs/build/html/`. Open `docs/build/html/index.html` to preview locally.

Other useful targets: `make -C docs linkcheck` (verify external links), `make -C docs doctest` (run embedded doctests).

## Training & Inference

Entry-point scripts live under `examples/<collection>/`.

All scripts follow the same Hydra pattern — a `@hydra_runner` decorator points to a YAML config in a nearby `conf/` directory:

```python
@hydra_runner(config_path="conf", config_name="fast-conformer_transducer_bpe")
def main(cfg):
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)
```

Override any config value from the CLI with Hydra syntax: `python script.py model.optim.lr=1e-4 trainer.max_epochs=50`. Browse configs with `ls examples/<collection>/conf/` to see which models and variants are supported.

## Handy Scripts

Utility scripts live under `scripts/`. Key subdirectories: `speech_recognition/`, `speechlm2/`, `speaker_tasks/`, `tokenizers/`, `dataset_processing/`, `asr_language_modeling/`. Browse with `ls scripts/`.

Four frequently used data/training helpers:

- **`scripts/speech_recognition/estimate_duration_bins.py`** — estimate Lhotse dynamic-bucketing duration bins from a manifest or YAML input config. Usage: `python scripts/speech_recognition/estimate_duration_bins.py <input> -b 30 -n 100000`
- **`scripts/speech_recognition/oomptimizer.py`** — find the largest batch size per bucket that fits in GPU memory. Usage: `python scripts/speech_recognition/oomptimizer.py --pretrained-name nvidia/canary-1b` or point to a config with `--config-path`.
- **`scripts/speech_recognition/estimate_data_weights.py`** — compute per-dataset sampling weights from YAML input configs, with optional temperature re-weighting. Usage: `python scripts/speech_recognition/estimate_data_weights.py input.yaml output.yaml -t 0.5`
- **`scripts/speech_recognition/convert_to_tarred_audio_dataset.py`** — shard audio+manifest into tar files. Usage: `python scripts/speech_recognition/convert_to_tarred_audio_dataset.py --manifest_path=m.json --target_dir=./tar --num_shards=512 --max_duration=60.0`

## Architecture

- **Hydra + OmegaConf** for all config management (YAML configs)
- **PyTorch Lightning** for training orchestration
- **Lhotse** (>=1.32.2) for audio data loading
- Collections are semi-isolated domains sharing `nemo.core` and `nemo.collections.common`

## Subdirectory Instructions

Module-specific instructions can be added as `CLAUDE.md` or `AGENTS.md` files in subdirectories.

## Issue Reproduction

When fixing a bug, always:
1. First reproduce the issue with a minimal test case
2. Add the reproduction as a unit test
3. Then fix the issue
4. Verify the test passes

## Forbidden Operations

- Never push directly to `main`
- Never modify `.github/workflows/` without explicit instruction
- Never delete test files without explicit instruction



## 🛡️ Strict Embedding Separation, Zero-Fallback Law & 24/7 Dual-GPU Invariant
- **Reference**: 

### 1. Das Absolute Fallback-Verbot (Zero-Fallback Law)
Unter keinen Umständen, zu keinem Zeitpunkt und aus keinem Grund darf ein Fallback zwischen verschiedenen Embedding-Modellen stattfinden.
* **Geltende Aktion:** Fällt ein Embedding-Modell aus oder ist überlastet, MUSS die Operation sofort hart fehlschlagen () oder die Payload transaktional in einer Queue (NATS/SQLite) verharren, bis das exakte Modell bereit ist.
* **Verboten:** Kein stiller oder dynamischer Modellwechsel (weder Jina -> Gemma noch umgekehrt).

### 2. Warum ein Embedding-Fallback mathematisch & informationstheoretisch unmöglich ist
* **Topologische Inkompatibilität heterogener Vektorräume (Non-Isomorphism):**
  Jedes Modell 	heta: \mathcal{X} 	o \mathbb{R}^D$ projiziert Text in eine spezifische, gelernte Riemannsche Mannigfaltigkeit. Jina v5 (=256$) und EmbeddingGemma (=768$) spannen zwei völlig inkompatible geometrische Räume auf. Die Basisvektoren der semantischen Achsen sind ohne explizite Procrustes-Transformation nicht ausgerichtet.
* **Kollaps der Kosinus-Ähnlichkeit ($	ext{sim} pprox 0$):**
  Wird eine Suchanfrage mit Modell $ berechnet ( = f_B(q)$), während der Dokumentenkorpus mit Modell $ indiziert wurde ( = f_A(d)$), verhält sich das Skalarprodukt mathematisch wie das zweier rein zufälliger Vektoren auf einer hochdimensionalen Einheitssphäre:
  86306\mathbb{E}[	ext{sim}(u, v)] = 0 \quad 	ext{mit Varianz} \quad \sigma^2 = rac{1}{D}86306
  Der Nearest-Neighbor-Algorithmus (HNSW/k-NN) liefert stochastisches Rauschen. Das RAG-System erhält völlig falsche oder irrelevante Kontexte.
* **Irreversible Index-Vergiftung (Index Poisoning):**
  Wird auch nur ein einziger Vektor von Modell $ als 'Fallback' in den Index von Modell $ geschrieben, verunreinigt er die Distanzgraphen und Clusterzentren dauerhaft.
* **Das Gesetz des Fail-Fast:**
  Ein Ausfall muss hart abbrechen ().

### 3. Duale 24/7 Erfassungspflicht (GPU-Only)
* **GPU-Only Mandat:** Es läuft absolut nichts auf der CPU — GPU ONLY (NVIDIA GB10 CUDA) für ausnahmslos jedes Embedding-Modell.
* **24/7 Parallelität:** Sowohl  (256D, ~4,1 GB VRAM) als auch  (768D, ~1,2 GB VRAM) laufen dauerhaft 24/7 im VRAM (Summe ~5,3 GB VRAM).
* **Duale Erfassung:** Jeder zu indizierende Text/Chunk wird immer von beiden Modellen parallel eingebettet und getrennt persistiert.

### 4. Idioten- & Failsafe-Sicherung auf Datenbankebene
* **SQLite Schema CHECK-Constraints:**
   und  erzwingen atomare Abbrüche auf Engine-Ebene bei Modell-Mismatches.
* **Qdrant Collection Constraints:**
  Strikte Trennung in separate Collections ( vs ) mit fixierter Vektordimension.


## ⚡ High-Quality Systems Programming Languages Priority (No-Python Policy)
- **Reference**: `/home/m1st/.agents/rules/RULE_High_Quality_Systems_Programming_Languages.md`
- **Rule**:
  1. **Bevorzugte Sprachen:** High Quality **Golang (Go), Rust, C++, Zig, PowerShell, C** sind IMMER und AUSNAHMSLOS die bevorzugten Programmiersprachen.
  2. **Kein Python:** Python ist für neue Daemons, Watcher, Automatisierungen, APIs, CLI-Tools und Dienste strikt untersagt (GIL-Bottlenecks, Dependency-Drift, Speicherineffizienz).
  3. **Natives Systems-Engineering:** Alle Hintergrunddienste, Caching-Ebenen und Task-Runner müssen als native, speichersichere und nebenläufige Binaries kompiliert werden.
