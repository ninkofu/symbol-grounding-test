# Symbol Grounding Test Project

This repository provides a skeleton implementation for a **symbol‑grounded image generation system**. The goal of this project is to move beyond simple text‑to‑image diffusion and towards a pipeline where users can control **what** is drawn, **where** it appears and **how** it looks using natural language. Inspired by neuro‑symbolic research and the discussions captured in the provided screenshots, the system is organised into several composable components:

1. **Scene graph extraction** – parse a natural language prompt into a structured description of objects, attributes and relations.
2. **Layout generation** – convert the scene graph into spatial constraints (bounding boxes or masks) that describe where each object should appear.
3. **Diffusion renderer** – given the prompt and layout, synthesise an image that respects the placement constraints.  A thin wrapper around existing diffusion models can be swapped in here.
4. **Object‑centred latent learning** – train an auto‑encoder with slot attention and disentanglement constraints to obtain latent codes for each object.  Separate modules handle nouns (identity), adjectives (style) and verbs (transformations).
5. **Binding and composition** – combine noun, adjective and verb codes via tensor‑product representations to construct complex scenes and to enable fine‑grained editing after the initial generation.

The current codebase focuses on providing **readable, modular scaffolding** rather than a fully trained model. Each module lives in its own file under the `symbol_grounding` package and includes docstrings to explain the intended behaviour. Where appropriate, dummy implementations return simple structures so that you can run end‑to‑end tests without heavy dependencies.

## Repository structure

```
symbol_grounding/
├── __init__.py                     # Package entry point
├── pipeline.py                     # Dummy (Pillow) end-to-end pipeline
├── pipeline_diffusion.py           # Diffusers pipeline orchestrator (optional)
├── scene_graph.py                  # Parse text into a structured scene graph
├── layout.py                       # Convert scene graphs into spatial layouts
├── diffusion_interface.py          # Dummy renderer (Pillow)
├── backends/
│   └── diffusers_backend.py        # Compatibility wrapper (diffusion/)
├── diffusion/
│   ├── conditioning.py             # Layout -> control image utilities
│   └── diffusers_backend.py        # Stable Diffusion + ControlNet backend
├── slot_attention.py               # Stub for slot attention autoencoder
├── typed_slots.py                  # Separate modules for nouns, adjectives and verbs
├── tpr.py                          # Tensor-product representation utilities
├── utils.py                        # Shared helper functions and types
└── scripts/
    ├── generate_image.py           # Dummy pipeline CLI
    └── generate_diffusion.py       # Diffusers pipeline CLI
requirements.txt              # Dependencies for the project
```

## Quickstart

Install dependencies:

```
pip install -r requirements.txt
```

Using uv:

```
uv sync
```

Run the placeholder (Pillow) pipeline:

```
python -m symbol_grounding.scripts.generate_image --prompt "a red cat on a table" --output-dir outputs
```

With uv:

```
uv run -m symbol_grounding.scripts.generate_image --prompt "a red cat on a table" --output-dir outputs
```

Run the System 2 symbolic pipeline (scene planning → layout → rendering):

```
python -m symbol_grounding.scripts.run_system2_pipeline \
  --prompt "a lonely room with a table and a cat" \
  --output-dir outputs
```

Example edits (move + attribute changes + attention locks):

```
python -m symbol_grounding.scripts.run_system2_pipeline \
  --prompt "a lonely room with a flower on a table" \
  --move obj1:0.1:0.0 \
  --set-attr obj1:color=blue \
  --lock-token table \
  --output-dir outputs
```

Run the diffusers pipeline (GPU recommended, first run downloads model weights):

```
python -m symbol_grounding.scripts.generate_diffusion --prompt "a red cat on a table" --out outputs/
```

With uv:

```
uv run -m symbol_grounding.scripts.generate_diffusion --prompt "a red cat on a table" --out outputs/
```

The generated image filename includes a timestamp and seed. A control image is also saved next to the output.

ControlNet (scribble-style conditioning from the layout wireframe):

```
python -m symbol_grounding.scripts.generate_diffusion \
  --prompt "a red cat on a table" \
  --out outputs/ \
  --controlnet-model lllyasviel/sd-controlnet-scribble \
  --control-mode scribble
```

Use `--control-mode seg` for a simple segmentation-style control image.

## Editing (inpainting)

Edit a masked region using an inpainting pipeline:

```
python -m symbol_grounding.scripts.edit_inpaint \
  --image inputs/base.png \
  --mask inputs/mask.png \
  --prompt "a blue cat" \
  --base-prompt "a cat on a table" \
  --out outputs/
```

With uv:

```
uv run -m symbol_grounding.scripts.edit_inpaint \
  --image inputs/base.png \
  --mask inputs/mask.png \
  --prompt "a blue cat" \
  --base-prompt "a cat on a table" \
  --out outputs/
```

Generate a mask from a prompt-derived layout:

```
python -m symbol_grounding.scripts.make_mask_from_layout \
  --prompt "a red cat on a table" \
  --target obj1 \
  --out outputs/mask.png
```

List candidate objects from the prompt:

```
python -m symbol_grounding.scripts.make_mask_from_layout \
  --prompt "a red cat on a table" \
  --list-objects
```

Mask quality options (pad + blur):

```
python -m symbol_grounding.scripts.make_mask_from_layout \
  --prompt "a red cat on a table" \
  --target obj1 \
  --out outputs/mask.png \
  --mask-pad-px 8 \
  --mask-blur 8
```

Auto-generate a mask during editing (no mask file required):

```
python -m symbol_grounding.scripts.edit_inpaint \
  --image inputs/base.png \
  --prompt "a blue cat" \
  --base-prompt "a cat on a table" \
  --auto-mask-from-prompt \
  --target obj1 \
  --mask-pad-px 8 \
  --mask-blur 8 \
  --strength 0.3 \
  --out outputs/
```

## Locality evaluation

Compute mask leakage metrics between before/after images:

```
python -m symbol_grounding.scripts.eval_locality \
  --before outputs/base.png \
  --after outputs/edited.png \
  --mask outputs/mask.png \
  --out outputs/metrics.json \
  --save-diff outputs/diff.png
```

With uv:

```
uv run -m symbol_grounding.scripts.eval_locality \
  --before outputs/base.png \
  --after outputs/edited.png \
  --mask outputs/mask.png \
  --out outputs/metrics.json
```

Interpret leakage using a **null baseline** (edit_prompt empty) and compare the adjusted metrics
(outside_edit - outside_null). The experiment harness writes both metrics and the aggregator computes *_adj.

## Semantic evaluation

Evaluate semantic success on the edited region using CLIP:

```
uv run -m symbol_grounding.scripts.eval_semantic \
  --after outputs/edited.png \
  --mask outputs/mask.png \
  --text "a blue cat" \
  --out outputs/semantic.json
```

Semantic evaluation reuses cached CLIP models during batch runs for speed.

Optional margin (positive vs negative text):

```
uv run -m symbol_grounding.scripts.eval_semantic \
  --after outputs/edited.png \
  --mask outputs/mask.png \
  --text "a blue cat" \
  --neg-text "a red cat" \
  --out outputs/semantic.json
```

## Disentangled VAE (synthetic shapes)

Train a beta-VAE with explicit shape/color/position partitions using synthetic shapes:

```
python -m symbol_grounding.train.disentangled_vae_train --config configs/disentangled_vae.json
```

With uv:

```
uv run -m symbol_grounding.train.disentangled_vae_train --config configs/disentangled_vae.json
```

Training writes reconstructions and latent traversals to `outputs/disentangled_vae/samples`.

Evaluate a saved checkpoint:

```
python -m symbol_grounding.scripts.eval_disentangled_vae \
  --checkpoint outputs/disentangled_vae/checkpoints/model_final.pt \
  --config configs/disentangled_vae.json \
  --out outputs/disentangled_vae/metrics.json
```

With uv:

```
uv run -m symbol_grounding.scripts.eval_disentangled_vae \
  --checkpoint outputs/disentangled_vae/checkpoints/model_final.pt \
  --config configs/disentangled_vae.json \
  --out outputs/disentangled_vae/metrics.json
```

## Disentangled VAE (real images)

Train on real images from a folder (unsupervised attributes):

```
python -m symbol_grounding.train.disentangled_vae_train --config configs/disentangled_vae_real.json
```

The dataset expects images under `inputs/real_images`.

## Grounded diffusion with attention locks

Run diffusion with layout control and optional Prompt-to-Prompt attention locking:

```
python -m symbol_grounding.scripts.generate_grounded_diffusion \
  --prompt "a red cat on a table" \
  --base-prompt "a cat on a table" \
  --lock-token cat \
  --controlnet-model lllyasviel/sd-controlnet-scribble \
  --control-mode scribble \
  --out outputs/
```

## LLM planner → grounded diffusion

Use an LLM to produce a scene graph before layout + diffusion. If `OPENAI_API_KEY`
is not set, the planner falls back to the rule-based parser.

```
python -m symbol_grounding.scripts.generate_llm_grounded_diffusion \
  --prompt "a red cat on a table" \
  --out outputs/ \
  --llm-model gpt-4o-mini
```

## Slot + Typed Slots + TPR pipeline

Run the integration pipeline on random inputs to verify the module wiring:

```
python -m symbol_grounding.scripts.run_slot_tpr_pipeline --batch 2 --image-size 64 --num-slots 7
```

## Experiment harness

Run a batch experiment from a JSON config:

```
uv run -m symbol_grounding.scripts.run_experiment \
  --config configs/experiments/locality_demo.json \
  --out outputs/experiments/
```

Aggregate metrics (baseline-aware):

```
uv run -m symbol_grounding.scripts.aggregate_locality \
  --results outputs/experiments/<experiment_dir>
```

Strength sweep: use `strength_list` in the experiment config. The aggregator will emit
`summary_by_strength` with baseline-adjusted metrics.

Semantic metrics can be enabled via:

```
"eval": {
  "threshold": 0.0392156862745098,
  "semantic": { "enabled": true, "model_id": "openai/clip-vit-base-patch32", "device": "auto", "pad_px": 8 }
}
```

Semantic baselines use the SAME text for edited/null by default. You can override per edit:

```
{
  "edit_prompt": "a blue cat",
  "semantic_text": "a blue cat",
  "semantic_neg_text": "a red cat"
}
```

The aggregator will add `clip_similarity_delta_raw`, `clip_similarity_delta_clipped`
(and `clip_margin_*` if neg text is provided), and include them in `summary_by_strength`.
It also emits a `pareto_front` list for leakage vs semantic gain trade-offs.

Semantic-enabled demo config:

```
uv run -m symbol_grounding.scripts.run_experiment \
  --config configs/experiments/locality_semantic_demo.json \
  --out outputs/experiments/
```

For more stable edits, you can split prompts:
`generate_prompt` for base image generation and `edit_base_prompt` for inpaint conditioning.
This avoids contradictions like "red cat" + "blue cat" in the same inpaint prompt.

## Experiment suites

Run multiple experiment configs and auto-aggregate:

```
uv run -m symbol_grounding.scripts.run_suite \
  --configs-dir configs/experiments \
  --out outputs/experiments_suites/suite_YYYYMMDD \
  --pattern "locality_*demo.json" \
  --aggregate
```

Benchmark v1 (color edits):

```
uv run -m symbol_grounding.scripts.run_suite \
  --configs-dir configs/experiments \
  --out outputs/suites/bench_v1 \
  --pattern "bench_v1_color.json" \
  --aggregate
```

Plot suite results (Pareto + strength sweep) and emit CSVs:

```
uv run -m symbol_grounding.scripts.plot_suite \
  --suite-index outputs/experiments_suites/suite_YYYYMMDD/suite_index.json \
  --out outputs/experiments_suites/suite_YYYYMMDD/plots
```

Generate a visual markdown report:

```
uv run -m symbol_grounding.scripts.report_suite \
  --suite-index outputs/suites/bench_v1/suite_index.json \
  --out outputs/suites/bench_v1/report \
  --topk 5
```

Optional flags:

```
uv run -m symbol_grounding.scripts.run_experiment \
  --config configs/experiments/locality_demo.json \
  --out outputs/experiments/ \
  --skip-generate
```

## Typical workflow

1) Generate a base image (`generate_diffusion`)
2) Create a mask (`make_mask_from_layout`) or auto-mask in `edit_inpaint`
3) Edit the image (`edit_inpaint`)
4) Evaluate leakage (`eval_locality`)
5) Aggregate with baseline-adjusted metrics (`aggregate_locality`)
6) Or run all in one batch (`run_experiment`)

## Recommended research workflow

1) `run_suite` (batch configs)
2) `aggregate_locality` (auto when `--aggregate` is set)
3) `plot_suite` (pareto + strength sweep)

## Slot Attention training

Train a minimal Slot Attention autoencoder on synthetic shapes:

```
python -m symbol_grounding.train.slot_ae_train --config configs/slot_ae.json
```

With uv:

```
uv run -m symbol_grounding.train.slot_ae_train --config configs/slot_ae.json
```

Reconstruction samples and slot masks are saved under `outputs/slot_ae/samples/`.

## Multimodal Slot Attention training (image + caption)

Train a minimal multimodal Slot Attention model with text conditioning:

```
python -m symbol_grounding.train.multimodal_slot_ae_train --config configs/multimodal_slot_ae.json
```

Reconstruction samples are saved under `outputs/multimodal_slot_ae/samples/`.

## Grammar VAE training (noun/adj/verb partitions)

Train a grammar-aware VAE with noun/adjective/verb latent partitions:

```
python -m symbol_grounding.train.grammar_vae_train --config configs/grammar_vae.json
```

Reconstructions and checkpoints are written under `outputs/grammar_vae/`.

## Tests

Run the minimal unit tests:

```
python -m unittest discover -s tests
```

### Next steps

1. **Fill in the implementations** – The classes and functions in this scaffold include docstrings and type hints but leave the heavy lifting unimplemented.  Replace the `NotImplementedError` exceptions with real logic.
2. **Train the object‑centric models** – Implement training loops in `slot_attention.py` using your favourite deep learning library (e.g. PyTorch).  See the accompanying comments for suggestions on disentanglement and typed slots.
3. **Expand the diffusion backend** – Add ControlNet, LoRA, or refiner support as needed, and integrate layout-derived conditioning beyond wireframes.
4. **Write tests** – To ensure maintainability, add unit tests in a `tests/` directory and use a continuous integration setup to run them.

We hope this codebase serves as a solid starting point for your symbol‑grounded generation experiments.
