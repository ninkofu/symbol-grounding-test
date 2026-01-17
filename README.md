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

Auto-generate a mask during editing (no mask file required):

```
python -m symbol_grounding.scripts.edit_inpaint \
  --image inputs/base.png \
  --prompt "a blue cat" \
  --base-prompt "a cat on a table" \
  --auto-mask-from-prompt \
  --target obj1 \
  --out outputs/
```

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
