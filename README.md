# Symbol Grounding Test Project

This repository provides a skeleton implementation for a **symbol‑grounded image generation system**.  The goal of this project is to move beyond simple text‑to‑image diffusion and towards a pipeline where users can control **what** is drawn, **where** it appears and **how** it looks using natural language.  Inspired by neuro‑symbolic research and the discussions captured in the provided screenshots, the system is organised into several composable components:

1. **Scene graph extraction** – parse a natural language prompt into a structured description of objects, attributes and relations.
2. **Layout generation** – convert the scene graph into spatial constraints (bounding boxes or masks) that describe where each object should appear.
3. **Diffusion renderer** – given the prompt and layout, synthesise an image that respects the placement constraints.  A thin wrapper around existing diffusion models can be swapped in here.
4. **Object‑centred latent learning** – train an auto‑encoder with slot attention and disentanglement constraints to obtain latent codes for each object.  Separate modules handle nouns (identity), adjectives (style) and verbs (transformations).
5. **Binding and composition** – combine noun, adjective and verb codes via tensor‑product representations to construct complex scenes and to enable fine‑grained editing after the initial generation.

The current codebase focuses on providing **readable, modular scaffolding** rather than a fully trained model.  Each module lives in its own file under the `symbol_grounding` package and includes extensive docstrings to explain the intended behaviour.  Where appropriate, dummy implementations return simple structures so that you can run end‑to‑end tests without heavy dependencies.

## Repository structure

```
symbol_grounding/
├── __init__.py               # Make this a Python package
├── pipeline.py               # High‑level orchestration of the full pipeline
├── scene_graph.py            # Parse text into a structured scene graph
├── layout.py                 # Convert scene graphs into spatial layouts
├── diffusion_interface.py    # Wrap diffusion models for constrained rendering
├── slot_attention.py         # Stub for slot attention autoencoder
├── typed_slots.py            # Separate modules for nouns, adjectives and verbs
├── tpr.py                    # Tensor‑product representation utilities
├── utils.py                  # Shared helper functions and types
└── scripts/
    └── generate_image.py     # Example command‑line entry point
requirements.txt              # Dependencies for the project
```

### Next steps

1. **Fill in the implementations** – The classes and functions in this scaffold include docstrings and type hints but leave the heavy lifting unimplemented.  Replace the `NotImplementedError` exceptions with real logic.
2. **Train the object‑centric models** – Implement training loops in `slot_attention.py` using your favourite deep learning library (e.g. PyTorch).  See the accompanying comments for suggestions on disentanglement and typed slots.
3. **Integrate a real diffusion backend** – The current `diffusion_interface.py` merely illustrates how to call a diffusion model given a prompt and optional layout.  Hook this up to an actual model (e.g. Stable Diffusion via the `diffusers` library) or your own implementation.
4. **Write tests** – To ensure maintainability, add unit tests in a `tests/` directory and use a continuous integration setup to run them.

We hope this codebase serves as a solid starting point for your symbol‑grounded generation experiments.