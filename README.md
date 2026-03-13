# Energy-Efficient, Bandwidth-Dense Co-Packaged Optics for AI Scale-Up

Interactive [marimo](https://marimo.io) notebook for OFC 2026 Workshop: *Chasing the Limit*.

## Run

```bash
uv run marimo run ofc_workshop.py --sandbox
```

Or to edit:

```bash
uv run marimo edit ofc_workshop.py --sandbox
```

No manual dependency installation needed — `uv run --sandbox` reads the inline PEP 723 metadata and sets up an isolated environment automatically.

## Slides

The notebook is configured with a slides layout. Use marimo's presentation mode (full-screen slides) for the talk.

## Contents

- **Scale-up interconnect requirements** — GPU growth, bandwidth density, power
- **Energy efficiency calculator** — interactive pJ/b breakdown for MRM and EAM configurations
- **Optical link budget** — waterfall chart with tunable coupling, routing, and channel losses
- **Latency breakdown** — stacked bar chart with per-block scaling models
- **Modulator comparison** — MZM, MRM, and EAM architectures
- **MRM WDM link examples** — 16-channel demonstrations
- **Toward 400–448G** — Si PN junction bandwidth analysis
