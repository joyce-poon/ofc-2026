# Memory

## Project
OFC 2026 Workshop — interactive marimo notebook on energy-efficient, bandwidth-dense co-packaged optics for AI scale-up. Deployed as WASM HTML to GitHub Pages.

## Deployment
| What | URL |
|------|-----|
| **Notebook** | joyce-poon.github.io/ofc-2026 |
| **Slides** | joyce-poon.github.io/ofc-2026/slides |
| **Repo** | github.com/joyce-poon/ofc-2026 (private) |
| **GitHub Pages source** | `docs/` directory on `main` branch |

## Key Files
| File | Purpose |
|------|---------|
| `ofc_workshop.py` | Main marimo notebook |
| `layouts/ofc_workshop.css` | Custom CSS |
| `layouts/ofc_workshop.slides.json` | Slides layout config |
| `layouts/head.html` | Custom HTML head |
| `docs/index.html` | WASM export (notebook view) |
| `docs/slides/index.html` | WASM export (slides view) |
| `test_calculations.py` | Pytest suite (95 tests) |

## Export Workflow
1. Export main: `marimo export html-wasm ofc_workshop.py -o docs --mode run -f`
2. Add `layout_file="layouts/ofc_workshop.slides.json"` to `marimo.App()` config
3. Export slides: `marimo export html-wasm ofc_workshop.py -o docs/slides --mode run -f`
4. Revert `layout_file` from notebook source
5. Copy assets: `cp -r images/ docs/slides/images/ && cp -r layouts/ docs/slides/layouts/`
6. Add bot-blocking meta tags to both `index.html` files
7. Commit & push

## Terms
| Term | Meaning |
|------|---------|
| **CPO** | Co-Packaged Optics |
| **MRM** | Micro-Ring Modulator |
| **EAM** | Electro-Absorption Modulator |
| **MZM** | Mach-Zehnder Modulator |
| **pJ/b** | picojoules per bit (energy metric) |
| **WASM** | WebAssembly (browser-based notebook export) |
| **NVLink** | NVIDIA GPU interconnect |
| **OPTICS_CONFIGS** | Dict of optical link configs (no hardcoded pJ/b; computed dynamically) |
| **computed_pjb** | Reactive variable from energy calculator → feeds What-If planner |

## Gotchas
- `layout_file` must be set in `marimo.App()` for slides to appear in WASM export — without it, `inline_layout_file()` has nothing to inline
- EAM λ-MUX power = λ-DEMUX power (linked via slider, label is `λ-(DE)MUX`)
- Image paths need WASM compatibility: `"pyodide" in sys.modules` check for `ASSET_DIR`
- SymPy can't simplify `erfc(erfcinv(x))` to `x` — Axiomatic reports `correct=False` but equations are numerically correct
