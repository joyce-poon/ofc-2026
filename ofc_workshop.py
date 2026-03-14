# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "plotly>=5.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(
    width="full",
    app_title="Energy-Efficient, Bandwidth-Dense Co-Packaged Optics for AI Scale-Up",
    css_file="layouts/ofc_workshop.css",
    html_head_file="layouts/head.html",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    import math
    from pathlib import Path
    import sys as _sys

    if "pyodide" in _sys.modules:
        class _UrlPath:
            """In WASM, produce string URLs so mo.image fetches via HTTP, not filesystem."""
            def __init__(self, base): self._base = base
            def __truediv__(self, name): return f"{self._base}/{name}"
        IMG = _UrlPath("./images")
    else:
        IMG = Path("images")

    # NVL72/144: 18 NVLink5 ports × 400 Gbps/port; NVL576: 1.5 PB/s total (NVIDIA)
    RACKS = {
        "nvl72": {"name": "GB200 NVL72", "year": 2025, "gpus": 72,
                  "bw_per_gpu_tbps": 18 * 400 / 1000, "rack_power_kw": 140},
        "nvl144": {"name": "Vera Rubin NVL144", "year": 2026, "gpus": 144,
                   "bw_per_gpu_tbps": 18 * 400 / 1000, "rack_power_kw": 210},
        "nvl576": {"name": "Rubin Ultra NVL576", "year": 2027, "gpus": 576,
                   "bw_per_gpu_tbps": 1.5 * 8 * 1000 / 576 / 2, "rack_power_kw": 600},
    }

    TX_PENALTY = {
        ("MRM", 50, "NRZ"): 3.2, ("MRM", 100, "NRZ"): 5.9,
        ("MRM", 100, "PAM4"): 3.4, ("MRM", 200, "PAM4"): 6.2,
        ("EAM", 50, "NRZ"): 3.2, ("EAM", 100, "NRZ"): 5.0,
        ("EAM", 100, "PAM4"): 6.0, ("EAM", 200, "PAM4"): 7.0,
    }

    TDECQ = {
        ("MRM", 50, "NRZ"): 0.5, ("MRM", 100, "NRZ"): 1.0,
        ("MRM", 100, "PAM4"): 2.0, ("MRM", 200, "PAM4"): 3.0,
        ("EAM", 100, "NRZ"): 1.5, ("EAM", 100, "PAM4"): 2.0,
        ("EAM", 200, "PAM4"): 3.5,
    }

    # --- Receiver sensitivity via Personick Q ---
    # Personick Q: Q = (I₁ - I₀) / (σ₁ + σ₀)
    # Thermal-noise-limited (σ₁ = σ₀ = σ): Q = OMA·R / (2σ)
    # Personick noise: σ² = S_th · I₂ · B  (I₂ = Personick integral, B = baud rate)
    #   → P_opt ∝ Q · √B  for same pulse shape
    #
    # BER from Personick Q:
    #   NRZ:  BER = ½ · erfc(Q / √2)
    #   PAM4 (Gray-coded, M=4):
    #     SER = 2(M-1)/M · ½ · erfc(Q/√2) = ¾ · erfc(Q/√2)
    #     Gray coding: each adjacent error flips 1 of log₂(M)=2 bits
    #     BER = SER / log₂(M) = (3/8) · erfc(Q/√2)
    #
    # PAM4 OMA penalty: eye opening = OMA/(M-1) = OMA/3
    #   → need 3× OMA for same Q  →  10·log₁₀(3) ≈ 4.77 dB

    def _personick_q(ber, fmt):
        """Solve for Personick Q given target BER and modulation format.
        Uses bisection on erfc: erfc(Q/√2) = target."""
        if fmt == "NRZ":
            target = 2 * ber            # erfc(Q/√2) = 2·BER
        else:  # PAM4 Gray-coded
            target = 8 * ber / 3        # erfc(Q/√2) = 8·BER/3
        lo, hi = 0.0, 12.0
        for _ in range(80):  # bisection converges in ~60 iterations to machine precision
            mid = (lo + hi) / 2
            if math.erfc(mid) > target:
                lo = mid
            else:
                hi = mid
        return mid * math.sqrt(2)       # Q = x·√2 where erfc(x) = target

    # Reference: 50G NRZ, BER = 1e-15, sensitivity = -12 dBm
    _REF_SENS = -12.0   # dBm
    _REF_BW = 25.0      # GHz  (50 Gbps NRZ → baud/2 = 25 GHz)
    _REF_Q = _personick_q(1e-15, "NRZ")

    def rx_sensitivity(rate_gbps, fmt):
        """Receiver sensitivity (dBm) from Personick Q formalism.
        NRZ at BER = 1e-15, PAM4 at BER = 1e-6, scaled from 50G NRZ reference."""
        ber = 1e-15 if fmt == "NRZ" else 1e-6
        q = _personick_q(ber, fmt)
        bw = rate_gbps / 2 if fmt == "NRZ" else rate_gbps / 4  # Nyquist BW
        pam4_pen = 10 * math.log10(3) if fmt == "PAM4" else 0  # OMA/(M-1) penalty
        return _REF_SENS + pam4_pen + 10 * math.log10(q / _REF_Q) + 5 * math.log10(bw / _REF_BW)

    RX_SENS = {
        (50, "NRZ"): rx_sensitivity(50, "NRZ"),
        (100, "NRZ"): rx_sensitivity(100, "NRZ"),
        (100, "PAM4"): rx_sensitivity(100, "PAM4"),
        (200, "PAM4"): rx_sensitivity(200, "PAM4"),
    }
    WDM_DEMUX = {32: 1.5, 16: 1.0, 8: 0.5}

    # --- EIC + XSR SerDes power model ---
    # SerDes lane power ≈ analog front-end (∝ baud rate) + small PAM4 overhead
    # Reference: 50G NRZ = 2.0 pJ/b at 50 Gbaud
    # At same baud rate, PAM4 adds overhead for multi-level DAC/ADC but gets 2 bits/symbol
    _EIC_REF_PJB = 2.0       # pJ/b at 50G NRZ reference
    _EIC_REF_BAUD = 50.0     # Gbaud reference

    def eic_xsr_pjb(rate_gbps, fmt, pam4_overhead=0.10, ref_pjb=None):
        """EIC+XSR SerDes energy (pJ/b) scaled from reference at 50 Gbaud.
        Lane power ∝ baud rate; PAM4 adds small overhead at same baud."""
        _ref = ref_pjb if ref_pjb is not None else _EIC_REF_PJB
        baud = rate_gbps if fmt == "NRZ" else rate_gbps / 2
        lane_mw = _ref * _EIC_REF_BAUD * (baud / _EIC_REF_BAUD) * (1 + pam4_overhead * (fmt == "PAM4"))
        return lane_mw / rate_gbps

    ENERGY_CONFIGS = [
        {"label": "MRM 32\u03bb\u00d750G NRZ", "mod": "MRM", "n_lam": 32, "rate": 50,
         "total_loss_dB": 18.2, "tdecq": 0.5, "rx_sens": rx_sensitivity(50, "NRZ"), "eam_mux": 0, "eic_pjb": eic_xsr_pjb(50, "NRZ")},
        {"label": "MRM 16\u03bb\u00d7100G NRZ", "mod": "MRM", "n_lam": 16, "rate": 100,
         "total_loss_dB": 20.4, "tdecq": 1.0, "rx_sens": rx_sensitivity(100, "NRZ"), "eam_mux": 0, "eic_pjb": eic_xsr_pjb(100, "NRZ")},
        {"label": "MRM 16\u03bb\u00d7100G PAM4", "mod": "MRM", "n_lam": 16, "rate": 100,
         "total_loss_dB": 17.9, "tdecq": 2.0, "rx_sens": rx_sensitivity(100, "PAM4"), "eam_mux": 0, "eic_pjb": eic_xsr_pjb(100, "PAM4")},
        {"label": "MRM 8\u03bb\u00d7200G PAM4", "mod": "MRM", "n_lam": 8, "rate": 200,
         "total_loss_dB": 20.2, "tdecq": 3.0, "rx_sens": rx_sensitivity(200, "PAM4"), "eam_mux": 0, "eic_pjb": eic_xsr_pjb(200, "PAM4")},
        {"label": "EAM 16\u03bb\u00d7100G NRZ", "mod": "EAM", "n_lam": 16, "rate": 100,
         "total_loss_dB": 19.5, "tdecq": 1.5, "rx_sens": rx_sensitivity(100, "NRZ"), "eam_mux": 20, "eic_pjb": eic_xsr_pjb(100, "NRZ")},
        {"label": "EAM 16\u03bb\u00d7100G PAM4", "mod": "EAM", "n_lam": 16, "rate": 100,
         "total_loss_dB": 20.5, "tdecq": 2.0, "rx_sens": rx_sensitivity(100, "PAM4"), "eam_mux": 20, "eic_pjb": eic_xsr_pjb(100, "PAM4")},
        {"label": "EAM 8\u03bb\u00d7200G PAM4", "mod": "EAM", "n_lam": 8, "rate": 200,
         "total_loss_dB": 21.0, "tdecq": 3.5, "rx_sens": rx_sensitivity(200, "PAM4"), "eam_mux": 20, "eic_pjb": eic_xsr_pjb(200, "PAM4")},
    ]

    # --- Latency model: reference at 50 Gbaud, scale ∝ 1/baud ---
    # --- Latency model: reference at 50 Gbaud, scale as (baud_ref/baud)^α per block ---
    # NRZ: no FEC, no DAC/ADC.  PAM4: lite FEC + DAC + ADC required.
    # Per-block exponent α: 1.0 = ideal (latency ∝ 1/baud), <1.0 = sub-linear (pipeline overhead)
    BAUD_REF = 50  # Gbaud
    LAT_REF = {"Serializer": 4, "Lite FEC": 50, "DAC TX": 1, "ADC RX": 0.5, "RX Eq DSP": 8, "RX DeSer+CDR": 8}
    LAT_EXPONENTS_DEFAULT = {"Serializer": 0.9, "Lite FEC": 0.4, "DAC TX": 0.8, "ADC RX": 0.8, "RX Eq DSP": 0.6, "RX DeSer+CDR": 0.8}

    def latency_for(baud, fmt, exponents):
        """Compute per-block latency (ns) at given baud rate with per-block scaling exponents."""
        result = {}
        for _k, _v in LAT_REF.items():
            _is_pam4_only = _k in ("Lite FEC", "DAC TX", "ADC RX")
            if _is_pam4_only and fmt != "PAM4":
                result[_k] = 0
            else:
                result[_k] = _v * (BAUD_REF / baud) ** exponents[_k]
        return result

    OPTICS_CONFIGS = {
        "mrm_16x100g_nrz":  {"label": "MRM 16\u03bb\u00d7100G NRZ",  "n_lam": 16, "rate": 100, "pjb": 3.89, "fibers_per_dir": 1},
        "mrm_16x100g_pam4": {"label": "MRM 16\u03bb\u00d7100G PAM4", "n_lam": 16, "rate": 100, "pjb": 3.68, "fibers_per_dir": 1},
        "mrm_32x50g_nrz":   {"label": "MRM 32\u03bb\u00d750G NRZ",   "n_lam": 32, "rate": 50,  "pjb": 3.58, "fibers_per_dir": 1},
        "mrm_8x200g_pam4":  {"label": "MRM 8\u03bb\u00d7200G PAM4",  "n_lam": 8,  "rate": 200, "pjb": 4.61, "fibers_per_dir": 1},
        "eam_16x100g_pam4": {"label": "EAM 16\u03bb\u00d7100G PAM4", "n_lam": 16, "rate": 100, "pjb": 4.44, "fibers_per_dir": 2},
        "eam_8x200g_pam4":  {"label": "EAM 8\u03bb\u00d7200G PAM4",  "n_lam": 8,  "rate": 200, "pjb": 5.14, "fibers_per_dir": 2},
    }

    PLOT_FONT = dict(family="Open Sans, Noto Sans, Roboto, sans-serif", size=16, color="#263238")
    PLOT_TITLE_FONT = dict(size=24, color="#1A237E")
    return (
        ENERGY_CONFIGS,
        IMG,
        LAT_EXPONENTS_DEFAULT,
        LAT_REF,
        OPTICS_CONFIGS,
        PLOT_FONT,
        PLOT_TITLE_FONT,
        RACKS,
        RX_SENS,
        TDECQ,
        TX_PENALTY,
        WDM_DEMUX,
        eic_xsr_pjb,
        go,
        latency_for,
        make_subplots,
        math,
        mo,
        np,
        pd,
    )


@app.cell(hide_code=True)
def _(IMG, mo):
    mo.vstack([
        mo.hstack([
            mo.md("""# Energy-Efficient, Bandwidth-Dense Co-Packaged Optics for AI Scale-Up

    ###Joyce Poon

    ####OFC Workshop: Chasing the Limit | March 15, 2026

    _Views expressed are my own and do not represent those of my former employer or any other organization._"""),
            mo.image(src=IMG / "qrcode_joyce-poon.github.io.png", width="150px"),
            mo.image(src=IMG / "uoft_logo.png", width="200px"),
        ], align="center", widths=[0.7, 0.15, 0.15]),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(IMG, mo):
    _left = mo.vstack([
        mo.md("""### Example: NVIDIA GB200 NVL72
    - 72 GPUs + 36 CPUs, ~130 kW per rack
    - BW per GPU: 7.8 Tbps (uni-dir)
    - ~1300 copper cables, 5184 lanes per rack"""),
        mo.md("""### Important metrics:
    - Bandwidth density
    - Power consumption
    - Latency
    - Reliability & link flaps"""),
    ])

    _right = mo.vstack([
        mo.image(src=IMG / "superpod_topology.png", width="90%", caption="NVIDIA DGX SuperPod topology"),
        mo.image(src=IMG / "gb200_hardware.jpeg", width="90%"),
        mo.md('<div style="background:#FFF3E0; border-left:4px solid #E65100; padding:8px 14px; font-size:18px; border-radius:4px;"><b>Bandwidth density matters:</b> the solution needs to fit!</div>'),
    ])
    _layout = mo.hstack([_left, _right], align="start", gap=2)
    mo.vstack([mo.md("## Scale-Up Interconnects"), _layout, mo.md("---")])
    return


@app.cell(hide_code=True)
def _(IMG, PLOT_FONT, PLOT_TITLE_FONT, RACKS, go, mo):
    fig_gpu = go.Figure(
        data=[go.Bar(
            x=[RACKS[k]["year"] for k in ["nvl72", "nvl144", "nvl576"]],
            y=[RACKS[k]["gpus"] for k in ["nvl72", "nvl144", "nvl576"]],
            text=[RACKS[k]["gpus"] for k in ["nvl72", "nvl144", "nvl576"]],
            textposition="outside", textfont=dict(size=18),
            marker_color="#00838F",
            hovertext=[RACKS[k]["name"] for k in ["nvl72", "nvl144", "nvl576"]],
        )]
    )
    fig_gpu.update_layout(
        title=dict(text="GPUs per Rack", font=PLOT_TITLE_FONT),
        xaxis_title="Year", yaxis_title="GPUs per Rack", yaxis_range=[0, 700],
        xaxis=dict(tickmode="array", tickvals=[2025, 2026, 2027]),
        template="plotly_white", height=500, font=PLOT_FONT,
        annotations=[dict(x=2026, y=420, text="<b>8\u00d7 in 2 years</b>", showarrow=False,
                         arrowhead=2, ax=-60, ay=-40, font=dict(size=18, color="#FFFFFF"),
                         bgcolor="#C62828", bordercolor="#C62828", borderpad=6, borderwidth=1)],
    )
    _layout = mo.hstack([
        mo.ui.plotly(fig_gpu),
        mo.vstack([
            mo.image(src=IMG / "rubin_nvl144.jpeg", width="100%", caption="Vera Rubin NVL144 (2H 2026)"),
            mo.image(src=IMG / "rubin_nvl576.jpeg", width="100%", caption="Rubin Ultra NVL576 (2H 2027)"),
        ], gap=1),
    ], justify="space-around", align="center", gap=2, widths=[0.7, 0.3])
    mo.vstack([mo.md("## GPU Growth per Rack"), _layout, mo.md("---")])
    return


@app.cell(hide_code=True)
def _(mo, pd):
    _IO_PJB_LO, _IO_PJB_HI = 10, 20  # pJ/b assumed range

    GPU_SYSTEMS = [
        {"name": "GB NVL72", "gpus": 72, "tech": "Copper",
         "ports_per_gpu": 18, "bw_per_port_gbps": 400,
         "lane_rate_gbps": 224, "rack_power_kw": 140},
        {"name": "Ironwood", "gpus": 64, "tech": "Copper",
         "lanes_per_gpu": 64, "lane_rate_gbps": 112,
         "rack_power_kw": None},
        {"name": "CloudMatrix", "gpus": 32, "tech": "LPO",
         "ports_per_gpu": 7, "bw_per_port_gbps": 400,
         "lane_rate_gbps": 112, "rack_power_kw": 600},
        {"name": "VR NVL144", "gpus": 144, "tech": "Copper",
         "ports_per_gpu": 18, "bw_per_port_gbps": 400,
         "lane_rate_gbps": 224, "rack_power_kw": (190, 230)},
        {"name": "VR NVL576", "gpus": 576, "tech": "Copper?",
         "rack_bw_PBps": 1.5, "lane_rate_gbps": "224?",
         "rack_power_kw": 600},
    ]

    for _s in GPU_SYSTEMS:
        if "ports_per_gpu" in _s:
            _s["bw_gpu_tbps"] = _s["ports_per_gpu"] * _s["bw_per_port_gbps"] / 1000
            _s["bw_rack_bidir_pbps"] = (
                _s["gpus"] * _s["ports_per_gpu"] * _s["bw_per_port_gbps"] * 2 / 1e6)
        elif "lanes_per_gpu" in _s:
            _s["bw_gpu_tbps"] = _s["lanes_per_gpu"] * _s["lane_rate_gbps"] / 1000
            _s["bw_rack_bidir_pbps"] = (
                _s["gpus"] * _s["lanes_per_gpu"] * _s["lane_rate_gbps"] * 2 / 1e6)
        elif "rack_bw_PBps" in _s:
            _s["bw_rack_bidir_pbps"] = _s["rack_bw_PBps"] * 8
            _s["bw_gpu_tbps"] = _s["bw_rack_bidir_pbps"] * 1000 / _s["gpus"] / 2
        _bw_bps = _s["bw_rack_bidir_pbps"] * 1e15
        _s["io_kw_lo"] = _IO_PJB_LO * 1e-12 * _bw_bps / 1000
        _s["io_kw_hi"] = _IO_PJB_HI * 1e-12 * _bw_bps / 1000

        _pw = _s["rack_power_kw"]
        if _pw is None:
            _s["pct_lo"] = _s["pct_hi"] = None
        elif isinstance(_pw, tuple):
            _s["pct_lo"] = _s["io_kw_lo"] / _pw[1] * 100
            _s["pct_hi"] = _s["io_kw_hi"] / _pw[0] * 100
        else:
            _s["pct_lo"] = _s["io_kw_lo"] / _pw * 100
            _s["pct_hi"] = _s["io_kw_hi"] / _pw * 100

    _rows = {}
    for _s in GPU_SYSTEMS:
        _pw = _s["rack_power_kw"]
        if "ports_per_gpu" in _s:
            _io = f'{_s["ports_per_gpu"]} ports'
            _bwl = str(_s["bw_per_port_gbps"])
        elif "lanes_per_gpu" in _s:
            _io = f'{_s["lanes_per_gpu"]} lanes'
            _bwl = str(_s["lane_rate_gbps"])
        else:
            _io, _bwl = "\u2014", "\u2014"

        if _pw is None:
            _pw_str = "\u2014"
        elif isinstance(_pw, tuple):
            _pw_str = f"{_pw[0]}\u2013{_pw[1]}"
        else:
            _pw_str = f"~{_pw}"

        _pct = ("\u2014" if _s["pct_lo"] is None
                else f'{_s["pct_lo"]:.0f}\u2013{_s["pct_hi"]:.0f}%')

        _rows[_s["name"]] = [
            str(_s["gpus"]), _s["tech"], _io, _bwl,
            f'{_s["bw_gpu_tbps"]:.1f}', str(_s["lane_rate_gbps"]),
            f'{_s["bw_rack_bidir_pbps"]:.1f}', _pw_str,
            f'{_s["io_kw_lo"]:.0f}\u2013{_s["io_kw_hi"]:.0f}', _pct,
        ]

    _df = pd.DataFrame({
        "": ["GPUs per rack", "Scale-up tech", "IO links / GPU",
             "BW / link (Gbps)", "BW/GPU (Tbps, uni)",
             "Lane rate (Gbps)", "BW/rack (Pbps, bidir)",
             "Rack power (kW)", f"IO power (kW) @ {_IO_PJB_LO}\u2013{_IO_PJB_HI} pJ/b",
             "% Power for IO"],
        **_rows,
    })
    mo.vstack([
        mo.md("## Bandwidth & Power Requirements"),
        mo.as_html(_df),
        mo.md(f"""**Derivations ({_IO_PJB_LO}\u2013{_IO_PJB_HI} pJ/b assumed):** BW/GPU = IO links \u00d7 BW/link &emsp;|\
    &emsp; BW/rack = GPUs \u00d7 BW/GPU \u00d7 2 (bidir) &emsp;|&emsp; VR NVL576: 1.5 PB/s (NVIDIA) = {1.5*8:.0f} Pbps
    IO power = pJ/b \u00d7 BW/rack &emsp;|&emsp; % Power = IO power \u00f7 rack power"""),
        mo.md('<div style="background:#FFF3E0; border-left:4px solid #E65100; padding:8px 14px; font-size:18px; border-radius:4px;">As GPUs per rack increases, IO could consume <b>20-40%</b> of rack power.</div>'),
        mo.md("---"),
    ])
    return (GPU_SYSTEMS,)


@app.cell(hide_code=True)
def _(mo):
    io_pjb_slider = mo.ui.slider(start=5, stop=25, step=1, value=15, label="Assumed IO energy efficiency (pJ/b)")
    return (io_pjb_slider,)


@app.cell(hide_code=True)
def _(GPU_SYSTEMS, PLOT_FONT, go, io_pjb_slider, make_subplots, math, mo):
    _pjb = io_pjb_slider.value

    _names = [s["name"] for s in GPU_SYSTEMS]
    _total_bw = [s["bw_rack_bidir_pbps"] * 1000 for s in GPU_SYSTEMS]  # Tbps
    _bw_per_gpu = [s["bw_gpu_tbps"] for s in GPU_SYSTEMS]
    _power, _io_pct = [], []
    for s in GPU_SYSTEMS:
        _pw = s["rack_power_kw"]
        if _pw is None:
            _power.append(0)
            _io_pct.append(0)
        elif isinstance(_pw, tuple):
            _pw_mid = (_pw[0] + _pw[1]) / 2
            _power.append(_pw_mid)
            _io_kw = s["bw_rack_bidir_pbps"] * 1e15 * _pjb * 1e-12 / 1000
            _io_pct.append(_io_kw / _pw_mid * 100)
        else:
            _power.append(_pw)
            _io_kw = s["bw_rack_bidir_pbps"] * 1e15 * _pjb * 1e-12 / 1000
            _io_pct.append(_io_kw / _pw * 100)

    _colors = ['#1A237E', '#00838F', '#E65100', '#FFA000', '#2E7D32']

    fig_systems = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>Total Bandwidth</b>",
            "<b>Bandwidth per GPU</b>",
            "<b>Rack Power</b>",
            "<b>IO Power %</b>"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    fig_systems.add_trace(go.Bar(
        x=_names, y=_total_bw, marker=dict(color=_colors),
        text=[f"{bw:.0f}" for bw in _total_bw],
        textposition='outside', textfont=dict(size=14),
    ), row=1, col=1)

    fig_systems.add_trace(go.Bar(
        x=_names, y=_bw_per_gpu, marker=dict(color=_colors),
        text=[f"{bw:.1f}" for bw in _bw_per_gpu],
        textposition='outside', textfont=dict(size=14),
    ), row=1, col=2)

    fig_systems.add_trace(go.Bar(
        x=_names, y=_power, marker=dict(color=_colors),
        text=[f"{p:.0f}" if p > 0 else "N/A" for p in _power],
        textposition='outside', textfont=dict(size=14),
    ), row=2, col=1)

    fig_systems.add_trace(go.Bar(
        x=_names, y=_io_pct, marker=dict(color=_colors),
        text=[f"{p:.1f}%" if p > 0 else "N/A" for p in _io_pct],
        textposition='outside', textfont=dict(size=14),
    ), row=2, col=2)

    def _y_max(vals):
        _m = max(vals) if vals else 1
        _step = 10 ** math.floor(math.log10(max(_m, 1)))
        return math.ceil(_m / _step + 1) * _step

    fig_systems.update_yaxes(title_text="Tbps (bi-dir)", range=[0, _y_max(_total_bw)], row=1, col=1)
    fig_systems.update_yaxes(title_text="Tbps (uni-dir)", range=[0, _y_max(_bw_per_gpu)], row=1, col=2)
    fig_systems.update_yaxes(title_text="kW", range=[0, _y_max(_power)], row=2, col=1)
    fig_systems.update_yaxes(title_text="% of Total", range=[0, _y_max(_io_pct)], row=2, col=2)

    fig_systems.update_layout(
        height=700, showlegend=False, font=PLOT_FONT, template="plotly_white",
    )
    for _ann in fig_systems.layout.annotations:
        _ann.font = dict(size=18, color="#1A237E")

    mo.vstack([
        mo.md("## System Comparison"),
        mo.hstack([io_pjb_slider], justify="start"),
        mo.as_html(fig_systems),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(IMG, mo):
    _left = mo.vstack([
        mo.md("""|  Approach | Description | Energy |
    |:---------|:------------|:------:|
    | **Pluggable Retimed** | ASIC \u2192 LR \u2192 DSP + OE (scale-out) | **15-20 pJ/b** |
    | **Near-Package / LPO** | ASIC \u2192 LR/MR \u2192 OE on substrate | **~10-15 pJ/b** |
    | **Co-Packaged Optics** | OE on substrate next to ASIC, XSR | **~5-10 pJ/b** |
    | **Optical Interposer** | OE on interposer under ASIC | **~3-5 pJ/b** |"""),
        mo.md('<div style="background:#E8F5E9; border-left:4px solid #2E7D32; padding:8px 14px; font-size:18px; border-radius:4px;">Moving optics closer to the ASIC reduces DSP and improves signal integrity \u2192 <b>3-6\u00d7 better energy efficiency</b></div>'),
    ])
    _right = mo.image(src=IMG / "packaging_approaches.png", caption="Pluggable \u2192 NPO \u2192 CPO \u2192 Optical Interposer")
    mo.vstack([
        mo.md("## Optics: The Promise"),
        mo.hstack([_left, _right], align="center", gap=2, widths=[0.4, 0.6]),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(IMG, mo):
    _left = mo.vstack([
        mo.md("""### Optical transceiver reliability (Huawei):
    - Optical transceivers responsible for 23% of hardware failures
    - Transceiver failure rate: **6.3 per 1000**
    - Training interruptions due to optics: 30% contamination, 70% single-channel failures
    - 90% of single-channel failure = **Laser**"""),
        mo.md("""### Link errors (Meta & Broadcom):
    - Test of Broadcom Tomahawk 5 switches (128 ports @ 400G)
    - 1049k port device hours is equivalent to 
      - 1 Pod [(18 ports x 72 GPU)<sub>GPU</sub> + (18 x 72)<sub>switch</sub> x 8 racks] = **50.6 hours**
      - 16 Pods: **3 hours** 
    - Temperature in compute tray likely higher than switch"""),
    ])
    _right = mo.vstack([
        mo.image(src=IMG / "huawei_failure_pie.png", width="90%", caption='Compute-node failure sources (<a href="https://www-file.huawei.com/admin/asset/v1/pro/view/e3026ae9d7b946e1b713079865da766b.pdf">Huawei 2024</a>)'),
        mo.image(src=IMG / "meta_fec_boxplot.png", width="80%", caption="Amiralizedeh et al., ECOC 2025."),
    ], gap=1)
    _layout = mo.hstack([_left, _right], align="start", gap=2, widths=[0.4, 0.6])
    mo.vstack([mo.md("## Why Not Optics? "), _layout, mo.md("---")])
    return


@app.cell(hide_code=True)
def _(mo):
    failure_rate_per_1000 = mo.ui.slider(start=1.0, stop=20.0, step=0.1, value=6.3, label="Failure rate (/1000/yr)")
    num_pods = mo.ui.slider(start=1, stop=64, step=1, value=16, label="Pods")
    gpus_per_rack = mo.ui.slider(start=32, stop=576, step=8, value=72, label="GPUs/rack")
    ports_per_gpu = mo.ui.slider(start=8, stop=36, step=2, value=18, label="Ports/GPU")
    racks_per_pod = mo.ui.slider(start=1, stop=64, step=1, value=8, label="Racks/pod")
    scaleout_ports_per_rack = mo.ui.slider(start=0, stop=512, step=16, value=0, label="Scale-out ports/rack")
    return (
        failure_rate_per_1000,
        gpus_per_rack,
        num_pods,
        ports_per_gpu,
        racks_per_pod,
        scaleout_ports_per_rack,
    )


@app.cell(hide_code=True)
def _(
    PLOT_FONT,
    PLOT_TITLE_FONT,
    failure_rate_per_1000,
    go,
    gpus_per_rack,
    mo,
    np,
    num_pods,
    ports_per_gpu,
    racks_per_pod,
    scaleout_ports_per_rack,
):
    _fr = failure_rate_per_1000.value / 1000
    _n_pods = num_pods.value
    _n_racks_pod = racks_per_pod.value
    _n_gpus_rack = gpus_per_rack.value
    _n_gpus_pod = _n_gpus_rack * _n_racks_pod

    _scaleup_links_rack = _n_gpus_rack * ports_per_gpu.value
    _scaleup_xcvrs_rack = _scaleup_links_rack * 2  # both endpoints in same rack (GPU + switch)
    _scaleout_xcvrs_rack = scaleout_ports_per_rack.value  # local end only; remote end counted on remote rack

    _xcvrs_rack = _scaleup_xcvrs_rack + _scaleout_xcvrs_rack
    _xcvrs_pod = _xcvrs_rack * _n_racks_pod
    _total_xcvrs = _xcvrs_pod * _n_pods

    _fail_yr = _total_xcvrs * _fr
    _fail_day = _fail_yr / 365
    _mtbf = 8760 / _fail_yr if _fail_yr > 0 else float("inf")

    _pods_arr = np.arange(1, 65)
    _fig = go.Figure()
    for _rate in [2.0, 4.0, 6, 10.0, 15.0]:
        _mtbf_arr = 8760 / (_xcvrs_pod * _pods_arr * _rate / 1000)
        _fig.add_trace(go.Scatter(x=_pods_arr, y=_mtbf_arr, mode="lines", name=f"{_rate}/1k/yr", line=dict(width=2)))

    _fig.add_hline(y=24, line_dash="dash", line_color="#E65100")
    _fig.add_hline(y=1, line_dash="dash", line_color="#C62828")
    _fig.add_annotation(x=1, y=1.4, text="<b>1 failure/day</b>", showarrow=False, xref="paper",
                        yanchor="bottom", xanchor="right", font=dict(size=14, color="#E65100"))
    _fig.add_annotation(x=1, y=0.5, text="<b>1 failure/hour</b>", showarrow=False, xref="paper",
                        yanchor="top", xanchor="right", font=dict(size=14, color="#C62828"))
    _fig.add_vline(x=_n_pods, line_dash="dot", line_color="#9E9E9E", annotation_text=f"{_n_pods} pods",
                   annotation_font=dict(size=14, color="#1A237E"))

    _fig.update_layout(
        title=dict(text="Fleet MTBF vs. Pod Count", font=PLOT_TITLE_FONT),
        xaxis_title="Pods", yaxis_title="MTBF (hours)", yaxis_type="log", yaxis_range=[-1, 4],
        template="plotly_white", height=420, font=PLOT_FONT,
        legend=dict(title=dict(text="Failure rate", font=dict(size=14)), font=dict(size=14)),
    )

    _so_row = f"\n    | Scale-out transceivers/rack | {_scaleout_xcvrs_rack:,} |" if _scaleout_xcvrs_rack > 0 else ""
    mo.vstack([
        mo.md("## Reliability & Fleet Failure Calculator"),
        mo.hstack([failure_rate_per_1000, num_pods, gpus_per_rack, racks_per_pod, ports_per_gpu, scaleout_ports_per_rack], justify="start", gap=0.5),
        mo.hstack([
            mo.md(f"""
    | Metric | Value |
    |:-------|------:|
    | GPUs/pod | {_n_gpus_pod:,} ({_n_gpus_rack} \u00d7 {_n_racks_pod} racks) |
    | Links/rack | {_scaleup_links_rack:,} |
    | Transceivers/rack | {_scaleup_xcvrs_rack:,} (\u00d72: GPU + switch) |{_so_row}
    | Transceivers/rack | {_xcvrs_rack:,} |
    | **Transceivers/pod** | **{_xcvrs_pod:,}** |
    | **Fleet total** | **{_total_xcvrs:,}** |
    | Failures/day | {_fail_day:,.1f} |
    | **MTBF** | **{_mtbf:,.1f} hrs** |

    - Scale-up: each cable has 2 transceivers (GPU-side + switch-side, both in same rack).
    - Scale-out: each port = 1 local transceivers; remote end counted on its own rack.
    - At 2/1000: MTBF = **{8760/(_total_xcvrs*0.002):,.0f} hrs**
    """),
            mo.as_html(_fig),
        ], justify="space-around", align="center", widths=[0.35, 0.65]),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(IMG, mo):
    _imgs1 = mo.hstack([
        mo.image(src=IMG / "cowos_coupe.png", width="100%", caption="TSMC CoWoS with Co-Packaged Optics"),
        mo.image(src=IMG / "coupe_3d_stack.png", width="90%", caption="3D-stacked EIC on PIC (Source: TSMC, OCP APAC 2025)"),
    ], justify="center", gap=2)
    _imgs2 = mo.hstack([
        mo.image(src=IMG / "fiber_bundle_facet.png", width="100%", caption="Fujikura fiber bundle, ~30k cores"),
        mo.image(src=IMG / "multicore_fiber.png", width="100%", caption="Custom MCF with Corning (Azadeh et al., 2022)"),
    ], justify="center", gap=2)
    mo.vstack([
        mo.md("## Bandwidth Density and 3D Integration"),
        _imgs1,
        mo.md("""- UCIe-2.5D interface density: **1-10 Tbps/mm** 
        - 10 Tbps/mm \u21d2 **~1.3 Tbps/fiber** @ 127 \u03bcm fiber pitch
        - 3D integration \u21d2  Compact transceivers (\u2272 55 \u03bcm \u00d7 55 \u03bcm)"""),
        mo.accordion({"Spatially wide-and-parallel links (side note)": mo.hstack([
            mo.md("""- 10 Tbps/mm \u2192 10 Tbps/mm\u00b2 \u21d2 **100 Gbps / (100 \u03bcm)\u00b2**
    - Multi-core fiber and imaging fiber bundles
        - Manage inter-core crosstalk
        - Multimode emission \u21d2 coupling loss to single-mode cores
        - Thick fiber bundles are stiff

    Requires fiber technology and packaging development!"""),
            _imgs2,
        ], justify="space-around", align="center", widths=[0.45, 0.55])}),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(IMG, mo):
    def _cimg(src, width):
        return mo.hstack([mo.image(src=src, width=width)], justify="center")

    def _cap(text):
        return mo.hstack([mo.md(f'<span style="font-size:16px; color:#555">{text}</span>')], justify="center")

    _col_mzm = mo.vstack([
        mo.md("""### Mach-Zehnder Modulators & Lattice Filters
    - **L = 2\u20133 mm**
    - Mature
    - Large footprint, large DWDM circuits
    - EO BW: up to ~90 GHz"""),
        _cimg(IMG / "mzm_photo.jpeg", "80%"),
        _cap("MZM"),
        _cimg(IMG / "mzi-lattice.png", "80%"),
        _cap("16\u03bb WDM"),
    ], gap=0.3)

    _col_mrm = mo.vstack([
        mo.md("""### Microring Modulators & Filters
    - **D = 12\u201340 \u03bcm**
    - Compact serial MUX/DEMUX
    - d\u03bb/dT \u2248 60 pm/K (\u221210.5 GHz/K)
    - Thermal stabilization to **~0.5 K**
    """),
        _cimg(IMG / "mrm_photo.jpeg", "70%"),
        _cap("MRM (D = 12\u201340 \u03bcm)"),
        _cimg(IMG / "mrm-crr.png", "120%"),
        _cap("Serial WDM and coupled ring filter"),
    ], gap=0.3)

    _col_eam = mo.vstack([
        mo.md("""### GeSi Electroabsorption Modulators
    - **L = 30\u201350 \u03bcm**
    - Bulk GeSi: operates near 1550 nm
    - Bandgap shift d\u03bb/dT = 0.8 nm/K
    - Thermal stabilization to **~5 K**"""),
        _cimg(IMG / "eam_photo.png", "50%"),
        _cap("GeSi EAM (imec)"),
        _cimg(IMG / "eam-temperature.png", "70%"),
        _cap("D. Feng et al. (Kotura), JSTQE, 2013."),
    ], gap=0.3)

    mo.vstack([
        mo.md("## Silicon Photonics "),
        mo.hstack([_col_mzm, _col_mrm, _col_eam], justify="space-around", align="start", gap=1.5),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(mo, pd):
    config_df = pd.DataFrame({
        "Option": ["A", "B", "C", "D", "E", "F"],
        "\u03bb count": [32, 16, 16, 8, 8, 4],
        "Lane Rate": ["50 Gbps", "100 Gbps", "100 Gbps", "200 Gbps", "200 Gbps", "400 Gbps"],
        "Format": ["NRZ", "PAM4", "NRZ", "PAM4", "NRZ", "PAM4"],
        "Modulator": ["MRM", "MRM", "MRM", "MRM, EAM*", "MRM, EAM*", "MRM, EAM*, MZM*"],
    })
    mo.vstack([
        mo.md("## 1.6 Tbps per Fiber \u2014 Configuration Options"),
        mo.md("### Spectrally wide-and-parallel vs. High-speed serial \u2014 all achieve 1.6 Tbps/fiber:"),
        mo.as_html(config_df),
        mo.md("Note: EAMs and MZMs need laser fibers per wavelength \u2192 lower bandwidth density than MRMs."),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mod_type = mo.ui.dropdown(options=["MRM", "EAM"], value="MRM", label="Modulator")
    config_select = mo.ui.dropdown(
        options=["32\u03bb \u00d7 50G NRZ", "16\u03bb \u00d7 100G NRZ", "16\u03bb \u00d7 100G PAM4", "8\u03bb \u00d7 200G PAM4"],
        value="16\u03bb \u00d7 100G PAM4", label="Configuration",
    )
    coupling_loss = mo.ui.slider(start=1.0, stop=5.0, step=0.5, value=2.5, label="Fiber coupling (dB)")
    routing_loss = mo.ui.slider(start=0.0, stop=2.0, step=0.5, value=0.5, label="Routing loss (dB)")
    channel_loss = mo.ui.slider(start=0.0, stop=6.0, step=0.5, value=3.0, label="Channel loss (dB)")
    return channel_loss, config_select, coupling_loss, mod_type, routing_loss


@app.cell(hide_code=True)
def _(
    IMG,
    PLOT_FONT,
    PLOT_TITLE_FONT,
    RX_SENS,
    TDECQ,
    TX_PENALTY,
    WDM_DEMUX,
    channel_loss,
    config_select,
    coupling_loss,
    go,
    mo,
    mod_type,
    routing_loss,
):
    _cfg = config_select.value
    _n_lam = 32 if "32\u03bb" in _cfg else (16 if "16\u03bb" in _cfg else 8)
    _rate = 50 if "50G" in _cfg else (100 if "100G" in _cfg else 200)
    _fmt = "NRZ" if "NRZ" in _cfg else "PAM4"
    _mod = mod_type.value

    _rel_oma = TX_PENALTY.get((_mod, _rate, _fmt), 4.0)
    _tdecq = TDECQ.get((_mod, _rate, _fmt), 1.5)
    _wdm_mux = 2.0
    _wdm_demux = WDM_DEMUX.get(_n_lam, 1.0)
    _rx_sens = RX_SENS.get((_rate, _fmt), -10.0)

    _losses = {
        "TX Coupling": coupling_loss.value, "TX Routing": routing_loss.value,
        f"OMA ({_mod})": _rel_oma, "WDM MUX": _wdm_mux,
        "Out Coupling": coupling_loss.value, "Channel": channel_loss.value,
        "In Coupling": coupling_loss.value, "WDM DEMUX": _wdm_demux,
        "RX Routing": routing_loss.value,
    }
    _total_loss = sum(_losses.values())
    _req_laser = _total_loss + _tdecq + _rx_sens

    _labels = list(_losses.keys()) + ["TDECQ", "RX Sens", "Laser/\u03bb"]
    _vals = list(_losses.values()) + [_tdecq, _rx_sens, None]
    _measures = ["relative"] * (len(_losses) + 2) + ["total"]
    _texts = [f"{v:.1f}" if v is not None else f"{_req_laser:.1f}" for v in _vals]
    _y = [v if v is not None else _req_laser for v in _vals]

    _fig = go.Figure(go.Waterfall(
        x=_labels, y=_y, measure=_measures,
        connector={"line": {"color": "#455A64"}},
        increasing={"marker": {"color": "#E65100"}},
        decreasing={"marker": {"color": "#2E7D32"}},
        totals={"marker": {"color": "#00838F"}},
        textposition="outside", text=_texts, textfont=dict(size=14),
    ))
    _fig.update_layout(
        title=dict(text=f"Link Budget: {_mod} {_n_lam}\u03bb \u00d7 {_rate}G {_fmt}", font=PLOT_TITLE_FONT),
        yaxis_title="dB", template="plotly_white", height=420, font=PLOT_FONT, showlegend=False,
        yaxis=dict(range=[min(0, _req_laser) * 1.3, max(_total_loss + max(_losses.values()), abs(_req_laser)) * 1.3]),
    )

    _pie_groups = {
        "Fiber-Chip Coupling": coupling_loss.value * 3,
        "Routing": routing_loss.value * 2,
        f"Modulator ({_mod})": _rel_oma + _tdecq,
        "WDM MUX/DEMUX": _wdm_mux + _wdm_demux,
        "Channel": channel_loss.value,
    }
    _pie_colors = ["#E65100", "#FFA000", "#00838F", "#1A237E", "#2E7D32"]
    _pie_vals = list(_pie_groups.values())
    _pie_labels = list(_pie_groups.keys())

    _fig_pie = go.Figure(go.Pie(
        labels=_pie_labels, values=_pie_vals,
        marker=dict(colors=_pie_colors),
        textinfo="label+value", texttemplate="%{label}<br>%{value:.1f} dB",
        textfont=dict(size=14), hole=0.3,
        hovertemplate="%{label}: %{value:.1f} dB (%{percent})<extra></extra>",
    ))
    _fig_pie.update_layout(
        title=dict(text="Loss Breakdown", font=PLOT_TITLE_FONT),
        height=380, font=PLOT_FONT, showlegend=False,
    )

    _diagram = mo.image(src=IMG / "link_budget_diagram.png", width="100%", caption="Laser \u2192 Coupler \u2192 Modulator \u2192 \u03bb-MUX \u2192 Routing \u2192 Channel \u2192 Coupler \u2192 Routing \u2192 \u03bb-DEMUX \u2192 Detector")
    mo.vstack([
        mo.md("## Optical Link Budget Calculator"),
        _diagram,
        mo.hstack([mod_type, config_select, coupling_loss, routing_loss, channel_loss], justify="start", gap=1),
        mo.hstack([
            mo.as_html(_fig),
            mo.vstack([
                mo.as_html(_fig_pie),
                mo.md(f"Path loss **{_total_loss:.1f} dB** + TDECQ {_tdecq:.1f} dB + RX sens {_rx_sens:.1f} dBm = **Laser/{_n_lam}\u03bb: {_req_laser:.1f} dBm** ({10**(_req_laser/10):.1f} mW)"),
            ]),
        ], justify="space-around", align="center", widths=[0.55, 0.45]),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    laser_wpe = mo.ui.slider(start=0.05, stop=0.25, step=0.01, value=0.12, label="Laser WPE")
    mrm_tuning_power = mo.ui.slider(start=5.0, stop=40.0, step=5.0, value=20.0, label="MRM tuning (mW/\u03bb)")
    demux_power = mo.ui.slider(start=5.0, stop=40.0, step=5.0, value=20.0, label="\u03bb-DEMUX (mW/\u03bb)")
    eam_tuning_power = mo.ui.slider(start=5.0, stop=40.0, step=5.0, value=10.0, label="EAM tuning (mW/λ)")
    pam4_overhead = mo.ui.slider(start=0.0, stop=0.50, step=0.05, value=0.10, label="PAM4 SerDes overhead")
    eic_ref_pjb = mo.ui.slider(start=0.5, stop=5.0, step=0.25, value=2.0, label="EIC SerDes ref (pJ/b @ 50G NRZ)")
    return (
        demux_power,
        eam_tuning_power,
        eic_ref_pjb,
        laser_wpe,
        mrm_tuning_power,
        pam4_overhead,
    )


@app.cell(hide_code=True)
def _(
    ENERGY_CONFIGS,
    PLOT_FONT,
    PLOT_TITLE_FONT,
    demux_power,
    eam_tuning_power,
    eic_ref_pjb,
    eic_xsr_pjb,
    go,
    laser_wpe,
    mo,
    mrm_tuning_power,
    np,
    pam4_overhead,
):
    _wpe = laser_wpe.value
    _mrm_tune = mrm_tuning_power.value
    _eam_tune = eam_tuning_power.value
    _demux_mw = demux_power.value
    _c_eam_mux = ENERGY_CONFIGS[4]["eam_mux"]  # EAM λ-MUX from config (mW/λ)
    _pam4_ovh = pam4_overhead.value
    _eic_ref = eic_ref_pjb.value

    _results = []
    for _c in ENERGY_CONFIGS:
        _req = _c["total_loss_dB"] + _c["tdecq"] + _c["rx_sens"]
        _laser_mW = 10 ** (_req / 10) / _wpe
        _pic_per_lam = (_mrm_tune + _demux_mw) if _c["mod"] == "MRM" else (_eam_tune + _c["eam_mux"] + _demux_mw)
        _pic_pjb = _pic_per_lam / _c["rate"]
        _laser_pjb = _laser_mW / _c["rate"]
        _opt_pjb = _pic_pjb + _laser_pjb
        _fmt = "PAM4" if "PAM4" in _c["label"] else "NRZ"
        _eic = eic_xsr_pjb(_c["rate"], _fmt, _pam4_ovh, ref_pjb=_eic_ref)
        _tot_pjb = _opt_pjb + _eic
        _results.append({"label": _c["label"], "pic_pjb": _pic_pjb, "laser_pjb": _laser_pjb,
                         "optical_pjb": _opt_pjb, "eic_pjb": _eic,
                         "total_pjb": _tot_pjb, "power_W": _tot_pjb * 1.6})

    _colors = ["#1A237E", "#00838F", "#E65100", "#FFA000", "#2E7D32", "#7B1FA2", "#C62828"]
    _fig = go.Figure()
    for _w in [5, 6, 7, 8, 9]:
        _x = np.linspace(0, 6, 50)
        _fig.add_trace(go.Scatter(x=_x, y=(_w / 1.6) - _x, mode="lines",
            line=dict(color="lightgray", dash="dash", width=1), showlegend=False, hoverinfo="skip"))
        _ly = 0.15
        _lx = _w / 1.6 - _ly
        if 0 <= _lx <= 5.5:
            _fig.add_annotation(x=_lx, y=_ly, text=f"{_w}W", showarrow=False,
                                font=dict(size=14, color="gray"), xanchor="center")

    _label_offsets = [
        (0, -50),     # 0  MRM 32x50G NRZ — higher above
        (-70, -35),   # 1  MRM 16x100G NRZ — upper-left
        (-90, 0),     # 2  MRM 16x100G PAM4 — left
        (-80, 40),    # 3  MRM 8x200G PAM4 — lower-left
        (0, 40),      # 4  EAM 16x100G NRZ — below
        (0, 40),      # 5  EAM 16x100G PAM4 — below
        (80, 0),      # 6  EAM 8x200G PAM4 — right
    ]
    for _idx, _r in enumerate(_results):
        _fig.add_trace(go.Scatter(
            x=[_r["optical_pjb"]], y=[_r["eic_pjb"]], mode="markers",
            marker=dict(size=16, color=_colors[_idx]), name=_r["label"],
            hovertemplate=f"<b>{_r['label']}</b><br>Total: {_r['total_pjb']:.2f} pJ/b<br>{_r['power_W']:.1f}W @ 1.6T<extra></extra>",
        ))
        _ax, _ay = _label_offsets[_idx % len(_label_offsets)]
        _fig.add_annotation(
            x=_r["optical_pjb"], y=_r["eic_pjb"], text=_r["label"],
            showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor=_colors[_idx],
            ax=_ax, ay=_ay, font=dict(size=13, color=_colors[_idx]),
            bgcolor="rgba(255,255,255,0.8)", borderpad=2,
        )

    _fig.update_layout(
        title=dict(text="Optical vs. Electrical Power", font=PLOT_TITLE_FONT),
        xaxis_title="PIC + Laser (pJ/b)", yaxis_title="EIC + XSR SerDes (pJ/b)",
        xaxis_range=[0, 4.5], yaxis_range=[0, 4.0],
        template="plotly_white", height=480, font=PLOT_FONT, showlegend=False,
    )

    _rows = "\n".join(
        f"| {_r['label']} | {_r['pic_pjb']:.2f} | {_r['laser_pjb']:.2f} | {_r['optical_pjb']:.2f} | {_r['eic_pjb']:.2f} | **{_r['total_pjb']:.2f}** | {_r['power_W']:.1f} |"
        for _r in _results
    )

    _tuning_table = mo.md(f"""| Component | MRM | EAM |
    |:----------|:---:|:---:|
    | Mod. tuning | {_mrm_tune:.0f} mW/λ | {_eam_tune:.0f} mW/λ |
    | λ-MUX | — | {_c_eam_mux:.0f} mW/λ |
    | λ-DEMUX | {_demux_mw:.0f} mW/λ | {_demux_mw:.0f} mW/λ |
    | **Total PIC thermal** | **{_mrm_tune + _demux_mw:.0f} mW/λ** | **{_eam_tune + _c_eam_mux + _demux_mw:.0f} mW/λ** |""")

    _power_table = mo.md(f"""| Config | PIC | Laser | Optical | EIC+XSR | **Total** | **1.6T (W)** |
    |:-------|:---:|:-----:|:-------:|:-------:|:---------:|:-----:|
    {_rows}""")

    _footer = mo.md(f"Laser is **50–90%** of optical pJ/b. WPE={_wpe*100:.0f}%. EIC ref={_eic_ref:.1f} pJ/b @ 50G NRZ.")

    _right = mo.vstack([mo.accordion({"PIC thermal tuning breakdown": _tuning_table}), _power_table, _footer])

    mo.vstack([
        mo.md("## Energy Efficiency Calculator"),
        mo.hstack([laser_wpe, mrm_tuning_power, eam_tuning_power, demux_power], justify="start", gap=1),
        mo.hstack([pam4_overhead, eic_ref_pjb], justify="start", gap=1),
        mo.hstack([mo.as_html(_fig), _right], widths=[0.5, 0.5], align="start"),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(LAT_EXPONENTS_DEFAULT, mo):
    fiber_length_m = mo.ui.slider(start=1, stop=100.0, step=1, value=5.0, label="Fiber length (m)")
    lat_exp_serdes = mo.ui.slider(start=0.3, stop=1.0, step=0.1, value=LAT_EXPONENTS_DEFAULT["Serializer"], label="\u03b1 Ser/Des/CDR")
    lat_exp_fec = mo.ui.slider(start=0.2, stop=1.0, step=0.1, value=LAT_EXPONENTS_DEFAULT["Lite FEC"], label="\u03b1 FEC")
    lat_exp_dac_adc = mo.ui.slider(start=0.3, stop=1.0, step=0.1, value=LAT_EXPONENTS_DEFAULT["DAC TX"], label="\u03b1 DAC/ADC")
    lat_exp_dsp = mo.ui.slider(start=0.3, stop=1.0, step=0.1, value=LAT_EXPONENTS_DEFAULT["RX Eq DSP"], label="\u03b1 DSP")
    return (
        fiber_length_m,
        lat_exp_dac_adc,
        lat_exp_dsp,
        lat_exp_fec,
        lat_exp_serdes,
    )


@app.cell(hide_code=True)
def _(
    LAT_REF,
    PLOT_FONT,
    PLOT_TITLE_FONT,
    fiber_length_m,
    go,
    lat_exp_dac_adc,
    lat_exp_dsp,
    lat_exp_fec,
    lat_exp_serdes,
    latency_for,
    mo,
):
    _c = 2.998e8
    _n_fiber = 1.468
    _L = fiber_length_m.value
    _tof_ns = _L * _n_fiber / _c * 1e9

    _exp = {
        "Serializer": lat_exp_serdes.value, "Lite FEC": lat_exp_fec.value,
        "DAC TX": lat_exp_dac_adc.value, "ADC RX": lat_exp_dac_adc.value,
        "RX Eq DSP": lat_exp_dsp.value, "RX DeSer+CDR": lat_exp_serdes.value,
    }
    LATENCY = {
        "50G NRZ": latency_for(50, "NRZ", _exp),
        "100G NRZ": latency_for(100, "NRZ", _exp),
        "100G PAM4": latency_for(50, "PAM4", _exp),
        "200G PAM4": latency_for(100, "PAM4", _exp),
    }

    _formats = list(LATENCY.keys())
    _components = ["Fiber ToF"] + list(LAT_REF.keys())

    _fig = go.Figure()
    for _comp_name in _components:
        _vals = []
        for _fmt in _formats:
            if _comp_name == "Fiber ToF":
                _vals.append(_tof_ns)
            else:
                _vals.append(LATENCY[_fmt][_comp_name])
        _fig.add_trace(go.Bar(name=_comp_name, x=_formats, y=_vals))

    _totals = {k: sum(v.values()) + _tof_ns for k, v in LATENCY.items()}
    _max_total = max(_totals.values())

    _fig.update_layout(
        barmode="stack",
        title=dict(text=f"One-Way Latency (fiber = {_L:.0f} m)", font=PLOT_TITLE_FONT),
        yaxis_title="Latency (ns)", template="plotly_white", height=420, font=PLOT_FONT,
        yaxis=dict(range=[0, _max_total * 1.15]),
    )

    for _fmt_name, _total_val in _totals.items():
        _fig.add_annotation(
            x=_fmt_name, y=_total_val, text=f"<b>{_total_val:.1f}</b>",
            showarrow=False, yshift=12, font=dict(size=14),
        )
    _total_md = " \u00b7 ".join(f"**{k}:** {v:.1f} ns" for k, v in _totals.items())

    _rows = []
    for _k, _ref in LAT_REF.items():
        _a = _exp[_k]
        _at100 = _ref * (50 / 100) ** _a
        _nrz = "\u2014" if _k in ("Lite FEC", "DAC TX", "ADC RX") else "\u2713"
        _rows.append(f"| {_k} | {_ref:.1f} ns | {_at100:.1f} ns | {_a:.1f} | {_nrz} | \u2713 |")
    _rows.append(f"| Fiber ToF | {_tof_ns:.1f} ns | {_tof_ns:.1f} ns | \u2014 | \u2713 | \u2713 |")
    _table_str = "| Component | 50 Gbd | 100 Gbd | \u03b1 | NRZ | PAM4 |\n|:----------|:--------:|:---------:|:--:|:---:|:----:|\n" + "\n".join(_rows)

    mo.vstack([
        mo.md("## Latency Breakdown"),
        mo.hstack([fiber_length_m, lat_exp_serdes, lat_exp_fec], justify="start", gap=0.5),
        mo.hstack([lat_exp_dac_adc, lat_exp_dsp], justify="start", gap=0.5),
        mo.hstack([
            mo.as_html(_fig),
            mo.vstack([
                mo.md(_table_str),
                mo.md(f"Scaling: latency = ref \u00d7 (50/baud)$^\\alpha$ | NRZ: no FEC/DAC/ADC |\nToF = {_L:.0f} m \u00d7 {_n_fiber}/c = {_tof_ns:.1f} ns"),
            ]),
        ], justify="space-around", align="start", widths=[0.6, 0.4]),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(IMG, mo):
    _table = mo.md("""| Group | Config | Total BW | Year |
    |:------|:-------|:---------|:-----|
    | **Intel** | 8λ × 32G NRZ (integrated lasers) | 256 Gbps | OFC 2024 |
    | **Ayar Labs** | 16λ × 32G NRZ | 512 Gbps | OFC 2025 |
    | **NVIDIA Research** | 8λ × 32G NRZ | 256 Gbps | ISSCC 2026 |
    | **Lightmatter** | 16λ × 56G NRZ (bi-dir) | 896 Gbps | OFC 2026 |
    | **Lightmatter** | 16λ × 112G PAM4 | **1.8 Tbps** | 2026 |
    """)
    def _cimg(src, width):
        return mo.hstack([mo.image(src=src, width=width)], justify="center")

    def _cap(text):
        return mo.hstack([mo.md(f'<span style="font-size:14px; color:#555">{text}</span>')], justify="center")

    _col_32g = mo.vstack([
        mo.md("### 32G NRZ x  8λ or 16λ"),
        _cimg(IMG / "intel_oci_system.png", "90%"),
        _cap("Intel 8λ × 32G NRZ, integrated lasers (OFC 2024)"),
        _cimg(IMG / "ayar_testboard.png", "90%"),
        _cap("Ayar Labs TeraPHY 16λ × 32G NRZ (OFC 2025)"),
        _cimg(IMG / "nvidia_board_link.png", "90%"),
        _cap("NVIDIA Research 8λ × 32G NRZ (ISSCC 2026)"),
    ], gap=0.3)

    _col_hbr = mo.vstack([
        mo.md("### 56G NRZ x 16λ"),
        _cimg(IMG / "lightmatter-bidi.png", "90%"),
        _cap("Lightmatter 16λ × 56G NRZ bi-dir (OFC 2026)"),
        mo.md("### 112G PAM4 x 16λ"),
        _cimg(IMG / "lightmatter-100g.png", "90%"),
        _cap("Lightmatter 16λ × 112G PAM4 (2026)"),
    ], gap=0.3)

    _imgs = mo.hstack([_col_32g, _col_hbr], justify="space-around", align="start", gap=1.5)
    mo.vstack([mo.md("## MRM WDM Link Examples"), _table, _imgs, mo.md("---")])
    return


@app.cell(hide_code=True)
def _(IMG, mo):
    mo.vstack([
        mo.md("## DWDM Laser Source Examples"),
        mo.image(src=IMG / "dwdm_laser_sources.png", width="100%"),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    rack_select = mo.ui.dropdown(
        options={"GB200 NVL72 (2025)": "nvl72", "Vera Rubin NVL144 (2026)": "nvl144",
                 "Rubin Ultra NVL576 (2027)": "nvl576"},
        value="Vera Rubin NVL144 (2026)", label="Rack",
    )
    optics_config = mo.ui.dropdown(
        options={"MRM 16\u03bb\u00d7100G NRZ": "mrm_16x100g_nrz", "MRM 16\u03bb\u00d7100G PAM4": "mrm_16x100g_pam4",
                 "MRM 32\u03bb\u00d750G NRZ": "mrm_32x50g_nrz", "MRM 8\u03bb\u00d7200G PAM4": "mrm_8x200g_pam4",
                 "EAM 16\u03bb\u00d7100G PAM4": "eam_16x100g_pam4", "EAM 8\u03bb\u00d7200G PAM4": "eam_8x200g_pam4"},
        value="MRM 16\u03bb\u00d7100G PAM4", label="Optics",
    )
    pjb_overhead = mo.ui.slider(start=0.5, stop=2.0, step=0.1, value=1.0, label="Power overhead \u00d7")
    return optics_config, pjb_overhead, rack_select


@app.cell(hide_code=True)
def _(
    OPTICS_CONFIGS,
    RACKS,
    math,
    mo,
    optics_config,
    pjb_overhead,
    rack_select,
):
    _rk = RACKS[rack_select.value]
    _op = OPTICS_CONFIGS[optics_config.value]

    _bw_bi = _rk["gpus"] * _rk["bw_per_gpu_tbps"] * 2
    _bw_fib = _op["n_lam"] * _op["rate"] / 1000
    _fibers_uni = math.ceil(_bw_bi / 2 / _bw_fib)
    _total_fibers = _fibers_uni * 2 * _op["fibers_per_dir"]
    _total_lanes = int(_bw_bi * 1000 / _op["rate"])
    _total_engines = _fibers_uni * 2

    _io_W = _op["pjb"] * _bw_bi * pjb_overhead.value
    _pct = _io_W / (_rk["rack_power_kw"] * 1000) * 100


    _metric_table = mo.md(f"""| Metric | Value |
    |:-------|------:|
    | GPUs | {_rk['gpus']} |
    | **Total BW (bi-dir)** | **{_bw_bi:,.0f} Tbps** |
    | BW/fiber | {_bw_fib} Tbps |
    | **Total fibers** | **{_total_fibers:,}** |
    | Total lanes | {_total_lanes:,} |
    | Photonic engines | {_total_engines:,} |""")

    _power_table = mo.md(f"""| Power | Value |
    |:------|------:|
    | Energy/bit | {_op['pjb']:.2f} pJ/b |
    | IO power ({pjb_overhead.value:.1f}\u00d7) | **{_io_W:,.0f} W** |
    | Rack budget | {_rk['rack_power_kw']} kW |
    | **% for IO** | **{_pct:.1f}%** |""")

    mo.vstack([
        mo.md("## Hypothetical  1.6Tbps/fiber CPO in a Rack "),
        mo.hstack([rack_select, optics_config, pjb_overhead], justify="start", gap=1),
        mo.md(f"""### {_rk['name']} + {_op['label']}"""),
        mo.hstack([_metric_table, _power_table], widths=[0.3, 0.7], align="start"),
        mo.md("---"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    ### System requirements on scale-up networks
    - AI compute clusters require ~O(10<sup>4</sup>) Tbps-class interconnects per rack
    - Huge opportunity and supply chain challenge
    - Key metrics: bandwidth density, power consumption, latency
    - Critical for depolyment: Reliability, link stability

    ### Co-packaged optics for 1.6 Tbps/fiber and beyond
    - WDM + 100+G lane rate → bandwidth density with 16λ laser arrays
    - NRZ for lowest latency
    - State of the art: MRM 16λ × 112G-PAM4 → **1.6 Tbps per fiber**

    ### Major opportunities
    - Components: fiber-to-chip couplers, efficient modulators, reliable lasers
    - Links: Telemetry, redundancy, network management for error-free operation
    """)
    return


if __name__ == "__main__":
    app.run()
