"""Pytest suite for all calculations in ofc_workshop.py."""
import math
import pytest

# ═══════════════════════════════════════════════════════════════════
# Re-implement the notebook's pure functions so tests are standalone
# ═══════════════════════════════════════════════════════════════════

def personick_q(ber, fmt):
    if fmt == "NRZ":
        target = 2 * ber
    else:
        target = 8 * ber / 3
    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erfc(mid) > target:
            lo = mid
        else:
            hi = mid
    return mid * math.sqrt(2)

REF_SENS = -12.0
REF_BW = 25.0
REF_Q = personick_q(1e-15, "NRZ")

def rx_sensitivity(rate_gbps, fmt):
    ber = 1e-15 if fmt == "NRZ" else 1e-6
    q = personick_q(ber, fmt)
    bw = rate_gbps / 2 if fmt == "NRZ" else rate_gbps / 4
    pam4_pen = 10 * math.log10(3) if fmt == "PAM4" else 0
    return REF_SENS + pam4_pen + 10 * math.log10(q / REF_Q) + 5 * math.log10(bw / REF_BW)

EIC_REF_PJB = 2.0
EIC_REF_BAUD = 50.0

def eic_xsr_pjb(rate_gbps, fmt, pam4_overhead=0.10, ref_pjb=None):
    _ref = ref_pjb if ref_pjb is not None else EIC_REF_PJB
    baud = rate_gbps if fmt == "NRZ" else rate_gbps / 2
    lane_mw = _ref * EIC_REF_BAUD * (baud / EIC_REF_BAUD) * (1 + pam4_overhead * (fmt == "PAM4"))
    return lane_mw / rate_gbps

BAUD_REF = 50
LAT_REF = {"Serializer": 4, "Lite FEC": 50, "DAC TX": 1, "ADC RX": 0.5, "RX Eq DSP": 8, "RX DeSer+CDR": 8}
LAT_EXPONENTS_DEFAULT = {"Serializer": 0.9, "Lite FEC": 0.4, "DAC TX": 0.8, "ADC RX": 0.8, "RX Eq DSP": 0.6, "RX DeSer+CDR": 0.8}

def latency_for(baud, fmt, exponents):
    result = {}
    for k, v in LAT_REF.items():
        is_pam4_only = k in ("Lite FEC", "DAC TX", "ADC RX")
        if is_pam4_only and fmt != "PAM4":
            result[k] = 0
        else:
            result[k] = v * (BAUD_REF / baud) ** exponents[k]
    return result

ENERGY_CONFIGS = [
    {"label": "MRM 32λ×50G NRZ", "mod": "MRM", "n_lam": 32, "rate": 50,
     "total_loss_dB": 18.2, "tdecq": 0.5, "rx_sens": rx_sensitivity(50, "NRZ"), "eam_mux": 0},
    {"label": "MRM 16λ×100G NRZ", "mod": "MRM", "n_lam": 16, "rate": 100,
     "total_loss_dB": 20.4, "tdecq": 1.0, "rx_sens": rx_sensitivity(100, "NRZ"), "eam_mux": 0},
    {"label": "MRM 16λ×100G PAM4", "mod": "MRM", "n_lam": 16, "rate": 100,
     "total_loss_dB": 17.9, "tdecq": 2.0, "rx_sens": rx_sensitivity(100, "PAM4"), "eam_mux": 0},
    {"label": "MRM 8λ×200G PAM4", "mod": "MRM", "n_lam": 8, "rate": 200,
     "total_loss_dB": 20.2, "tdecq": 3.0, "rx_sens": rx_sensitivity(200, "PAM4"), "eam_mux": 0},
    {"label": "EAM 16λ×100G NRZ", "mod": "EAM", "n_lam": 16, "rate": 100,
     "total_loss_dB": 19.5, "tdecq": 1.5, "rx_sens": rx_sensitivity(100, "NRZ"), "eam_mux": 20},
    {"label": "EAM 16λ×100G PAM4", "mod": "EAM", "n_lam": 16, "rate": 100,
     "total_loss_dB": 20.5, "tdecq": 2.0, "rx_sens": rx_sensitivity(100, "PAM4"), "eam_mux": 20},
    {"label": "EAM 8λ×200G PAM4", "mod": "EAM", "n_lam": 8, "rate": 200,
     "total_loss_dB": 21.0, "tdecq": 3.5, "rx_sens": rx_sensitivity(200, "PAM4"), "eam_mux": 20},
]

RACKS = {
    "nvl72":  {"name": "GB200 NVL72",        "gpus": 72,  "bw_per_gpu_tbps": 18 * 400 / 1000, "rack_power_kw": 140},
    "nvl144": {"name": "Vera Rubin NVL144",   "gpus": 144, "bw_per_gpu_tbps": 18 * 400 / 1000, "rack_power_kw": 210},
    "nvl576": {"name": "Rubin Ultra NVL576",  "gpus": 576, "bw_per_gpu_tbps": 1.5 * 8 * 1000 / 576 / 2, "rack_power_kw": 600},
}


# ═══════════════════════════════════════════════════════════════════
# 1. BANDWIDTH CALCULATIONS
# ═══════════════════════════════════════════════════════════════════

class TestBandwidth:
    def test_nvl72_bw_per_gpu(self):
        assert RACKS["nvl72"]["bw_per_gpu_tbps"] == pytest.approx(7.2)

    def test_nvl144_bw_per_gpu(self):
        assert RACKS["nvl144"]["bw_per_gpu_tbps"] == pytest.approx(7.2)

    def test_nvl576_bw_per_gpu(self):
        assert RACKS["nvl576"]["bw_per_gpu_tbps"] == pytest.approx(10.4167, rel=1e-3)

    def test_nvl72_rack_bidir(self):
        r = RACKS["nvl72"]
        bw = r["gpus"] * r["bw_per_gpu_tbps"] * 2
        assert bw == pytest.approx(1036.8)

    def test_nvl144_rack_bidir(self):
        r = RACKS["nvl144"]
        bw = r["gpus"] * r["bw_per_gpu_tbps"] * 2
        assert bw == pytest.approx(2073.6)

    def test_nvl576_rack_bidir(self):
        r = RACKS["nvl576"]
        bw = r["gpus"] * r["bw_per_gpu_tbps"] * 2
        assert bw == pytest.approx(12000.0)

    def test_nvl576_from_PBps(self):
        bidir_pbps = 1.5 * 8
        assert bidir_pbps == pytest.approx(12.0)
        bw_per_gpu = bidir_pbps * 1000 / 576 / 2
        assert bw_per_gpu == pytest.approx(10.4167, rel=1e-3)


class TestGpuSystems:
    """Verify the GPU_SYSTEMS bandwidth calculation paths."""

    def test_ports_per_gpu_path(self):
        bw_gpu = 18 * 400 / 1000
        bw_rack = 72 * 18 * 400 * 2 / 1e6
        assert bw_gpu == pytest.approx(7.2)
        assert bw_rack == pytest.approx(1.0368)

    def test_lanes_per_gpu_path(self):
        bw_gpu = 64 * 112 / 1000
        bw_rack = 64 * 64 * 112 * 2 / 1e6
        assert bw_gpu == pytest.approx(7.168)
        assert bw_rack == pytest.approx(0.917504, rel=1e-3)

    def test_rack_bw_PBps_path(self):
        bw_rack_bidir = 1.5 * 8
        bw_gpu = bw_rack_bidir * 1000 / 576 / 2
        assert bw_rack_bidir == pytest.approx(12.0)
        assert bw_gpu == pytest.approx(10.4167, rel=1e-3)

    def test_cloudmatrix(self):
        bw_gpu = 7 * 400 / 1000
        bw_rack = 32 * 7 * 400 * 2 / 1e6
        assert bw_gpu == pytest.approx(2.8)
        assert bw_rack == pytest.approx(0.1792)


# ═══════════════════════════════════════════════════════════════════
# 2. IO POWER
# ═══════════════════════════════════════════════════════════════════

class TestIOPower:
    @pytest.mark.parametrize("pjb,bw_pbps,expected_kw", [
        (10, 1.0368, 10.368),
        (20, 1.0368, 20.736),
        (10, 12.0, 120.0),
        (20, 12.0, 240.0),
    ])
    def test_io_power_formula(self, pjb, bw_pbps, expected_kw):
        io_kw = pjb * 1e-12 * bw_pbps * 1e15 / 1000
        assert io_kw == pytest.approx(expected_kw)

    def test_io_power_simplification(self):
        """pJ/b × Pbps simplifies to kW."""
        pjb, bw_pbps = 15.0, 2.0
        full = pjb * 1e-12 * bw_pbps * 1e15 / 1000
        simple = pjb * bw_pbps
        assert full == pytest.approx(simple)

    def test_dimensional_analysis(self):
        """pJ/b × Tbps = W."""
        pjb = 3.5    # pJ/b
        bw_tbps = 100  # Tbps
        power_W = pjb * 1e-12 * bw_tbps * 1e12
        assert power_W == pytest.approx(pjb * bw_tbps)

    @pytest.mark.parametrize("name,rack_kw,bw_pbps", [
        ("nvl72", 140, 1.0368),
        ("nvl144", 210, 2.0736),
        ("nvl576", 600, 12.0),
    ])
    def test_percent_power(self, name, rack_kw, bw_pbps):
        io_lo = 10 * bw_pbps
        io_hi = 20 * bw_pbps
        pct_lo = io_lo / rack_kw * 100
        pct_hi = io_hi / rack_kw * 100
        assert pct_lo < pct_hi
        assert pct_lo > 0
        assert pct_hi < 100


# ═══════════════════════════════════════════════════════════════════
# 3. PERSONICK Q & RECEIVER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════

class TestPersonickQ:
    def test_nrz_roundtrip(self):
        """BER = 0.5·erfc(Q/√2) should give back original BER."""
        ber = 1e-15
        q = personick_q(ber, "NRZ")
        ber_back = 0.5 * math.erfc(q / math.sqrt(2))
        assert ber_back == pytest.approx(ber, rel=1e-6)

    def test_pam4_roundtrip(self):
        """BER = (3/8)·erfc(Q/√2) should give back original BER."""
        ber = 1e-6
        q = personick_q(ber, "PAM4")
        ber_back = (3 / 8) * math.erfc(q / math.sqrt(2))
        assert ber_back == pytest.approx(ber, rel=1e-6)

    def test_q_nrz_value(self):
        q = personick_q(1e-15, "NRZ")
        assert q == pytest.approx(7.941, rel=1e-3)

    def test_q_pam4_value(self):
        q = personick_q(1e-6, "PAM4")
        assert q == pytest.approx(4.695, rel=1e-3)

    def test_q_increases_with_lower_ber(self):
        q_high = personick_q(1e-6, "NRZ")
        q_low = personick_q(1e-15, "NRZ")
        assert q_low > q_high

    def test_pam4_ber_formula_derivation(self):
        """SER = (3/4)erfc(Q/√2), Gray BER = SER/2 = (3/8)erfc(Q/√2)."""
        Q = 5.0
        M = 4
        SER = 2 * (M - 1) / M * 0.5 * math.erfc(Q / math.sqrt(2))
        BER = SER / math.log2(M)
        expected = (3 / 8) * math.erfc(Q / math.sqrt(2))
        assert BER == pytest.approx(expected, rel=1e-10)


class TestRxSensitivity:
    def test_reference_point(self):
        """50G NRZ should return the reference sensitivity exactly."""
        assert rx_sensitivity(50, "NRZ") == pytest.approx(REF_SENS, abs=1e-10)

    def test_sensitivity_degrades_with_rate(self):
        assert rx_sensitivity(100, "NRZ") > rx_sensitivity(50, "NRZ")

    def test_pam4_penalty(self):
        """PAM4 sensitivity should be worse (higher dBm) due to OMA/3 penalty."""
        sens_nrz = rx_sensitivity(100, "NRZ")
        sens_pam4 = rx_sensitivity(100, "PAM4")
        assert sens_pam4 > sens_nrz

    def test_pam4_oma_penalty_value(self):
        assert 10 * math.log10(3) == pytest.approx(4.771, rel=1e-3)

    def test_nyquist_bw_nrz(self):
        """NRZ Nyquist BW = rate/2."""
        assert 100 / 2 == 50.0

    def test_nyquist_bw_pam4(self):
        """PAM4 Nyquist BW = rate/4 (baud=rate/2, BW=baud/2)."""
        assert 100 / 4 == 25.0

    @pytest.mark.parametrize("rate,fmt,expected", [
        (50, "NRZ", -12.00),
        (100, "NRZ", -10.49),
        (100, "PAM4", -9.51),
        (200, "PAM4", -8.01),
    ])
    def test_sensitivity_values(self, rate, fmt, expected):
        assert rx_sensitivity(rate, fmt) == pytest.approx(expected, abs=0.02)

    def test_sensitivity_scaling_formula(self):
        """5·log10(BW/BW_ref) comes from 10·log10(√(BW/BW_ref))."""
        bw, bw_ref = 50.0, 25.0
        assert 5 * math.log10(bw / bw_ref) == pytest.approx(
            10 * math.log10(math.sqrt(bw / bw_ref)), rel=1e-10)


# ═══════════════════════════════════════════════════════════════════
# 4. EIC SERDES POWER MODEL
# ═══════════════════════════════════════════════════════════════════

class TestEicSerDes:
    def test_reference_point(self):
        """50G NRZ at ref should return exactly ref_pjb."""
        assert eic_xsr_pjb(50, "NRZ") == pytest.approx(2.0)

    def test_nrz_100g(self):
        """100G NRZ: baud=100, lane_mw = 2*50*(100/50)*1 = 200, pjb = 200/100 = 2.0."""
        assert eic_xsr_pjb(100, "NRZ") == pytest.approx(2.0)

    def test_pam4_100g(self):
        """100G PAM4: baud=50, lane_mw = 2*50*(50/50)*1.1 = 110, pjb = 110/100 = 1.1."""
        assert eic_xsr_pjb(100, "PAM4") == pytest.approx(1.1)

    def test_pam4_200g(self):
        """200G PAM4: baud=100, lane_mw = 2*50*(100/50)*1.1 = 220, pjb = 220/200 = 1.1."""
        assert eic_xsr_pjb(200, "PAM4") == pytest.approx(1.1)

    def test_nrz_pjb_constant_with_rate(self):
        """NRZ pJ/b is constant regardless of rate (power ∝ baud, bits ∝ baud)."""
        assert eic_xsr_pjb(50, "NRZ") == pytest.approx(eic_xsr_pjb(100, "NRZ"))

    def test_pam4_lower_than_nrz(self):
        """PAM4 gets 2 bits/symbol at same baud → lower pJ/b."""
        assert eic_xsr_pjb(100, "PAM4") < eic_xsr_pjb(100, "NRZ")

    def test_custom_ref_pjb(self):
        pjb = eic_xsr_pjb(50, "NRZ", ref_pjb=3.0)
        assert pjb == pytest.approx(3.0)

    def test_lane_power_simplification(self):
        """lane_mw = ref_pjb * ref_baud * (baud/ref_baud) simplifies to ref_pjb * baud."""
        ref_pjb, ref_baud, baud = 2.0, 50.0, 100.0
        full = ref_pjb * ref_baud * (baud / ref_baud)
        simple = ref_pjb * baud
        assert full == pytest.approx(simple)


# ═══════════════════════════════════════════════════════════════════
# 5. ENERGY EFFICIENCY CALCULATOR
# ═══════════════════════════════════════════════════════════════════

class TestEnergyCalculator:
    WPE = 0.12
    MRM_TUNE = 20.0
    EAM_TUNE = 10.0
    DEMUX_MW = 20.0

    def _compute(self, cfg):
        req = cfg["total_loss_dB"] + cfg["tdecq"] + cfg["rx_sens"]
        laser_mW = 10 ** (req / 10) / self.WPE
        if cfg["mod"] == "MRM":
            pic_per_lam = self.MRM_TUNE + self.DEMUX_MW
        else:
            pic_per_lam = self.EAM_TUNE + self.DEMUX_MW + self.DEMUX_MW
        pic_pjb = pic_per_lam / cfg["rate"]
        laser_pjb = laser_mW / cfg["rate"]
        fmt = "PAM4" if "PAM4" in cfg["label"] else "NRZ"
        eic = eic_xsr_pjb(cfg["rate"], fmt)
        total = pic_pjb + laser_pjb + eic
        return {"pic_pjb": pic_pjb, "laser_pjb": laser_pjb, "eic": eic, "total": total}

    def test_mrm_pic_tuning(self):
        """MRM PIC = MRM_tuning + DEMUX."""
        pic = self.MRM_TUNE + self.DEMUX_MW
        assert pic == 40.0

    def test_eam_pic_tuning(self):
        """EAM PIC = EAM_tuning + MUX(=DEMUX) + DEMUX."""
        pic = self.EAM_TUNE + self.DEMUX_MW + self.DEMUX_MW
        assert pic == 50.0

    def test_laser_power_positive(self):
        for cfg in ENERGY_CONFIGS:
            r = self._compute(cfg)
            assert r["laser_pjb"] > 0, f"Laser pjb should be positive for {cfg['label']}"

    def test_total_equals_sum(self):
        for cfg in ENERGY_CONFIGS:
            r = self._compute(cfg)
            assert r["total"] == pytest.approx(r["pic_pjb"] + r["laser_pjb"] + r["eic"])

    def test_power_at_1_6T(self):
        """Power(W) = total_pjb × 1.6 (dimensional: pJ/b × Tbps = W)."""
        for cfg in ENERGY_CONFIGS:
            r = self._compute(cfg)
            power_W = r["total"] * 1.6
            assert power_W > 0
            assert power_W == pytest.approx(r["total"] * 1.6)

    def test_laser_significant_fraction_of_optical(self):
        """Laser should be a major portion (>40%) of optical pJ/b for all configs."""
        for cfg in ENERGY_CONFIGS:
            r = self._compute(cfg)
            optical = r["pic_pjb"] + r["laser_pjb"]
            assert r["laser_pjb"] / optical > 0.4, f"Laser fraction too low for {cfg['label']}"

    @pytest.mark.parametrize("label,expected_total", [
        ("MRM 32λ×50G NRZ", 3.58),
        ("MRM 16λ×100G NRZ", 3.43),
        ("MRM 16λ×100G PAM4", 2.41),
        ("MRM 8λ×200G PAM4", 2.68),
        ("EAM 16λ×100G NRZ", 3.44),
        ("EAM 16λ×100G PAM4", 3.26),
        ("EAM 8λ×200G PAM4", 3.21),
    ])
    def test_computed_pjb_values(self, label, expected_total):
        cfg = next(c for c in ENERGY_CONFIGS if c["label"] == label)
        r = self._compute(cfg)
        assert r["total"] == pytest.approx(expected_total, abs=0.02)


# ═══════════════════════════════════════════════════════════════════
# 6. LINK BUDGET
# ═══════════════════════════════════════════════════════════════════

class TestLinkBudget:
    def test_required_laser_formula(self):
        """req_laser_dBm = total_loss + TDECQ + rx_sens."""
        loss, tdecq, rx = 17.9, 2.0, -9.51
        req = loss + tdecq + rx
        assert req == pytest.approx(10.39, abs=0.02)

    def test_dbm_to_mw(self):
        """10 dBm = 10 mW, 0 dBm = 1 mW, -10 dBm = 0.1 mW."""
        assert 10 ** (10 / 10) == pytest.approx(10.0)
        assert 10 ** (0 / 10) == pytest.approx(1.0)
        assert 10 ** (-10 / 10) == pytest.approx(0.1)

    def test_wpe_division(self):
        """Electrical laser power = optical / WPE."""
        optical_mW = 10.0
        wpe = 0.12
        elec = optical_mW / wpe
        assert elec == pytest.approx(83.33, rel=1e-3)

    def test_waterfall_total(self):
        """Sum of all loss components should equal total path loss."""
        coupling = 2.5
        routing = 0.5
        oma = 3.4
        mux = 2.0
        demux = 1.0
        channel = 3.0
        total = coupling + routing + oma + mux + coupling + channel + coupling + demux + routing
        assert total == pytest.approx(3 * coupling + 2 * routing + oma + mux + channel + demux)

    def test_pie_chart_coupling_group(self):
        """Fiber-Chip Coupling = coupling × 3 (TX, out, in)."""
        coupling = 2.5
        assert coupling * 3 == pytest.approx(7.5)


# ═══════════════════════════════════════════════════════════════════
# 7. RELIABILITY & MTBF
# ═══════════════════════════════════════════════════════════════════

class TestReliability:
    GPUS_RACK = 72
    RACKS_POD = 8
    PORTS_GPU = 18
    N_PODS = 16
    FR_PER_1K = 6.3

    @property
    def scaleup_links_rack(self):
        return self.GPUS_RACK * self.PORTS_GPU

    @property
    def scaleup_xcvrs_rack(self):
        return self.scaleup_links_rack * 2

    @property
    def xcvrs_pod(self):
        return self.scaleup_xcvrs_rack * self.RACKS_POD

    @property
    def total_xcvrs(self):
        return self.xcvrs_pod * self.N_PODS

    def test_scaleup_links(self):
        assert self.scaleup_links_rack == 1296

    def test_scaleup_xcvrs(self):
        """×2: GPU-side + switch-side transceivers in same rack."""
        assert self.scaleup_xcvrs_rack == 2592

    def test_xcvrs_pod(self):
        assert self.xcvrs_pod == 20736

    def test_total_xcvrs(self):
        assert self.total_xcvrs == 331776

    def test_failures_per_year(self):
        fr = self.FR_PER_1K / 1000
        fail_yr = self.total_xcvrs * fr
        assert fail_yr == pytest.approx(2090.2, rel=1e-3)

    def test_failures_per_day(self):
        fr = self.FR_PER_1K / 1000
        fail_day = self.total_xcvrs * fr / 365
        assert fail_day == pytest.approx(5.73, abs=0.01)

    def test_mtbf(self):
        fr = self.FR_PER_1K / 1000
        fail_yr = self.total_xcvrs * fr
        mtbf = 8760 / fail_yr
        assert mtbf == pytest.approx(4.19, abs=0.01)

    def test_reference_line_1_per_day(self):
        """MTBF = 24 hrs ⟺ 1 failure/day."""
        assert 8760 / 365 == pytest.approx(24.0)

    def test_reference_line_1_per_hour(self):
        """MTBF = 1 hr ⟺ 1 failure/hour."""
        assert 8760 / 8760 == pytest.approx(1.0)

    def test_scaleout_adds_correctly(self):
        scaleout = 128
        total_rack = self.scaleup_xcvrs_rack + scaleout
        assert total_rack == 2592 + 128

    def test_mtbf_formula_dimensional(self):
        """MTBF(hrs) = hours_per_year / failures_per_year."""
        hrs_yr = 8760
        fail_yr = 100
        mtbf = hrs_yr / fail_yr
        assert mtbf == pytest.approx(87.6)


# ═══════════════════════════════════════════════════════════════════
# 8. LATENCY MODEL
# ═══════════════════════════════════════════════════════════════════

class TestLatency:
    def test_nrz_no_fec_dac_adc(self):
        """NRZ should have zero latency for FEC, DAC, ADC blocks."""
        lat = latency_for(50, "NRZ", LAT_EXPONENTS_DEFAULT)
        assert lat["Lite FEC"] == 0
        assert lat["DAC TX"] == 0
        assert lat["ADC RX"] == 0

    def test_pam4_has_fec_dac_adc(self):
        """PAM4 should have nonzero FEC, DAC, ADC."""
        lat = latency_for(50, "PAM4", LAT_EXPONENTS_DEFAULT)
        assert lat["Lite FEC"] > 0
        assert lat["DAC TX"] > 0
        assert lat["ADC RX"] > 0

    def test_reference_baud_returns_ref_values(self):
        """At 50 Gbaud, latency = ref for PAM4 (all blocks active)."""
        lat = latency_for(50, "PAM4", LAT_EXPONENTS_DEFAULT)
        for k, v in LAT_REF.items():
            assert lat[k] == pytest.approx(v)

    def test_higher_baud_lower_latency(self):
        """Higher baud rate → lower latency (fewer ns per symbol)."""
        lat_50 = latency_for(50, "PAM4", LAT_EXPONENTS_DEFAULT)
        lat_100 = latency_for(100, "PAM4", LAT_EXPONENTS_DEFAULT)
        for k in LAT_REF:
            assert lat_100[k] <= lat_50[k]

    def test_scaling_formula(self):
        """lat = ref × (baud_ref/baud)^α."""
        baud = 100
        for k, ref_val in LAT_REF.items():
            alpha = LAT_EXPONENTS_DEFAULT[k]
            expected = ref_val * (BAUD_REF / baud) ** alpha
            lat = latency_for(baud, "PAM4", LAT_EXPONENTS_DEFAULT)
            assert lat[k] == pytest.approx(expected)

    def test_time_of_flight(self):
        """ToF = length × n_fiber / c."""
        c = 2.998e8
        n_fiber = 1.468
        length = 5.0
        tof_ns = length * n_fiber / c * 1e9
        assert tof_ns == pytest.approx(24.48, abs=0.05)

    def test_100g_pam4_uses_50_gbaud(self):
        """100G PAM4 → 50 Gbaud (2 bits/symbol)."""
        lat_100pam4 = latency_for(50, "PAM4", LAT_EXPONENTS_DEFAULT)
        lat_50nrz = latency_for(50, "NRZ", LAT_EXPONENTS_DEFAULT)
        assert lat_100pam4["Serializer"] == lat_50nrz["Serializer"]

    def test_200g_pam4_uses_100_gbaud(self):
        """200G PAM4 → 100 Gbaud."""
        lat = latency_for(100, "PAM4", LAT_EXPONENTS_DEFAULT)
        for k in LAT_REF:
            expected = LAT_REF[k] * (50 / 100) ** LAT_EXPONENTS_DEFAULT[k]
            assert lat[k] == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════
# 9. WHAT-IF PLANNER
# ═══════════════════════════════════════════════════════════════════

class TestWhatIfPlanner:
    OPTICS = {
        "mrm_16x100g_pam4": {"label": "MRM 16λ×100G PAM4", "n_lam": 16, "rate": 100, "fibers_per_dir": 1},
        "mrm_32x50g_nrz":   {"label": "MRM 32λ×50G NRZ",   "n_lam": 32, "rate": 50,  "fibers_per_dir": 1},
        "eam_8x200g_pam4":  {"label": "EAM 8λ×200G PAM4",  "n_lam": 8,  "rate": 200, "fibers_per_dir": 2},
    }

    def test_bw_per_fiber(self):
        op = self.OPTICS["mrm_16x100g_pam4"]
        bw_fib = op["n_lam"] * op["rate"] / 1000
        assert bw_fib == pytest.approx(1.6)

    def test_bw_per_fiber_32x50(self):
        op = self.OPTICS["mrm_32x50g_nrz"]
        bw_fib = op["n_lam"] * op["rate"] / 1000
        assert bw_fib == pytest.approx(1.6)

    def test_fibers_uni(self):
        rk = RACKS["nvl144"]
        op = self.OPTICS["mrm_16x100g_pam4"]
        bw_bi = rk["gpus"] * rk["bw_per_gpu_tbps"] * 2
        bw_fib = op["n_lam"] * op["rate"] / 1000
        fibers_uni = math.ceil(bw_bi / 2 / bw_fib)
        assert fibers_uni == 648

    def test_total_fibers_mrm(self):
        """MRM: fibers_per_dir=1 → total = fibers_uni × 2."""
        rk = RACKS["nvl144"]
        op = self.OPTICS["mrm_16x100g_pam4"]
        bw_bi = rk["gpus"] * rk["bw_per_gpu_tbps"] * 2
        bw_fib = op["n_lam"] * op["rate"] / 1000
        fibers_uni = math.ceil(bw_bi / 2 / bw_fib)
        total = fibers_uni * 2 * op["fibers_per_dir"]
        assert total == 1296

    def test_total_fibers_eam(self):
        """EAM: fibers_per_dir=2 → total = fibers_uni × 2 × 2."""
        rk = RACKS["nvl144"]
        op = self.OPTICS["eam_8x200g_pam4"]
        bw_bi = rk["gpus"] * rk["bw_per_gpu_tbps"] * 2
        bw_fib = op["n_lam"] * op["rate"] / 1000
        fibers_uni = math.ceil(bw_bi / 2 / bw_fib)
        total = fibers_uni * 2 * op["fibers_per_dir"]
        assert total == fibers_uni * 4

    def test_io_power(self):
        rk = RACKS["nvl144"]
        bw_bi = rk["gpus"] * rk["bw_per_gpu_tbps"] * 2
        pjb = 2.41
        overhead = 1.0
        io_W = pjb * bw_bi * overhead
        assert io_W == pytest.approx(2.41 * 2073.6, rel=1e-3)

    def test_percent_io(self):
        rk = RACKS["nvl144"]
        bw_bi = rk["gpus"] * rk["bw_per_gpu_tbps"] * 2
        pjb = 2.41
        io_W = pjb * bw_bi
        pct = io_W / (rk["rack_power_kw"] * 1000) * 100
        assert pct == pytest.approx(2.38, abs=0.05)

    def test_total_lanes(self):
        rk = RACKS["nvl144"]
        op = self.OPTICS["mrm_16x100g_pam4"]
        bw_bi = rk["gpus"] * rk["bw_per_gpu_tbps"] * 2
        lanes = int(bw_bi * 1000 / op["rate"])
        assert lanes == 20736


# ═══════════════════════════════════════════════════════════════════
# 10. UNIT CONVERSION SANITY
# ═══════════════════════════════════════════════════════════════════

class TestUnitConversions:
    def test_gbps_to_tbps(self):
        assert 1000 * 1e9 == 1e12

    def test_tbps_to_pbps(self):
        assert 1000 * 1e12 == 1e15

    def test_pbs_to_pbps(self):
        """1 PB/s = 8 Pbps."""
        assert 1 * 8 == 8

    def test_pjb_times_tbps_equals_watts(self):
        """1 pJ/b × 1 Tbps = 1e-12 J/b × 1e12 b/s = 1 W."""
        assert 1e-12 * 1e12 == pytest.approx(1.0)

    def test_mw_to_pjb(self):
        """mW per lambda / (rate Gbps) = pJ/b."""
        mw_per_lam = 20.0
        rate_gbps = 100.0
        pjb = mw_per_lam / rate_gbps
        assert pjb == pytest.approx(0.2)
