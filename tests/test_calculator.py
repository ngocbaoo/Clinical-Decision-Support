"""Unit tests for src/scoring/calculator.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scoring.calculator import (  # noqa: E402
    calculate_map, calculate_qsofa, calculate_news2, calculate_egfr,
    convert_creatinine_to_mgdl,
)


# ── MAP ────────────────────────────────────────────────
def test_map_normal():
    obs = {"systolic_bp": {"value": 120}, "diastolic_bp": {"value": 80}}
    r = calculate_map(obs)
    assert r["value"] == pytest.approx(93.3, abs=0.1)
    assert "Bình thường" in r["interpretation"]


def test_map_critically_low():
    obs = {"systolic_bp": {"value": 80}, "diastolic_bp": {"value": 50}}
    r = calculate_map(obs)
    assert r["value"] < 65
    assert "Nghiêm trọng" in r["interpretation"]


def test_map_missing_dbp():
    obs = {"systolic_bp": {"value": 120}}
    r = calculate_map(obs)
    assert r["missing"] is True
    assert r["value"] is None


# ── qSOFA ──────────────────────────────────────────────
def test_qsofa_positive():
    obs = {"gcs": {"value": 13}, "resp_rate": {"value": 24},
           "systolic_bp": {"value": 95}}
    r = calculate_qsofa(obs)
    assert r["total"] == 3
    assert r["positive"] is True


def test_qsofa_negative():
    obs = {"gcs": {"value": 15}, "resp_rate": {"value": 16},
           "systolic_bp": {"value": 120}}
    r = calculate_qsofa(obs)
    assert r["total"] == 0
    assert r["positive"] is False


def test_qsofa_partial():
    obs = {"gcs": {"value": 13}, "resp_rate": {"value": 25}}
    r = calculate_qsofa(obs)
    assert r["reliability"] == "PARTIAL"
    assert "systolic_bp" in r["missing_components"]


def test_qsofa_gcs_string_no_crash():
    """GCS recorded as FHIR valueCodeableConcept text must not raise."""
    obs = {"gcs": {"value": "10 (E2 V3 M5) - sedated"},
           "resp_rate": {"value": 18}, "systolic_bp": {"value": 120}}
    r = calculate_qsofa(obs)
    assert r["components"]["gcs"]["score"] == 1  # 10 < 15
    assert r["total"] == 1


# ── NEWS2 ──────────────────────────────────────────────
def test_news2_high_risk():
    obs = {"spo2": {"value": 88, "unit": "%"}, "resp_rate": {"value": 24},
           "heart_rate": {"value": 118}, "systolic_bp": {"value": 105},
           "temperature": {"value": 38.9}, "gcs": {"value": 15}}
    r = calculate_news2(obs, [], [], [], [])
    assert r["total"] >= 7
    assert r["risk_level"] == "HIGH"


def test_news2_low_risk():
    obs = {"spo2": {"value": 98, "unit": "%"}, "resp_rate": {"value": 16},
           "heart_rate": {"value": 72}, "systolic_bp": {"value": 120},
           "temperature": {"value": 37.0}, "gcs": {"value": 15}}
    r = calculate_news2(obs, [], [], [], [])
    assert r["total"] <= 2
    assert r["risk_level"] == "LOW"


# ── eGFR ───────────────────────────────────────────────
def test_egfr_normal():
    obs = {"creatinine": {"value": 80, "unit": "umol/L"}}
    pat = {"age": 40, "gender": "male"}
    r = calculate_egfr(obs, pat)
    assert r["egfr"] > 60
    assert r["dose_adjustment"] is False


def test_egfr_renal_impairment():
    obs = {"creatinine": {"value": 180, "unit": "umol/L"}}
    pat = {"age": 65, "gender": "male"}
    r = calculate_egfr(obs, pat)
    assert r["egfr"] < 60
    assert r["dose_adjustment"] is True
    assert "Stage 3" in r["stage"]


def test_egfr_missing():
    obs = {}
    pat = {"age": 65, "gender": "male"}
    r = calculate_egfr(obs, pat)
    assert r["missing"] is True
    assert r["egfr"] is None


def test_egfr_age_none_no_crash():
    """Patient with no birthDate (age None) must not raise."""
    obs = {"creatinine": {"value": 90, "unit": "umol/L"}}
    pat = {"age": None, "gender": "unknown"}
    r = calculate_egfr(obs, pat)
    assert r["missing"] is True
    assert r["egfr"] is None


def test_creatinine_conversion():
    assert convert_creatinine_to_mgdl(88.4, "umol/L") == pytest.approx(1.0, abs=0.01)
    assert convert_creatinine_to_mgdl(1.0, "mg/dL") == 1.0
