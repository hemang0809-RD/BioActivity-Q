"""Tests for the prediction and AD logic."""
from src.inference.predict import ApplicabilityDomain


def test_ad_label_in_domain():
    assert ApplicabilityDomain.label(0.5) == "IN DOMAIN"
    assert ApplicabilityDomain.label(0.40) == "IN DOMAIN"
    assert ApplicabilityDomain.label(1.0) == "IN DOMAIN"


def test_ad_label_borderline():
    assert ApplicabilityDomain.label(0.3) == "BORDERLINE"
    assert ApplicabilityDomain.label(0.25) == "BORDERLINE"
    assert ApplicabilityDomain.label(0.39) == "BORDERLINE"


def test_ad_label_out():
    assert ApplicabilityDomain.label(0.1) == "OUT OF DOMAIN"
    assert ApplicabilityDomain.label(0.0) == "OUT OF DOMAIN"
    assert ApplicabilityDomain.label(0.24) == "OUT OF DOMAIN"
