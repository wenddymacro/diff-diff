"""Tests for CIParams bootstrap scaling in conftest.py."""

import math

import tests.conftest as conftest_module
from tests.conftest import CIParams


class TestCIParamsBootstrap:
    def test_min_n_capped_at_49_in_pure_python_mode(self, monkeypatch):
        """min_n is capped at 49 in pure Python mode."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(499, min_n=199) == 49

    def test_min_n_passthrough_in_rust_mode(self, monkeypatch):
        """min_n has no effect when Rust backend is available."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", False)
        assert CIParams.bootstrap(499, min_n=199) == 499

    def test_min_n_cap_then_n_cap(self, monkeypatch):
        """min_n cap (49) applies, then result is min(n, effective_floor)."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        # effective_min = min(199, 49) = 49; max(49, 16) = 49; min(100, 49) = 49
        assert CIParams.bootstrap(100, min_n=199) == 49

    def test_n_lte_10_ignores_min_n(self, monkeypatch):
        """n <= 10 always returns n regardless of min_n or mode."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(10, min_n=199) == 10

    def test_default_min_n_preserves_existing_behavior(self, monkeypatch):
        """Default min_n=11 matches pre-change behavior."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(499) == max(11, int(math.sqrt(499) * 1.6))  # 35

    def test_min_n_cap_with_high_min_n(self, monkeypatch):
        """min_n=249 is also capped at 49 in pure Python mode."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        assert CIParams.bootstrap(499, min_n=249) == 49

    def test_n_still_caps_result(self, monkeypatch):
        """Original n still caps the result when min_n is below cap."""
        monkeypatch.setattr(conftest_module, "_PURE_PYTHON_MODE", True)
        # effective_min = min(40, 49) = 40; max(40, 8) = 40; min(30, 40) = 30
        assert CIParams.bootstrap(30, min_n=40) == 30
