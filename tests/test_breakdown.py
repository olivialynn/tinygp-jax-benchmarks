from __future__ import annotations

import numpy as np

from benchmarks.breakdown import _compile_callable
from benchmarks.breakdown_compare import render_markdown
from benchmarks.scenarios import get_scenario_specs


def test_get_scenario_specs_matches_current_quasisep_set() -> None:
    specs = get_scenario_specs()

    assert set(specs) == {
        "quasisep_cpu",
        "quasisep_matern32_cpu",
        "quasisep_matern52_cpu",
    }


def test_breakdown_render_markdown_includes_stage_ratios() -> None:
    baseline = {
        "jax": "0.4.31",
        "profile": "smoke",
        "results": [
            {
                "scenario": "quasisep_cpu",
                "n": 50,
                "stage": "jit_compile",
                "samples": 1,
                "median_s": 1.1,
                "mean_s": 1.1,
                "stdev_s": 0.0,
            }
        ],
    }
    candidate = {
        "jax": "0.9.1",
        "profile": "smoke",
        "results": [
            {
                "scenario": "quasisep_cpu",
                "n": 50,
                "stage": "jit_compile",
                "samples": 1,
                "median_s": 2.2,
                "mean_s": 2.2,
                "stdev_s": 0.0,
            }
        ],
    }

    markdown = render_markdown(baseline, candidate)

    assert "| scenario | n | stage | baseline mean (s) | candidate mean (s) | ratio |" in markdown
    assert "| quasisep_cpu | 50 | jit_compile | 1.100000 | 2.200000 | 2.000 |" in markdown


def test_compile_callable_returns_compiled_executable() -> None:
    compiled_fn = _compile_callable(lambda x: x + 1, np.array([1.0, 2.0]))

    result = compiled_fn(np.array([3.0, 4.0]))

    np.testing.assert_allclose(np.asarray(result), np.array([4.0, 5.0]))


def test_compile_callable_supports_pytree_arguments() -> None:
    compiled_fn = _compile_callable(
        lambda payload, x: payload["scale"] * x + payload["offset"],
        {"scale": np.array([2.0, 3.0]), "offset": np.array([1.0, 1.0])},
        np.array([4.0, 5.0]),
    )

    result = compiled_fn(
        {"scale": np.array([2.0, 3.0]), "offset": np.array([1.0, 1.0])},
        np.array([4.0, 5.0]),
    )

    np.testing.assert_allclose(np.asarray(result), np.array([9.0, 16.0]))


def test_compile_callable_supports_pytree_outputs() -> None:
    compiled_fn = _compile_callable(
        lambda x: {"left": x + 1, "right": x * 2},
        np.array([1.0, 2.0]),
    )

    result = compiled_fn(np.array([3.0, 4.0]))

    np.testing.assert_allclose(np.asarray(result["left"]), np.array([4.0, 5.0]))
    np.testing.assert_allclose(np.asarray(result["right"]), np.array([6.0, 8.0]))
