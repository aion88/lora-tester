import argparse

import pytest

import lora_test
import comfy_workflow_doctor as doctor


def test_parse_strengths_accepts_numbers_and_spaces():
    assert lora_test.parse_strengths("0, 0.5,1.0") == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("bad", ["", ",", "a,b", "0.5;1"])
def test_parse_strengths_rejects_bad_input(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        lora_test.parse_strengths(bad)


def test_parse_args_defaults_and_overrides(tmp_path):
    args = lora_test.parse_args([])
    assert args.lora is None and args.strengths is None and args.notes == ""
    args = lora_test.parse_args(
        ["--lora", "a.safetensors", "--strengths", "0,1", "--name", "x", "--out", str(tmp_path)]
    )
    assert args.lora == "a.safetensors"
    assert args.strengths == [0.0, 1.0]
    assert args.out == str(tmp_path)


def test_grid_has_one_cell_per_prompt_and_strength():
    cells = lora_test.plan_grid(["a", "b", "c"], [0.0, 0.5, 1.0], base_seed=42)
    assert len(cells) == 9
    assert {(c["prompt_index"], c["strength"]) for c in cells} == {
        (p, s) for p in range(3) for s in (0.0, 0.5, 1.0)
    }


def test_seed_depends_on_prompt_only():
    cells = lora_test.plan_grid(["a", "b"], [0.0, 0.5, 1.0], base_seed=42)
    for c in cells:
        assert c["seed"] == 42 + c["prompt_index"]


def test_workflow_skips_lora_loader_at_zero_strength():
    wf = lora_test.build_workflow("p", "x.safetensors", 0.0, 1, "pre")
    assert all(n["class_type"] != "LoraLoader" for n in wf.values())


def test_workflow_applies_strength_to_model_and_clip():
    wf = lora_test.build_workflow("p", "x.safetensors", 0.75, 1, "pre")
    lora = next(n for n in wf.values() if n["class_type"] == "LoraLoader")
    assert lora["inputs"]["strength_model"] == lora["inputs"]["strength_clip"] == 0.75
    assert lora["inputs"]["lora_name"] == "x.safetensors"


def test_no_credentials_in_defaults():
    import re

    text = open(lora_test.__file__).read()
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "/Users/" not in text


def test_doctor_flags_flux_controlnet_without_vae():
    api = {
        "1": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "flux1-union-pro.safetensors"}},
        "2": {"class_type": "ControlNetApplyAdvanced", "inputs": {"control_net": ["1", 0]}},
    }
    kinds = [f["kind"] for f in doctor.static_lint_api(api)]
    assert kinds == ["missing_cn_vae"]


def test_doctor_ignores_console_noise():
    assert doctor.is_noise("GET /favicon.ico 404")
    assert not doctor.is_noise("Cannot execute because a node is missing")
