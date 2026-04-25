"""Smoke tests — verify Hydra configs load without errors.

Run with: ``pytest tests/test_configs.py -v``
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig


CONFIG_PATH = "../config"


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).parent.parent


def test_root_config_loads() -> None:
    """The root config.yaml should load without overrides."""
    with initialize(version_base="1.3", config_path=CONFIG_PATH):
        cfg = compose(config_name="config")
    assert cfg.project.name == "xai-vit-medical"
    assert cfg.project.phase == 1
    assert cfg.seed == 42


@pytest.mark.parametrize(
    "model_name",
    ["resnet50", "vit_base", "deit_base", "dinov2", "swin_base"],
)
def test_model_configs_load(model_name: str) -> None:
    """Each model config must compose with the root config."""
    with initialize(version_base="1.3", config_path=CONFIG_PATH):
        cfg = compose(config_name="config", overrides=[f"model={model_name}"])
    assert cfg.model.name == model_name
    assert cfg.model.architecture.num_classes == 8
    assert "training_overrides" in cfg.model


def test_vit_base_is_anchor() -> None:
    """ViT-Base must declare the SOTA anchor role (per CLAUDE.md)."""
    with initialize(version_base="1.3", config_path=CONFIG_PATH):
        cfg = compose(config_name="config", overrides=["model=vit_base"])
    assert cfg.model.role == "sota_anchor_xai"


@pytest.mark.parametrize(
    "xai_name",
    [
        "gradcam",
        "integrated_gradients",
        "attention_rollout",
        "generic_attention",
        "lrp",
        "sae",
        "activation_patching",
        "evaluation",
    ],
)
def test_xai_configs_load(xai_name: str) -> None:
    """Each XAI config must compose with the root config."""
    with initialize(version_base="1.3", config_path=CONFIG_PATH):
        cfg = compose(config_name="config", overrides=[f"xai={xai_name}"])
    assert cfg.xai is not None


def test_dataset_isic_config() -> None:
    """ISIC 2019 dataset config — sanity check structure."""
    with initialize(version_base="1.3", config_path=CONFIG_PATH):
        cfg = compose(config_name="config")
    assert cfg.dataset.name == "isic2019"
    assert cfg.dataset.num_classes == 8
    assert len(cfg.dataset.classes) == 8
    assert cfg.dataset.splits.group_by == "patient_id"  # CRITICAL


def test_no_phase2_dataset_present() -> None:
    """Phase 2 datasets must NOT be in config/dataset/."""
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "config" / "dataset"
    forbidden = {"brats2023.yaml", "chexpert.yaml", "camelyon17.yaml"}
    present = {p.name for p in dataset_dir.glob("*.yaml")}
    overlap = forbidden & present
    assert not overlap, f"Phase 2 datasets found in Phase 1 config: {overlap}"
