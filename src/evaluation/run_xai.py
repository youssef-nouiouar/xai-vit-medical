"""XAI evaluation entry point.

Usage:
    python -m src.evaluation.run_xai \\
        model=deit_base \\
        xai=generic_attention \\
        checkpoint=outputs/models/deit_base_best.pth

    # Multirun across XAI methods:
    python -m src.evaluation.run_xai -m \\
        model=deit_base \\
        xai=gradcam,integrated_gradients,attention_rollout,generic_attention,lrp
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Pipeline:

    1. Load checkpoint of trained model.
    2. Build test dataloader.
    3. Run the chosen XAI method on test set.
    4. Compute faithfulness (Insertion / Deletion AUC).
    5. Compute localization (Pointing Game, IoU vs lesion mask).
    6. Run sanity checks (model randomization).
    7. Save: saliency arrays, overlays, metrics CSV, summary JSON.
    8. Log to W&B.
    """
    raise NotImplementedError("Implement in this file.")


if __name__ == "__main__":
    main()
