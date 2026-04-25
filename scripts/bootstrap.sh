#!/usr/bin/env bash
# =====================================================================
# bootstrap.sh — One-shot environment setup for xai-vit-medical
# =====================================================================
# Usage:
#   bash scripts/bootstrap.sh           # full setup
#   bash scripts/bootstrap.sh --no-env  # skip conda env creation
# =====================================================================

set -euo pipefail

ENV_NAME="xai-vit"
PYTHON_VERSION="3.11"

# Pretty colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================================${NC}"
echo -e "${BLUE}  xai-vit-medical — Phase 1 Bootstrap${NC}"
echo -e "${BLUE}========================================================${NC}"

# ---- Check we're at project root ----
if [[ ! -f "CLAUDE.md" ]]; then
    echo -e "${RED}[ERROR]${NC} Run from project root (where CLAUDE.md lives)."
    exit 1
fi

# ---- 1. Conda environment ----
if [[ "${1:-}" != "--no-env" ]]; then
    echo -e "\n${YELLOW}[1/5]${NC} Creating conda environment '${ENV_NAME}'..."

    if ! command -v conda &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} conda not found. Install Miniconda/Anaconda first."
        exit 1
    fi

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo -e "${YELLOW}[skip]${NC} Env '${ENV_NAME}' already exists."
    else
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi

    # Source conda for activation in script
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

# ---- 2. Install dependencies ----
echo -e "\n${YELLOW}[2/5]${NC} Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# ---- 3. Set up .env ----
echo -e "\n${YELLOW}[3/5]${NC} Setting up .env..."
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo -e "${YELLOW}[!]${NC} Created .env from template — EDIT IT before running experiments."
else
    echo -e "${GREEN}[ok]${NC} .env already exists."
fi

# ---- 4. Validate Hydra configs ----
echo -e "\n${YELLOW}[4/5]${NC} Validating Hydra configs..."
python -c "
from hydra import compose, initialize
with initialize(version_base='1.3', config_path='config'):
    cfg = compose(config_name='config')
print(f'  [ok] Root config — phase {cfg.project.phase}')
for m in ['resnet50', 'vit_base', 'deit_base', 'dinov2', 'swin_base']:
    with initialize(version_base='1.3', config_path='config'):
        cfg = compose(config_name='config', overrides=[f'model={m}'])
    print(f'  [ok] model={m} role={cfg.model.role}')
" || { echo -e "${RED}[FAIL]${NC} Config validation failed."; exit 1; }

# ---- 5. Run smoke tests ----
echo -e "\n${YELLOW}[5/5]${NC} Running smoke tests..."
pytest tests/test_configs.py -v --tb=short || {
    echo -e "${RED}[FAIL]${NC} Some tests failed."
    exit 1
}

# ---- Summary ----
echo -e "\n${GREEN}========================================================${NC}"
echo -e "${GREEN}  Bootstrap complete!${NC}"
echo -e "${GREEN}========================================================${NC}"
echo -e "Next steps:"
echo -e "  1. Activate env:    ${BLUE}conda activate ${ENV_NAME}${NC}"
echo -e "  2. Edit .env:       ${BLUE}nvim .env${NC} (set WANDB_API_KEY)"
echo -e "  3. Place ISIC data: ${BLUE}data/isic2019/${NC} (see data/README.md)"
echo -e "  4. Implement stubs: ${BLUE}src/data/isic_dataset.py${NC}"
echo -e "  5. Train baseline:  ${BLUE}python -m src.training.trainer model=resnet50${NC}"
echo -e "  6. Train ViT:       ${BLUE}python -m src.training.trainer model=vit_base${NC}"
