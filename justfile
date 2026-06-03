# Task runner for the agent-memory research repo.
# Run `just` (no args) to see all available recipes.
# Requires: uv (https://docs.astral.sh/uv/), just (https://just.systems/)

# Show available recipes (default)
default:
    @just --list

# === Setup ===

# Install all dependencies (creates .venv automatically via uv)
install:
    uv sync --all-groups

# Register the Jupyter kernel so it appears in JupyterLab / VS Code
kernel:
    uv run python -m ipykernel install --user --name research --display-name research

# Install pre-commit hooks (nbstripout, trailing whitespace, etc.)
hooks:
    uv run pre-commit install

# Complete first-time setup: install + kernel + hooks
setup: install kernel hooks
    @echo "✓ Setup complete. Run 'just lab' to start JupyterLab."

# === Running experiments ===

# Start JupyterLab in the current directory (binds 0.0.0.0 so it's reachable from a host browser when running in a container / VM)
lab:
    uv run jupyter lab --ip=0.0.0.0 --no-browser

# Run a specific notebook headless (e.g. just run experiments/embeddings/01-anisotropy.py)
run NOTEBOOK:
    uv run jupytext --to ipynb {{NOTEBOOK}}
    uv run jupyter nbconvert --execute --to notebook --inplace "$(echo {{NOTEBOOK}} | sed 's/\.py$/.ipynb/')"

# Convert a single .py to a paired .ipynb (without running)
to-ipynb FILE:
    uv run jupytext --to ipynb {{FILE}}

# Convert all experiment .py files to .ipynb (so they can be opened in JupyterLab)
convert-all:
    @find experiments -name "*.py" -not -path "*/shared/*" -not -path "*/__pycache__/*" -exec uv run jupytext --to ipynb {} \;

# Sync .py and .ipynb in both directions for a given file
sync-nb FILE:
    uv run jupytext --sync {{FILE}}

# === Maintenance ===

# Update the lock file (uv.lock) without installing
lock:
    uv lock

# Sync the environment from the lock file (after pulling new changes)
sync:
    uv sync --all-groups

# Strip outputs from all tracked .ipynb files
strip:
    @find experiments -name "*.ipynb" -exec uv run nbstripout {} \; 2>/dev/null || true

# Clean all generated files: venv, caches, generated notebooks
clean:
    rm -rf .venv .ipynb_checkpoints
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name .ipynb_checkpoints -type d -exec rm -rf {} + 2>/dev/null || true
    find experiments -name "*.ipynb" -delete 2>/dev/null || true

# Show installed packages
deps:
    uv pip list

# Show Python version in use
python:
    uv run python --version

# === Quick experiments ===

# Run the anisotropy experiment (notebook 01) end-to-end
anisotropy:
    just run experiments/embeddings/01-anisotropy.py

# Run the distance-ablation experiment (notebook 02) end-to-end
distance-ablation:
    just run experiments/embeddings/02-distance-ablation.py
