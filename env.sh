# Usage: source env.sh   (must be sourced, not executed)
export BIG=/goinfre/$USER

# uv
export UV_CACHE_DIR=$BIG/uv-cache
export UV_PYTHON_INSTALL_DIR=$BIG/uv-python
# export UV_PROJECT_ENVIRONMENT=$BIG/rag-venv   # the venv (torch lives here)
export UV_LINK_MODE=copy

# Hugging Face / PyTorch / pip / temp files / generic ~/.cache
export HF_HOME=$BIG/hf
export TORCH_HOME=$BIG/torch
export PIP_CACHE_DIR=$BIG/pip-cache
export TMPDIR=$BIG/tmp
export XDG_CACHE_HOME=$BIG/cache

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$TORCH_HOME" \
         "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
