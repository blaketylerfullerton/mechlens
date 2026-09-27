#!/bin/sh
# Install Mechlens on a GPU machine and connect it to Mechlens Cloud.
#
#   curl -fsSL https://raw.githubusercontent.com/blaketylerfullerton/mechlens/master/install.sh | sh
#
# Anything after `sh -s --` is passed to `mechlens serve`, e.g.
#
#   curl -fsSL .../install.sh | sh -s -- --model gemma-2-2b
#   curl -fsSL .../install.sh | sh -s -- --cloud http://localhost:5175
#
# Re-running upgrades in place. Nothing is installed outside $MECHLENS_HOME
# except uv (in ~/.local/bin) when it is missing.
#
# Environment:
#   MECHLENS_HOME   where the venv lives (default: ~/.mechlens)
#   MECHLENS_CLOUD  cloud to pair with when --cloud is not given
#   MECHLENS_REF    git branch, tag or commit to install (default: master)
#   MECHLENS_SOURCE package to install instead, e.g. a local checkout's path
set -eu

MECHLENS_HOME=${MECHLENS_HOME:-"$HOME/.mechlens"}
MECHLENS_CLOUD=${MECHLENS_CLOUD:-https://mechlens-cloud-fgstc.ondigitalocean.app}
MECHLENS_REF=${MECHLENS_REF:-master}
MECHLENS_SOURCE=${MECHLENS_SOURCE:-"mechlens @ git+https://github.com/blaketylerfullerton/mechlens@$MECHLENS_REF"}
VENV="$MECHLENS_HOME/venv"

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is required (apt-get install -y git)"
command -v curl >/dev/null 2>&1 || die "curl is required"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    say "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    say "No NVIDIA GPU found; Mechlens will run on the CPU, slowly"
fi

# uv fetches Python itself and picks the torch build that matches the CUDA
# driver, so the box's own Python and torch (if any) are left alone.
if command -v uv >/dev/null 2>&1; then
    UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null
    UV="$HOME/.local/bin/uv"
fi

if [ ! -x "$VENV/bin/python" ]; then
    say "Creating $VENV"
    "$UV" venv --quiet --python 3.12 "$VENV"
fi

say "Installing $MECHLENS_SOURCE; the first install downloads torch and takes a few minutes"
"$UV" pip install --python "$VENV/bin/python" --torch-backend auto --upgrade \
    "$MECHLENS_SOURCE"

case " $* " in
    *" --cloud "* | *" --cloud="*) ;;
    *) set -- --cloud "$MECHLENS_CLOUD" "$@" ;;
esac

if [ -n "${SSH_CONNECTION:-}" ] && [ -z "${TMUX:-}${STY:-}" ]; then
    say "Tip: closing this SSH session stops the GPU. Run this inside tmux to keep it going."
fi
say "Starting mechlens serve (run it again later with: $VENV/bin/mechlens serve $*)"
exec "$VENV/bin/mechlens" serve "$@"
