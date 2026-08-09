#!/bin/bash
# Install the local-only audit toolkit into your Claude Code config directory.
# Override the destination with CLAUDE_DIR=/some/path ./install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CLAUDE_DIR:-$HOME/.claude}"

mkdir -p "$DEST/commands" "$DEST/scripts"

install_file() {
    local src="$1"
    local dst="$2"

    if [[ -f "$dst" ]] && ! cmp -s "$src" "$dst"; then
        echo "  ! $dst differs — backing up to $(basename "$dst").bak"
        cp "$dst" "$dst.bak"
    fi

    echo "  + $dst"
    cp "$src" "$dst"
}

echo "Installing into $DEST"
install_file "$SCRIPT_DIR/commands/local-audit.md" "$DEST/commands/local-audit.md"
install_file "$SCRIPT_DIR/scripts/local_audit.py" "$DEST/scripts/local_audit.py"
install_file "$SCRIPT_DIR/scripts/probe_models.py" "$DEST/scripts/probe_models.py"

chmod +x "$DEST/scripts/local_audit.py" "$DEST/scripts/probe_models.py"

cat <<EOF

Next steps

  1. Create your routing config:
       mkdir -p ~/.config/ollama_mcp
       cp examples/configs/routes.local-only.json ~/.config/ollama_mcp/routes.json
     Then edit the "url" and the model names to match your own Ollama host.

  2. Find out which of your models can actually do this:
       python3 $DEST/scripts/probe_models.py
     It prints a routes.json snippet naming the models that passed. Do not
     skip this — model behaviour varies wildly and fails silently.

  3. Restart Claude Code so /local-audit is picked up, then run it in any
     repository:
       /local-audit
EOF
