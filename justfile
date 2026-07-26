# crustytts — Claude Code Stop hook TTS announcer

# Build and run a test invocation (foreground mode, no real transcript)
run:
    cargo run --release

# Speak arbitrary text directly (bypasses transcript + Ollama). Usage: just say "hello world"
say text:
    cargo run --release -- --say {{text}}

# Summarize and speak a CherryPi transcript. Usage: just transcript path cwd
transcript path cwd:
    cargo run --release -- --transcript {{path}} --cwd {{cwd}}

# Ensure crustytts is installed and up-to-date on PATH, then configure the hook.
# Writes "crustytts" (bare name) to settings.json so Claude Code resolves it via PATH.
setup: _ensure-installed
    cargo run --release -- --setup

# Install the binary to ~/.cargo/bin so it's on PATH
install: build
    cargo install --path .

# Build only
build:
    cargo build --release

# Validate the Open Plugins manifest and plugin structure
check-plugin:
    @echo "=== Plugin Structure ==="
    @find plugin -type f | sort
    @echo
    @echo "=== Manifest ==="
    @python3 -c "import json; m=json.load(open('plugin/.plugin/plugin.json')); print(f'Name: {m[\"name\"]}'); print(f'Version: {m.get(\"version\",\"N/A\")}'); print(f'Description: {m.get(\"description\",\"N/A\")}')"
    @echo
    @echo "=== Hooks ==="
    @python3 -c "import json; h=json.load(open('plugin/hooks/hooks.json')); [print(f'  {e}: {a[\"type\"]} -> {a[\"command\"]}') for e,rs in h['hooks'].items() for r in rs for a in r['hooks']]"

# Validate JSON syntax of all config files
validate-plugin:
    @python3 -c "import json; json.load(open('plugin/.plugin/plugin.json')); print('\u2705 plugin.json')"
    @python3 -c "import json; json.load(open('plugin/hooks/hooks.json')); print('\u2705 hooks.json')"

# Link plugin into CherryPi's plugins directory for discovery
plugin-link:
    @mkdir -p ../CherryPi/plugins
    ln -sfn $(pwd)/plugin ../CherryPi/plugins/crustytts
    @echo "Linked plugin -> ../CherryPi/plugins/crustytts"

# Internal: install if missing or outdated
_ensure-installed:
    @if ! command -v crustytts >/dev/null 2>&1; then \
        echo "crustytts not found on PATH — installing..."; \
        just install; \
    elif [ target/release/crustytts -nt "$(command -v crustytts)" ]; then \
        echo "installed crustytts is older than current build — reinstalling..."; \
        just install; \
    else \
        echo "crustytts is up-to-date on PATH"; \
    fi
