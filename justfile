# crustytts — Claude Code Stop hook TTS announcer

# Build and run a test invocation (foreground mode, no real transcript)
run:
    cargo run --release

# Speak arbitrary text directly (bypasses transcript + Ollama). Usage: just say "hello world"
say text:
    cargo run --release -- --say {{text}}

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
