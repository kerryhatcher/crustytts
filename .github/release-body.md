# crustytts

Text-to-speech notifications for AI coding agents using Kokoro ONNX TTS.

## Downloads

| Platform | Architecture | Archive |
|---|---|---|
| Linux | x86_64 | `crustytts-x86_64-unknown-linux-gnu.tar.gz` |
| macOS | Apple Silicon | `crustytts-aarch64-apple-darwin.tar.gz` |
| Windows | x86_64 | `crustytts-x86_64-pc-windows-msvc.zip` |

## Usage

```bash
# Extract
tar xzf crustytts-*.tar.gz    # Linux / macOS
unzip crustytts-*.zip          # Windows

# Set up the Claude Code Stop hook
./crustytts --setup
```

See [README.md](https://github.com/kerryhatcher/crustytts#readme) for full documentation.
