# llama-server Launcher

A simple GUI launcher for [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` using tktinker. 

## Requirements

- Python 3.10+ 
- `llama-server.exe` (from a llama.cpp build) placed in the same folder as the launcher
- Optional: `cloudflared.exe` in the same folder, only if using the tunnel feature

## How to run

Windows: just double-click `start.bat`.

Any OS:

```bash
python llama_launcher.py
```

Then pick your model and args per server, hit Start. Settings are saved to `llama_launcher_config.json` in the program folder.

