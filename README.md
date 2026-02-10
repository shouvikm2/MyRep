# SmartCam v1

A real-time AI camera system that uses a local Ollama vision model to continuously monitor scenes, remember what it observes, respond to voice commands, and record video on demand. Designed as the foundation for a progression toward a smart dashcam (v1.1) and an autonomous robot platform (v2).

---

## Features

- **AI Scene Analysis**: Uses a local Ollama vision model (`llava-phi3`) to describe what it sees in real time
- **Motion-Triggered Analysis**: AI runs only on significant motion or after a configurable interval — no wasted processing
- **Scene Memory**: Persistently stores observations to a JSON file on disk, zone-aware for multi-camera setups
- **Memory-Augmented Prompts**: Injects recent observations into each prompt so the AI notices changes and missing items
- **Voice Commands**: Full offline voice control via vosk (no internet required)
- **Voice Output (TTS)**: Speaks responses — pyttsx3 on Windows, espeak-ng on Linux, piper (neural) optional
- **Video Recording**: Timestamped MP4 recording toggled by voice or keyboard
- **Headless Mode**: Runs without a display for Jetson production deployment
- **Runtime Tuning**: Adjust motion sensitivity and check interval live via voice
- **Video Enhancement**: Optional CLAHE contrast boost and bilateral filter denoising

---

## Hardware Target

- **Development / Testing**: Any Windows or Linux PC with a webcam
- **Production**: NVIDIA Jetson Orin 8GB unified memory

---

## Requirements

### Software
- Python 3.10+
- Ollama running locally with `llava-phi3` pulled
- OpenCV, requests, numpy

```bash
pip install opencv-python requests numpy
```

### TTS
- **Windows**: `pip install pyttsx3` (uses built-in SAPI voice, no extra install)
- **Linux / Jetson**: `sudo apt install espeak-ng`
- **Optional neural TTS**: piper (see `--piper-model`)

### Voice Commands (optional)
```bash
pip install vosk pyaudio
```
Download a vosk model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
(recommended: `vosk-model-small-en-us-0.15`, ~50MB)

---

## Setup

1. **Install and run Ollama:**
   ```bash
   ollama serve
   ollama pull llava-phi3
   ```

2. **Install Python dependencies** (see Requirements above)

3. **Connect camera** — USB webcam (index `0`) or IP camera URL

---

## Usage

**Minimal — just works:**
```bash
python SmartCam.py 0
```

**With scene memory:**
```bash
python SmartCam.py 0 --memory-path C:\AIsanity\memory.json
```

**With voice commands:**
```bash
python SmartCam.py 0 --memory-path C:\AIsanity\memory.json --vosk-model C:\vosk-models\vosk-model-small-en-us-0.15
```

**Headless (Jetson production):**
```bash
python SmartCam.py 0 --headless --memory-path /mnt/ssd/memory.json --vosk-model /opt/vosk/vosk-model-small-en-us-0.15
```

**IP camera source:**
```bash
python SmartCam.py http://192.168.1.109:8080/video
```

---

## Voice Commands

| Say | Action |
|-----|--------|
| `camera what do you see` / `camera look again` | Trigger fresh analysis and speak result |
| `camera repeat` | Speak last observation again |
| `camera status` | Speak recording / sleep / mute state |
| `camera record on` / `camera record off` | Start / stop recording |
| `camera are you recording` | Yes / no response |
| `camera remember this` | Force-save current view to memory |
| `camera what have you seen` | Read back last 3 memory entries |
| `camera go to sleep` / `camera wake up` | Pause / resume all analysis |
| `camera mute` / `camera unmute` | Silence / restore TTS responses |
| `camera more sensitive` / `camera less sensitive` | Adjust motion threshold live |
| `camera check more often` / `camera check less often` | Adjust check interval live |
| `camera help` | Read all available commands |

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Toggle recording |

---

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `source` | *(required)* | Camera index (e.g. `0`) or IP camera URL |
| `--model` | `llava-phi3` | Ollama vision model |
| `--prompt` | *(built-in)* | Override the default prompt |
| `--interval` | `10` | Seconds between periodic checks |
| `--motion-threshold` | `5000` | Changed pixels to trigger motion analysis |
| `--headless` | off | Run without display window |
| `--enhance-video` | off | CLAHE contrast enhancement |
| `--denoise` | off | Bilateral filter denoising |
| `--timeout` | `30` | Ollama API timeout in seconds |
| `--piper-model` | *(none)* | Path to piper .onnx model for neural TTS |
| `--memory-path` | *(none)* | Path to JSON memory file on SSD |
| `--zone` | `default` | Memory zone label for this camera view |
| `--memory-context` | `3` | Past observations injected into each prompt |
| `--vosk-model` | *(none)* | Path to vosk model dir for voice commands |
| `--record-dir` | `./recordings` | Directory for recorded video files |
| `--ollama-url` | `http://localhost:11434/api/generate` | Ollama API endpoint |

---

## Architecture

```
Main Thread (UI + capture + voice command dispatch)
├── AI Worker Thread (motion detection → analysis trigger)
│   └── Analysis Thread (Ollama API call, memory write, TTS)
└── Voice Listener Thread (mic → vosk STT → command queue)
```

- Resolution: 480×360 (balanced for object ID + payload size)
- JPEG quality: 65 for API transmission
- Memory: bounded at 100 observations per zone, ~20KB JSON file

---

## Roadmap

| Version | Features |
|---------|----------|
| **v1** (current) | SmartCam — AI vision, memory, voice I/O, recording |
