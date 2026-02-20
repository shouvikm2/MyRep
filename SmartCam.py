import subprocess
import requests
import base64
import time
import json
import os
import re
import numpy as np
import argparse
import cv2
import queue
import threading
import sys
import platform
import logging
from collections import deque
from datetime import datetime
import signal


# --- Health logging ---
def _setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Setup file + console logging for production monitoring."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("smartcam")
    logger.setLevel(logging.INFO)
    # Rotating-style: new file per day
    log_file = os.path.join(log_dir, f"smartcam_{datetime.now().strftime('%Y%m%d')}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    # Also log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(ch)
    return logger

LOG = _setup_logging()


class HealthMonitor:
    """Tracks system health metrics and logs periodic heartbeats."""
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.start_time = time.time()
        self.analysis_count = 0
        self.error_count = 0
        self.consecutive_500s = 0
        self.last_analysis_time = 0
        self.last_heartbeat = 0
        self._lock = threading.Lock()

    def record_analysis(self):
        with self._lock:
            self.analysis_count += 1
            self.last_analysis_time = time.time()
            self.consecutive_500s = 0

    def record_error(self, error_type: str = ""):
        with self._lock:
            self.error_count += 1
            if error_type == "500":
                self.consecutive_500s += 1
            else:
                self.consecutive_500s = 0

    def get_consecutive_500s(self) -> int:
        with self._lock:
            return self.consecutive_500s

    def heartbeat(self):
        """Log periodic health status. Call from main loop."""
        now = time.time()
        if now - self.last_heartbeat < self.interval:
            return
        self.last_heartbeat = now
        uptime = int(now - self.start_time)
        uptime_str = f"{uptime // 3600}h{(uptime % 3600) // 60}m"
        # Memory usage
        try:
            with open('/proc/meminfo') as f:
                meminfo = f.read()
            total = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1)) // 1024
            avail = int(re.search(r'MemAvailable:\s+(\d+)', meminfo).group(1)) // 1024
            mem_str = f"{avail}MB free / {total}MB total"
        except Exception:
            mem_str = "unknown"

        with self._lock:
            last_ago = int(now - self.last_analysis_time) if self.last_analysis_time else -1
            msg = (f"HEARTBEAT | uptime={uptime_str} | analyses={self.analysis_count} | "
                   f"errors={self.error_count} | last_analysis={last_ago}s ago | mem={mem_str}")
        LOG.info(msg)
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

HEALTH = HealthMonitor()


def ensure_ollama_running(ollama_url: str, model: str, startup_timeout: int = 60) -> bool:
    """
    Check if Ollama is running; start it if not.
    Returns True if Ollama is ready, False if it could not be started.
    Called BEFORE main() so the app never enters the capture loop with a dead API.
    """
    base_url = ollama_url.rsplit('/api/', 1)[0] 

    def _is_ready() -> bool:
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _systemd_unit_exists() -> bool:
        try:
            result = subprocess.run(
                ['systemctl', 'is-enabled', 'ollama.service'],
                capture_output=True, text=True, timeout=5
            )
            return 'not-found' not in result.stdout and 'not-found' not in result.stderr
        except Exception:
            return False

    if _is_ready():
        print("[Ollama] Already running.")
        _ensure_model_pulled(base_url, model)
        return True

    use_systemd = _systemd_unit_exists()

    if use_systemd:
        print("[Ollama] Not responding — restarting via systemd ...")
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', 'ollama.service'],
                           capture_output=True, timeout=15)
        except Exception as e:
            print(f"[Ollama] ERROR restarting service: {e}", file=sys.stderr)
            return False
    else:
        print("[Ollama] Not running — attempting to start with 'ollama serve' ...")
        try:
            subprocess.run(['pkill', '-f', 'ollama serve'], capture_output=True, timeout=5)
            time.sleep(1)
        except Exception:
            pass
        try:
            with open('/tmp/ollama_serve.log', 'a') as log:
                subprocess.Popen(
                    ['ollama', 'serve'],
                    stdout=log,
                    stderr=log,
                    start_new_session=True 
                )
        except FileNotFoundError:
            print("[Ollama] ERROR: 'ollama' binary not found. Install from https://ollama.ai", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[Ollama] ERROR starting ollama serve: {e}", file=sys.stderr)
            return False

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _is_ready():
            print("[Ollama] Server is up.")
            _ensure_model_pulled(base_url, model)
            return True
        time.sleep(1)
        remaining = int(deadline - time.time())
        print(f"[Ollama] Waiting for server... ({remaining}s)", end='\r', flush=True)

    print(f"\n[Ollama] ERROR: Server did not respond within {startup_timeout}s.", file=sys.stderr)
    return False


def _ensure_model_pulled(base_url: str, model: str):
    """Pull the model if it is not already present."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            names = [m.get('name', '') for m in r.json().get('models', [])]
            if any(model in n for n in names):
                print(f"[Ollama] Model '{model}' already present.")
                return
        print(f"[Ollama] Pulling model '{model}' — this may take several minutes ...")
        subprocess.run(['ollama', 'pull', model], check=False)
        print(f"[Ollama] Model '{model}' ready.")
    except Exception as e:
        print(f"[Ollama] WARNING: Could not verify/pull model: {e}", file=sys.stderr)


def _detect_capabilities() -> dict:
    """Probe hardware acceleration available on this platform."""
    caps = {}
    caps['is_jetson'] = (platform.machine() == "aarch64"
                         and os.path.exists("/etc/nv_tegra_release"))
    caps['has_gstreamer'] = "GStreamer" in cv2.getBuildInformation()
    try:
        caps['has_cuda_cv'] = (hasattr(cv2, 'cuda')
                               and cv2.cuda.getCudaEnabledDeviceCount() > 0)
    except Exception:
        caps['has_cuda_cv'] = False
    return caps


HW_CAPS = _detect_capabilities()

API_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava-phi3" 
DEFAULT_PROMPT = "Describe what you see in this image in one sentence. Be specific about objects, people, and actions. Only describe what is visible right now."
CHECK_INTERVAL = 10 
MOTION_THRESHOLD = 5000 
MEMORY_MAX_PER_ZONE = 100 
MOTION_DOWNSCALE = 2 
JPEG_QUALITY = 85 
FRAME_BUFFER_SIZE = 3 


def _clean_vlm_response(text: str) -> str:
    """Strip markdown fences, prompt echoes, and garbage from VLM output."""
    if not text:
        return ""
    import re as _re
    cleaned = _re.sub(r'```\w*\n?', '', text)
    cleaned = cleaned.replace('```', '')
    for marker in ["Output only", "no preamble", "Note anything new",
                   "In one sentence, describe"]:
        idx = cleaned.find(marker)
        if idx > 0:
            cleaned = cleaned[:idx]
    cleaned = cleaned.strip()
    paragraphs = [p.strip() for p in cleaned.split('\n') if p.strip()]
    if len(paragraphs) > 1:
        cleaned = max(paragraphs, key=len)
    return cleaned.strip()


def _frame_sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


_exit_count = 0

def setup_signals(stop_event):
    def handle_exit(sig, frame):
        global _exit_count
        _exit_count += 1
        if _exit_count >= 2:
            print("\n[SHUTDOWN] Second signal — forcing immediate exit.", file=sys.stderr)
            stop_speaking()
            os._exit(1)
        print("\n[SHUTDOWN] Exit signal received. Cleaning up threads... (press Ctrl-C again to force-quit)")
        stop_event.set()
        stop_speaking()
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)


def _find_usb_camera() -> int:
    import glob
    candidates = sorted(glob.glob("/dev/video*"))
    for dev in candidates:
        try:
            idx = int(dev.replace("/dev/video", ""))
        except ValueError:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"  Auto-detected USB camera: /dev/video{idx}")
                return idx
    print("  WARNING: USB camera auto-detect failed, defaulting to index 0", file=sys.stderr)
    return 0


def _build_gstreamer_pipeline(source, width: int, height: int) -> str | None:
    src = str(source).lower()
    
    # CSI camera via nvarguscamerasrc
    if src.startswith("csi"):
        sensor_id = src.replace("csi", "")
        sensor_id = int(sensor_id) if sensor_id.isdigit() else 0
        # 1280x720@30fps — matches IMX219 native mode; lower fps reduces NVMM buffer usage
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} "
            f"! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1,format=NV12 "
            f"! nvvidconv flip-method=0 "
            f"! video/x-raw,width={width},height={height},format=BGRx "
            f"! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0"
        )
        
    if src.startswith("rtsp://"):
        return (
            f"rtspsrc location={src} latency=200 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0"
        )
    if src.isdigit():
        return (
            f"v4l2src device=/dev/video{src} io-mode=2 ! "
            f"image/jpeg, width={width}, height={height}, framerate=30/1 ! "
            "jpegparse ! nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
    return None


VOICE_COMMANDS = {
    # With wake-word prefixes (cam/gam/ham — Vosk misrecognitions)
    "cam what do you see": "analyze_now",
    "gam what do you see": "analyze_now",
    "ham what do you see": "analyze_now",
    "cam look again": "analyze_now",
    "gam look again": "analyze_now",
    "ham look again": "analyze_now",
    "cam repeat": "speak_last",
    "gam repeat": "speak_last",
    "ham repeat": "speak_last",
    "cam record on": "record_on",
    "ham record on": "record_on",
    "cam record off": "record_off",
    "gam record off": "record_off",
    "ham record off": "record_off",
    "cam are you recording": "recording_status",
    "ham are you recording": "recording_status",
    "gam are you recording": "recording_status",
    "cam remember this": "remember",
    "ham remember this": "remember",
    "gam remember this": "remember",
    "cam what have you seen": "recall",
    "ham what have you seen": "recall",
    "gam what have you seen": "recall",
    "cam status": "status",
    "gam status": "status",
    "ham status": "status",
    "cam go to sleep": "sleep",
    "cam wake up": "wake",
    "gam wake up": "wake",
    "ham wake up": "wake",
    "cam mute": "mute",
    "ham mute": "mute",
    "gam mute": "mute",
    "cam unmute": "unmute",
    "gam unmute": "unmute",
    "ham unmute": "unmute",
    "cam help": "help",
    "gam help": "help",
    "ham help": "help",
    # Without wake-word — Vosk often drops the prefix
    "what do you see": "analyze_now",
    "look again": "analyze_now",
    "repeat": "speak_last",
    "record on": "record_on",
    "record off": "record_off",
    "are you recording": "recording_status",
    "remember this": "remember",
    "what have you seen": "recall",
    "go to sleep": "sleep",
    "wake up": "wake",
}

_tts_processes: list = []  
_tts_busy = threading.Event()  
_tts_lock = threading.Lock()

def stop_speaking():
    with _tts_lock:
        for proc in _tts_processes:
            if proc and proc.poll() is None:
                proc.terminate()
        _tts_processes.clear()
        _tts_busy.clear()


analysis_lock = threading.Lock()
analysis_in_progress = False


class MemoryStore:
    """Persistent scene memory backed by JSON.
    Supports rotation: max records per zone, auto-archives old data.
    """
    MAX_TOTAL_RECORDS = 500  # across all zones; triggers rotation

    def __init__(self, path: str, max_per_zone: int = MEMORY_MAX_PER_ZONE):
        self.path = path
        self.max_per_zone = max_per_zone
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                size = os.path.getsize(self.path)
                if size == 0:
                    print("Memory file is empty, starting fresh.", file=sys.stderr)
                    return {}
                with open(self.path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
                print(f"Memory file has unexpected format, starting fresh.", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"Memory load error: {e} — starting fresh.", file=sys.stderr)
            except Exception as e:
                print(f"Memory load error: {e}", file=sys.stderr)
        return {}

    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"Memory save error: {e}", file=sys.stderr)

    def _total_records(self) -> int:
        return sum(len(v) for v in self._data.values() if isinstance(v, list))

    def _rotate(self):
        """Archive old records when total exceeds MAX_TOTAL_RECORDS."""
        total = self._total_records()
        if total <= self.MAX_TOTAL_RECORDS:
            return
        # Archive to timestamped file
        archive_path = self.path.replace('.json', f'_archive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        try:
            with open(archive_path, 'w') as f:
                json.dump(self._data, f, indent=2)
            LOG.info(f"Memory rotated: {total} records archived to {archive_path}")
        except Exception as e:
            LOG.error(f"Memory rotation failed: {e}")
            return
        # Keep only the most recent half per zone
        for zone in self._data:
            if isinstance(self._data[zone], list) and len(self._data[zone]) > 0:
                keep = max(len(self._data[zone]) // 2, 10)
                self._data[zone] = self._data[zone][-keep:]
        self._save()

    def add(self, description: str, zone: str = "default"):
        with self._lock:
            if zone not in self._data:
                self._data[zone] = []
            self._data[zone].append({
                "ts": datetime.now().isoformat(timespec='seconds'),
                "desc": description
            })
            if len(self._data[zone]) > self.max_per_zone:
                self._data[zone] = self._data[zone][-self.max_per_zone:]
            self._rotate()
            self._save()

    def get_context(self, zone: str = "default", n: int = 3) -> str:
        with self._lock:
            entries = self._data.get(zone, [])
            if not entries:
                return ""
            recent = entries[-n:][::-1] 
            lines = [f"- [{e['ts'][11:16]}] {e['desc']}" for e in recent]
            return "Recent observations:\n" + "\n".join(lines)


def _setup_shared_alsa(card: int = 0) -> tuple[str, str]:
    """Configure ALSA dsnoop for shared capture on a USB audio device.
    
    The CA Essential SP speakerphone is a single USB device for mic + speaker.
    dsnoop allows the mic to stay open while aplay uses plughw for playback.
    dmix is avoided because it's fragile on USB audio (ring buffer init failures).
    
    Returns (capture_device, playback_device) ALSA device strings.
    """
    asoundrc_path = os.path.expanduser("~/.asoundrc")
    
    config = f"""# Auto-generated by SmartCam for shared USB audio capture
# dsnoop allows mic to stay open while aplay uses plughw for playback

pcm.usb_mic {{
    type dsnoop
    ipc_key 65536
    ipc_perm 0666
    slave {{
        pcm "hw:{card},0"
        channels 1
        format S16_LE
    }}
}}

pcm.mic {{
    type plug
    slave.pcm "usb_mic"
}}
"""
    playback_dev = f"plughw:{card},0"
    try:
        needs_write = True
        if os.path.exists(asoundrc_path):
            with open(asoundrc_path, 'r') as f:
                existing = f.read()
            if 'Auto-generated by SmartCam' in existing:
                if f'"hw:{card},0"' in existing and 'rate 48000' not in existing:
                    needs_write = False
        
        if needs_write:
            with open(asoundrc_path, 'w') as f:
                f.write(config)
            print(f"  ALSA config written to {asoundrc_path} (card {card}: dsnoop for capture)")
        else:
            print(f"  ALSA config OK ({asoundrc_path}, card {card})")
        
        return "mic", playback_dev
    except Exception as e:
        print(f"  ALSA config failed: {e} — falling back to plughw", file=sys.stderr)
        return f"plughw:{card},0", playback_dev


def _find_usb_audio_card() -> int:
    """Find the USB audio card number. Returns card index or 0 as default."""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if 'usb' in line.lower() or 'ca essential' in line.lower():
                m = re.search(r'card (\d+)', line, re.IGNORECASE)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return 0


def _find_usb_alsa_playback_card() -> str:
    """Return playback device — always uses plughw for reliability.
    dmix is avoided because it's fragile on USB audio chipsets.
    The voice listener uses dsnoop for capture, which doesn't block plughw playback.
    """
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if 'usb' in line.lower() or 'ca essential' in line.lower():
                m = re.search(r'card (\d+).*device (\d+)', line, re.IGNORECASE)
                if m:
                    card, dev = m.group(1), m.group(2)
                    return f"plughw:{card},{dev}"
    except Exception:
        pass
    return 'default'


_USB_ALSA_PLAYBACK = None  
_PIPER_BIN = None          


def _find_piper_binary(piper_model_path: str | None = None) -> str:
    import shutil
    candidates = []
    if piper_model_path:
        model_dir = os.path.dirname(os.path.abspath(piper_model_path))
        candidates.append(os.path.join(model_dir, 'piper', 'piper'))
        candidates.append(os.path.join(model_dir, 'piper'))
        parent = os.path.dirname(model_dir)
        candidates.append(os.path.join(parent, 'piper', 'piper'))

    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        candidates.append(os.path.join(venv, 'bin', 'piper'))

    candidates.extend([
        os.path.expanduser('~/.local/bin/piper'),
        '/usr/local/bin/piper',
        '/usr/bin/piper',
    ])

    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    found = shutil.which('piper')
    if found:
        return found
    return 'piper'  


def speak(text: str, piper_model: str | None = None):
    global _USB_ALSA_PLAYBACK, _PIPER_BIN
    stop_speaking()
    with _tts_lock:
        if _USB_ALSA_PLAYBACK is None:
            _USB_ALSA_PLAYBACK = _find_usb_alsa_playback_card()
            print(f"  TTS playback device: {_USB_ALSA_PLAYBACK}")
        if _PIPER_BIN is None and piper_model:
            _PIPER_BIN = _find_piper_binary(piper_model)
            print(f"  TTS piper binary: {_PIPER_BIN}")
        try:
            if piper_model:
                _tts_busy.set()
                piper_env = os.environ.copy()
                piper_dir = os.path.dirname(os.path.abspath(_PIPER_BIN))
                ld = piper_env.get('LD_LIBRARY_PATH', '')
                piper_env['LD_LIBRARY_PATH'] = f"{piper_dir}:{ld}" if ld else piper_dir

                piper = subprocess.Popen(
                    [_PIPER_BIN, '--model', piper_model, '--output_raw'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, env=piper_env
                )
                aplay = subprocess.Popen(
                    ['aplay', '-D', _USB_ALSA_PLAYBACK,
                     '-t', 'raw', '-r', '22050', '-f', 'S16_LE', '-c', '1'],
                    stdin=piper.stdout, stderr=subprocess.PIPE
                )
                piper.stdin.write(text.encode('utf-8'))
                piper.stdin.close()
                _tts_processes.extend([piper, aplay])

                def _monitor(pp, ap):
                    pp.wait()
                    ap.wait()
                    if pp.returncode != 0:
                        err = pp.stderr.read().decode(errors='replace').strip() if pp.stderr else ''
                        print(f"TTS piper error (rc={pp.returncode}): {err}", file=sys.stderr)
                    if ap.returncode != 0:
                        err = ap.stderr.read().decode(errors='replace').strip() if ap.stderr else ''
                        print(f"TTS aplay error (rc={ap.returncode}): {err}", file=sys.stderr)
                    _tts_busy.clear()
                threading.Thread(target=_monitor, args=(piper, aplay), daemon=True).start()
            elif sys.platform == "win32":
                _tts_busy.set()
                def _pyttsx3_speak(t: str):
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.say(t)
                        engine.runAndWait()
                        engine.stop()
                    except Exception as e:
                        print(f"TTS error: {e}", file=sys.stderr)
                    finally:
                        _tts_busy.clear()
                threading.Thread(target=_pyttsx3_speak, args=(text,), daemon=True).start()
            else:
                _tts_busy.set()
                proc = subprocess.Popen(
                    ['espeak-ng', text],
                    stderr=subprocess.DEVNULL
                )
                _tts_processes.append(proc)
                def _monitor(p):
                    p.wait()
                    _tts_busy.clear()
                threading.Thread(target=_monitor, args=(proc,), daemon=True).start()
        except FileNotFoundError as e:
            _tts_busy.clear()
            print(f"TTS binary not found ({e}). Response: {text}", file=sys.stderr)
        except Exception as e:
            _tts_busy.clear()
            print(f"TTS error: {e}", file=sys.stderr)


def get_ai_analysis(image_data: str, ollama_url: str, model: str, prompt: str,
                    timeout: int, num_gpu: int | None = None) -> str | None:
    # Optimized options for Jetson Orin Nano
    options = {
        "num_ctx": 1024,      # Keep context window tight
        "num_predict": 75,    # Hard cap on output tokens
        "temperature": 0.1,   # Make answers deterministic
        "num_thread": 4       # Match 6-core ARM CPU, leaving 2 cores for the camera pipeline
    }
    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],
        "stream": False,
        "keep_alive": -1,
        "options": options
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        if response.status_code == 500:
            body = response.text[:300]
            print(f"[Ollama] HTTP 500 — model may have crashed. Body: {body}", file=sys.stderr)
            return "__ERROR_500__"
        response.raise_for_status()
        return response.json().get('response')
    except requests.exceptions.ConnectionError:
        print("[Ollama] Connection refused — is ollama serve running?", file=sys.stderr)
        return "__ERROR_CONN__"
    except requests.exceptions.Timeout:
        print(f"[Ollama] Request timed out after {timeout}s", file=sys.stderr)
        return "__ERROR_TIMEOUT__"
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] API error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[Ollama] Unexpected error: {e}", file=sys.stderr)
        return None


def _restart_ollama():
    """Auto-restart Ollama service after repeated failures."""
    LOG.warning("Auto-restarting Ollama service after repeated 500 errors...")
    print(f"[{time.strftime('%H:%M:%S')}] Auto-restarting Ollama...", file=sys.stderr)
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', 'ollama'],
                       capture_output=True, timeout=30)
        time.sleep(10)  # give Ollama time to start
        LOG.info("Ollama restart completed")
    except Exception as e:
        LOG.error(f"Ollama restart failed: {e}")


def _execute_analysis(image_data: str, ollama_url: str, model: str, prompt: str,
                      shared_state: dict, timeout: int, piper_model: str | None,
                      memory: MemoryStore | None, zone: str,
                      speak_once_event: threading.Event | None = None,
                      num_gpu: int | None = None):
    global analysis_in_progress

    try:
        result = get_ai_analysis(image_data, ollama_url, model, prompt, timeout, num_gpu=num_gpu)
        current_time = time.strftime('%H:%M:%S')

        is_api_error = isinstance(result, str) and result.startswith("__ERROR_")

        if is_api_error or result is None:
            error_label = result if is_api_error else "__ERROR_NONE__"
            if error_label in ("__ERROR_500__", "__ERROR_CONN__"):
                backoff = 30
                HEALTH.record_error("500" if error_label == "__ERROR_500__" else "conn")
                # Auto-restart Ollama after 3 consecutive 500s
                if HEALTH.get_consecutive_500s() >= 3:
                    _restart_ollama()
                    backoff = 45  # extra time after restart
            elif error_label == "__ERROR_TIMEOUT__":
                backoff = 10
                HEALTH.record_error("timeout")
            else:
                backoff = 15
                HEALTH.record_error()
            with shared_state['lock']:
                shared_state['backoff_until'] = time.time() + backoff
                shared_state['latest_status'] = f"API error ({error_label}) — retrying in {backoff}s"
            LOG.warning(f"API error: {error_label}")
            print(f"[{current_time}] {shared_state['latest_status']}", file=sys.stderr)
        else:
            cleaned = _clean_vlm_response(result)
            if not cleaned:
                cleaned = result.strip()

            HEALTH.record_analysis()
            with shared_state['lock']:
                shared_state['latest_status'] = cleaned
                shared_state['backoff_until'] = 0 
                print(f"[{current_time}] {cleaned}")

            if speak_once_event is not None and speak_once_event.is_set():
                speak(cleaned, piper_model)
                speak_once_event.clear()
            if memory is not None:
                memory.add(cleaned, zone)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] _execute_analysis crashed: {e}", file=sys.stderr)
    finally:
        with analysis_lock:
            analysis_in_progress = False


def ai_worker(ollama_url: str, model: str, interval: int, motion_threshold: int,
              shared_state: dict, stop_event: threading.Event, timeout: int,
              piper_model: str | None, memory: MemoryStore | None, zone: str,
              base_prompt: str, memory_context_n: int,
              force_analyze_event: threading.Event | None = None,
              speak_once_event: threading.Event | None = None,
              sleep_event: threading.Event | None = None,
              num_gpu: int | None = None):
    global analysis_in_progress
    print("AI worker started.")
    previous_frame_gray = None
    last_ai_analysis_time = time.time()

    while not stop_event.is_set():
        if sleep_event is not None and sleep_event.is_set():
            stop_event.wait(0.25)
            continue

        with shared_state['lock']:
            backoff_until = shared_state.get('backoff_until', 0)
        if backoff_until and time.time() < backoff_until:
            remaining = int(backoff_until - time.time())
            if remaining % 10 == 0:  
                print(f"[{time.strftime('%H:%M:%S')}] API backoff — retrying in {remaining}s")
            stop_event.wait(1.0)
            continue

        frame_to_process = None
        cur_threshold = motion_threshold
        cur_interval = interval
        with shared_state['lock']:
            if shared_state['latest_frame'] is not None:
                frame_to_process = shared_state['latest_frame'].copy()
            cur_threshold = shared_state.get('motion_threshold', motion_threshold)
            cur_interval = shared_state.get('interval', interval)

        if frame_to_process is not None:
            mh, mw = frame_to_process.shape[:2]
            motion_small = cv2.resize(frame_to_process,
                                      (mw // MOTION_DOWNSCALE, mh // MOTION_DOWNSCALE),
                                      interpolation=cv2.INTER_NEAREST)
            current_frame_gray = cv2.cvtColor(motion_small, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (9, 9), 0)

            trigger_analysis = False
            is_motion_trigger = False
            motion_area = 0
            scaled_threshold = cur_threshold // (MOTION_DOWNSCALE * MOTION_DOWNSCALE)

            if previous_frame_gray is None:
                trigger_analysis = True
                print(f"[{time.strftime('%H:%M:%S')}] First frame, triggering analysis...")
            else:
                frame_delta = cv2.absdiff(previous_frame_gray, current_frame_gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
                motion_area = cv2.countNonZero(thresh)

                if motion_area > scaled_threshold:
                    trigger_analysis = True
                    is_motion_trigger = True
                elif time.time() - last_ai_analysis_time >= cur_interval:
                    trigger_analysis = True

            is_forced = False
            if not trigger_analysis and force_analyze_event is not None and force_analyze_event.is_set():
                trigger_analysis = True
                is_forced = True
                force_analyze_event.clear()

            with analysis_lock:
                can_start_analysis = not analysis_in_progress

            if trigger_analysis and can_start_analysis:
                with analysis_lock:
                    analysis_in_progress = True
                last_ai_analysis_time = time.time()

                if is_forced:
                    print(f"[{time.strftime('%H:%M:%S')}] Voice command: analysing NOW...")
                elif is_motion_trigger:
                    print(f"[{time.strftime('%H:%M:%S')}] Motion detected (area: {motion_area:.0f}), analysing...")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Periodic check...")

                # Memory context — framed to prevent model from chaining old observations
                prompt = base_prompt
                if memory is not None:
                    ctx = memory.get_context(zone, n=memory_context_n)
                    if ctx:
                        prompt = (
                            f"[Background - DO NOT repeat this]\n{ctx}\n\n"
                            f"[Instruction]\n{base_prompt}"
                        )

                # Voice commands use latest frame; motion/periodic pick sharpest from buffer
                if is_forced:
                    vlm_frame = frame_to_process
                else:
                    with shared_state['lock']:
                        buffered = [f.copy() for f in shared_state['frame_buffer']]
                    if len(buffered) > 1:
                        vlm_frame = max(buffered, key=_frame_sharpness)
                    else:
                        vlm_frame = frame_to_process

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                _, buffer = cv2.imencode('.jpg', vlm_frame, encode_param)
                img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

                analysis_thread = threading.Thread(
                    target=_execute_analysis,
                    args=(img_base64, ollama_url, model, prompt, shared_state, timeout,
                          piper_model, memory, zone, speak_once_event, num_gpu),
                    daemon=True
                )
                analysis_thread.start()

            elif trigger_analysis and not can_start_analysis and not is_motion_trigger:
                last_ai_analysis_time = time.time()

            previous_frame_gray = current_frame_gray

        stop_event.wait(0.25)
    print("AI worker stopped.")


def start_recording(record_dir: str, resolution: tuple) -> cv2.VideoWriter:
    os.makedirs(record_dir, exist_ok=True)
    filename = os.path.join(record_dir, f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, 4.0, resolution)  # 4 FPS matches headless capture rate
    print(f"[{time.strftime('%H:%M:%S')}] Recording started: {filename}")
    return writer


def stop_recording(writer: cv2.VideoWriter | None):
    if writer is not None:
        writer.release()
    print(f"[{time.strftime('%H:%M:%S')}] Recording stopped.")


def voice_listener(command_queue, stop_event, vosk_model_path, ready_event=None):
    try:
        import vosk
    except ImportError as e:
        print(f"Voice listener disabled ({e}). Run: pip install vosk", file=sys.stderr)
        return

    abs_model_path = os.path.abspath(vosk_model_path)
    if not os.path.isdir(abs_model_path):
        print(f"Voice listener disabled: model directory not found: {abs_model_path}", file=sys.stderr)
        return

    # Setup shared ALSA config for concurrent mic + speaker on same USB device
    usb_card = _find_usb_audio_card()
    capture_dev, _ = _setup_shared_alsa(usb_card)
    print(f"  Audio capture device: {capture_dev} (card {usb_card})")

    audio_rate = 16000
    model = vosk.Model(abs_model_path)
    rec = vosk.KaldiRecognizer(model, audio_rate)

    # Try the configured capture device with retries
    arecord_proc = None
    usb_hw_device = None
    # Try dsnoop-based "mic" first, then fallback to plughw, then default
    devices_to_try = [capture_dev]
    if capture_dev != f"plughw:{usb_card},0":
        devices_to_try.append(f"plughw:{usb_card},0")
    devices_to_try.append("default")

    for device in devices_to_try:
        for attempt in range(3):
            # Wait for any active TTS before each attempt
            tts_wait = time.time()
            while _tts_busy.is_set() and time.time() - tts_wait < 5:
                time.sleep(0.3)
            try:
                arecord_proc = subprocess.Popen(
                    ['arecord', '-D', device, '-f', 'S16_LE', '-c', '1',
                     '-r', str(audio_rate), '-t', 'raw', '--buffer-size', '8000'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                test = arecord_proc.stdout.read(4000)
                if len(test) > 0:
                    usb_hw_device = device
                    print(f"  Audio mode: arecord ({device}) @ {audio_rate} Hz")
                    break
                err = arecord_proc.stderr.read(500).decode(errors='replace')
                arecord_proc.kill()
                arecord_proc.wait()
                arecord_proc = None
                if 'busy' in err.lower() and attempt < 2:
                    print(f"  {device} busy, retrying ({attempt+1}/3)...", file=sys.stderr)
                    time.sleep(2.0)
                    continue
                print(f"  {device} failed: {err.strip()[:80]}", file=sys.stderr)
                break
            except Exception as e:
                if arecord_proc:
                    try: arecord_proc.kill(); arecord_proc.wait()
                    except: pass
                    arecord_proc = None
                print(f"  {device} error: {e}", file=sys.stderr)
                break
        if usb_hw_device:
            break

    if not usb_hw_device or arecord_proc is None:
        print("Voice listener init failed: no working audio input device found.", file=sys.stderr)
        print("  Check: is the USB mic plugged in? Run 'arecord -l'", file=sys.stderr)
        return

    print("Voice listener started.")
    if ready_event is not None:
        ready_event.set()

    CHUNK = 4000
    tts_was_active = False
    while not stop_event.is_set():
        if _tts_busy.is_set():
            try: arecord_proc.stdout.read(CHUNK)
            except Exception: pass
            tts_was_active = True
            stop_event.wait(0.05)
            continue
        if tts_was_active:
            tts_was_active = False
            time.sleep(0.3)
            try: arecord_proc.stdout.read(CHUNK)
            except Exception: pass
            rec.FinalResult()
            continue
        try:
            data = arecord_proc.stdout.read(CHUNK)
            if len(data) == 0:
                if arecord_proc.poll() is not None:
                    print("Voice listener: arecord died.", file=sys.stderr)
                    break
                continue
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower().strip()
                if text:
                    print(f"[{time.strftime('%H:%M:%S')}] Heard: \"{text}\"")
                    for phrase, action in sorted(VOICE_COMMANDS.items(), key=lambda x: -len(x[0])):
                        if phrase in text:
                            command_queue.put(action)
                            print(f"[{time.strftime('%H:%M:%S')}] Voice command: {action}")
                            break
            else:
                # FIX 3: Catch commands in continuous noise streams where Vosk fails to detect a silence gap
                partial_result = json.loads(rec.PartialResult())
                text = partial_result.get("partial", "").lower().strip()
                if text:
                    for phrase, action in sorted(VOICE_COMMANDS.items(), key=lambda x: -len(x[0])):
                        if phrase in text:
                            command_queue.put(action)
                            print(f"[{time.strftime('%H:%M:%S')}] Voice command (partial): {action}")
                            rec.Reset() # Reset the recognizer to prevent duplicate triggers
                            break

        except Exception as e:
            print(f"Voice listener error: {e}", file=sys.stderr)

    if arecord_proc and arecord_proc.poll() is None:
        try: arecord_proc.terminate(); arecord_proc.wait(timeout=2)
        except Exception:
            try: arecord_proc.kill()
            except Exception: pass
    print("Voice listener stopped.")


def main(video_source: str | int, ollama_url: str, model: str, initial_prompt: str,
         interval: int, motion_threshold: int, headless: bool, enhance_video: bool,
         timeout: int, denoise: bool, piper_model: str | None,
         memory_path: str | None, zone: str, memory_context_n: int,
         record_dir: str, vosk_model: str | None, num_gpu: int | None = None):
    print("Starting SmartCam...")
    print(f" - Source: {video_source}")
    print(f" - Model: {model}")
    print(f" - Interval: {interval}s | Motion threshold: {motion_threshold}px")
    print(f" - Headless: {headless} | Denoise: {denoise} | Enhance: {enhance_video}")
    print(f" - TTS: {'piper (' + piper_model + ')' if piper_model else 'espeak-ng'}")
    print(f" - Memory: {memory_path or 'disabled'} | Zone: {zone} | Context: last {memory_context_n}")
    print(f" - Record dir: {record_dir} | Toggle: 'r' key or voice command")
    print(f" - Voice: {'vosk (' + vosk_model + ')' if vosk_model else 'disabled (--vosk-model)'}")
    print(f" - HW accel: Jetson={'yes' if HW_CAPS['is_jetson'] else 'no'}"
          f" | GStreamer={'yes' if HW_CAPS['has_gstreamer'] else 'no'}"
          f" | CUDA-CV={'yes' if HW_CAPS['has_cuda_cv'] else 'no'}"
          f" | GPU layers: {num_gpu if num_gpu is not None else 'all (default)'}")
    if vosk_model and not os.path.isdir(vosk_model):
        print(f" *** WARNING: vosk model path '{os.path.abspath(vosk_model)}' does not exist!")
    print("-" * 30)

    memory = MemoryStore(memory_path) if memory_path else None
    command_queue: queue.Queue = queue.Queue()
    force_analyze_event = threading.Event()
    speak_once_event = threading.Event()
    sleep_event = threading.Event()
    mute_flag = threading.Event()

    resolution = (640, 480)  # VGA: good for VLM without excessive memory on 8GB Jetson

    hw_resize = False
    is_csi = str(video_source).startswith('csi')
    if HW_CAPS['has_gstreamer'] and HW_CAPS['is_jetson']:
        gst_pipe = _build_gstreamer_pipeline(video_source, resolution[0], resolution[1])
        if gst_pipe:
            cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                hw_resize = True
                if is_csi:
                    print(" - Capture: CSI (nvarguscamerasrc + VIC resize)")
                else:
                    print(" - Capture: GStreamer (NVDEC decode + VIC resize)")
            else:
                print(" - GStreamer open failed, falling back to default", file=sys.stderr)
                if not is_csi:
                    cap = cv2.VideoCapture(video_source)
                else:
                    print("   CSI camera requires GStreamer. Check: sudo dmesg | grep imx",
                          file=sys.stderr)
                    return
        else:
            cap = cv2.VideoCapture(video_source)
    else:
        if is_csi:
            print("Error: CSI camera requires GStreamer + Jetson.", file=sys.stderr)
            return
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Cannot open source {video_source}", file=sys.stderr)
        return

    clahe = None
    use_cuda_clahe = False
    if enhance_video:
        if HW_CAPS['has_cuda_cv']:
            try:
                clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # type: ignore[attr-defined]
                use_cuda_clahe = True
                print(" - CLAHE: CUDA-accelerated")
            except Exception:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    shared_state = {
        'latest_frame': None,
        'latest_status': "Initializing...",
        'lock': threading.Lock(),
        'motion_threshold': motion_threshold,
        'interval': interval,
        'frame_buffer': deque(maxlen=FRAME_BUFFER_SIZE),
        'backoff_until': 0,   
    }
    stop_event = threading.Event()
    setup_signals(stop_event)

    worker_thread = threading.Thread(
        target=ai_worker,
        args=(ollama_url, model, interval, motion_threshold, shared_state, stop_event,
              timeout, piper_model, memory, zone, initial_prompt, memory_context_n,
              force_analyze_event, speak_once_event, sleep_event, num_gpu),
        daemon=True
    )
    worker_thread.start()

    voice_ready = threading.Event()
    if vosk_model:
        voice_thread = threading.Thread(
            target=voice_listener,
            args=(command_queue, stop_event, vosk_model, voice_ready),
            daemon=True
        )
        voice_thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            if voice_ready.is_set() or not voice_thread.is_alive():
                break
            time.sleep(0.1)
        if not voice_ready.is_set():
            print("WARNING: Voice listener failed to start. Check errors above.", file=sys.stderr)

    if not headless:
        cv2.namedWindow("SmartCam (q to quit)", cv2.WINDOW_NORMAL)

    video_writer: cv2.VideoWriter | None = None
    recording = False
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10
    reconnect_attempts = 0
    MAX_RECONNECTS = 5

    while not stop_event.is_set():
        try:
            if not cap.grab():
                reconnect_attempts += 1
                print(f"Stream lost, reconnecting ({reconnect_attempts}/{MAX_RECONNECTS})...",
                      file=sys.stderr)
                cap.release()
                if reconnect_attempts >= MAX_RECONNECTS:
                    print("FATAL: Camera not recoverable. Shutting down.", file=sys.stderr)
                    stop_event.set()
                    break
                time.sleep(3)
                # Rebuild GStreamer pipeline for CSI/RTSP/USB
                if hw_resize and HW_CAPS['has_gstreamer']:
                    gst_pipe = _build_gstreamer_pipeline(video_source, resolution[0], resolution[1])
                    if gst_pipe:
                        cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
                    else:
                        cap = cv2.VideoCapture(video_source)
                else:
                    cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    continue 
                continue
            reconnect_attempts = 0  
            ret, frame = cap.retrieve()
            
            # NEW: Safety check to prevent cv2 crashes on empty/corrupt frames
            if not ret or frame is None or frame.size == 0:
                continue

            if not hw_resize:
                frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)

            if denoise:
                if HW_CAPS['is_jetson']:
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)
                else:
                    frame = cv2.bilateralFilter(frame, 9, 75, 75)

            if enhance_video and clahe:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                if use_cuda_clahe:
                    gpu_l = cv2.cuda_GpuMat()  # type: ignore[attr-defined]
                    gpu_l.upload(l)
                    gpu_l = clahe.apply(gpu_l, cv2.cuda.Stream.Null())  # type: ignore[attr-defined]
                    l = gpu_l.download()
                else:
                    l = clahe.apply(l)
                frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

            if recording and video_writer is not None:
                video_writer.write(frame)

            with shared_state['lock']:
                shared_state['latest_frame'] = frame.copy()
                shared_state['frame_buffer'].append(frame.copy())
                display_text = shared_state['latest_status']

            if not headless:
                display_frame = frame.copy()
                (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                rect_x2 = min(10 + tw, display_frame.shape[1])
                rect_y1 = max(0, display_frame.shape[0] - th - 20)
                cv2.rectangle(display_frame, (0, rect_y1), (rect_x2, display_frame.shape[0]), (0, 0, 0), cv2.FILLED)
                cv2.putText(display_frame, display_text, (5, display_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if recording:
                    cv2.circle(display_frame, (display_frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                    cv2.putText(display_frame, "REC", (display_frame.shape[1] - 55, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("SmartCam (q to quit)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('r'):
                    if not recording:
                        video_writer = start_recording(record_dir, resolution)
                        recording = True
                    else:
                        stop_recording(video_writer)
                        video_writer = None
                        recording = False
            else:
                # Headless: ~10 FPS — fast enough for responsive voice commands
                if stop_event.wait(0.1):
                    break

            try:
                while True:
                    cmd = command_queue.get_nowait()
                    stop_speaking()  
                    if cmd == "analyze_now":
                        force_analyze_event.set()
                        if not mute_flag.is_set():
                            speak_once_event.set()
                    elif cmd == "speak_last":
                        with shared_state['lock']:
                            last = shared_state['latest_status']
                        speak(last, piper_model)
                    elif cmd == "record_on":
                        if not recording:
                            video_writer = start_recording(record_dir, resolution)
                            recording = True
                            speak("Recording started.", piper_model)
                    elif cmd == "record_off":
                        if recording:
                            stop_recording(video_writer)
                            video_writer = None
                            recording = False
                            speak("Recording stopped.", piper_model)
                    elif cmd == "recording_status":
                        speak("Yes, recording." if recording else "No, not recording.", piper_model)
                    elif cmd == "remember":
                        force_analyze_event.set()
                        speak("Noted.", piper_model)
                    elif cmd == "recall":
                        if memory is not None:
                            ctx = memory.get_context(zone, n=3)
                            if ctx:
                                lines = [l.strip() for l in ctx.split('\n') if l.startswith('- [')]
                                readable = '. '.join(l.replace('- [', 'At ').replace(']', ',') for l in lines)
                                speak(readable, piper_model)
                            else:
                                speak("Nothing in memory yet.", piper_model)
                        else:
                            speak("Memory is disabled.", piper_model)
                    elif cmd == "status":
                        with shared_state['lock']:
                            last = shared_state['latest_status']
                        parts = []
                        parts.append("recording" if recording else "not recording")
                        parts.append("sleeping" if sleep_event.is_set() else "awake")
                        parts.append("muted" if mute_flag.is_set() else "unmuted")
                        speak(f"I am {', '.join(parts)}. Last: {last}", piper_model)
                    elif cmd == "sleep":
                        sleep_event.set()
                        speak("Going to sleep.", piper_model)
                    elif cmd == "wake":
                        sleep_event.clear()
                        speak("I'm awake.", piper_model)
                    elif cmd == "mute":
                        mute_flag.set()
                        speak("Muted.", piper_model)
                    elif cmd == "unmute":
                        mute_flag.clear()
                        speak("Unmuted.", piper_model)
                    elif cmd == "help":
                        speak("Commands: what do you see, look again, repeat, status, "
                              "remember this, what have you seen, record on, record off, "
                              "are you recording, go to sleep, wake up, mute, unmute, "
                              "more sensitive, less sensitive, check more often, check less often.",
                              piper_model)
            except queue.Empty:
                pass

        except KeyboardInterrupt:
            stop_event.set()
            print("\nStopped.")
            break
        except Exception as e:
            consecutive_errors += 1
            print(f"Main loop error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}",
                  file=sys.stderr)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("FATAL: Too many consecutive errors, shutting down.", file=sys.stderr)
                stop_event.set()
                break
            time.sleep(0.5)  
            continue
        consecutive_errors = 0
        HEALTH.heartbeat()  # periodic health log

    stop_event.set()
    LOG.info(f"Shutdown after {HEALTH.analysis_count} analyses, {HEALTH.error_count} errors")
    stop_speaking()

    if recording and video_writer is not None:
        stop_recording(video_writer)
    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    for name, t in [("AI worker", worker_thread)] + ([("Voice listener", voice_thread)] if vosk_model else []):
        t.join(timeout=3)
        if t.is_alive():
            print(f"WARNING: {name} thread did not exit in time.", file=sys.stderr)

    print("[SHUTDOWN] Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartCam - AI camera with voice output and scene memory.")
    parser.add_argument("source", help="Video source: URL or camera index (e.g. 0)")
    parser.add_argument("--ollama-url", default=API_URL, help=f"Ollama API URL (default: {API_URL})")
    parser.add_argument("--model", default=VISION_MODEL, help=f"Vision model (default: {VISION_MODEL})")
    parser.add_argument("--prompt", type=str, help="Custom prompt. Overrides default.")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL,
                        help=f"Seconds between periodic checks (default: {CHECK_INTERVAL})")
    parser.add_argument("--motion-threshold", type=int, default=MOTION_THRESHOLD,
                        help=f"Motion sensitivity in pixels (default: {MOTION_THRESHOLD})")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display — for Jetson production deployment")
    parser.add_argument("--enhance-video", action="store_true",
                        help="CLAHE contrast enhancement for dark scenes")
    parser.add_argument("--denoise", action="store_true",
                        help="Bilateral filter denoising (adds CPU load)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Ollama API call timeout in seconds (default: 300)")
    parser.add_argument("--piper-model", type=str, default=None,
                        help="Path to piper .onnx model file for neural TTS. Falls back to espeak-ng if not set.")
    parser.add_argument("--memory-path", type=str, default=None,
                        help="Path to JSON memory file on SSD (e.g. /mnt/ssd/smartcam_memory.json). Disabled if not set.")
    parser.add_argument("--zone", type=str, default="default",
                        help="Memory zone label for this camera view (default: 'default'). Use named zones for multi-cam.")
    parser.add_argument("--memory-context", type=int, default=3,
                        help="Number of past observations to inject into prompt (default: 3)")
    parser.add_argument("--vosk-model", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "vosk-model-small-en-us-0.15"),
                        help="Path to vosk model dir for voice commands. Download from alphacephei.com/vosk/models")
    parser.add_argument("--record-dir", type=str, default="./recordings",
                        help="Directory to save recordings (default: ./recordings). Toggle with 'r' key or voice.")
    parser.add_argument("--num-gpu", type=int, default=None,
                        help="Number of model layers to put on GPU (rest go to CPU). "
                             "Auto-detected for Jetson if not set. Use 0 for full CPU, "
                             "or a low number (e.g. 15) to avoid CUDA OOM.")

    args = parser.parse_args()

    if str(args.source).lower() == 'auto':
        video_source = _find_usb_camera()
    elif str(args.source).lower().startswith('csi'):
        video_source = args.source.lower()
    else:
        try:
            video_source = int(args.source)
        except ValueError:
            video_source = args.source

    final_prompt = args.prompt if args.prompt else DEFAULT_PROMPT

    num_gpu = args.num_gpu
    is_csi_source = str(args.source).lower().startswith('csi')
    if num_gpu is None and HW_CAPS['is_jetson']:
        try:
            mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            total_gb = mem / (1024 ** 3)
            if total_gb <= 8:
                if args.headless:
                    num_gpu = 12 if is_csi_source else 15
                else:
                    num_gpu = 5
                print(f"[Auto] Jetson {total_gb:.0f}GB ({'headless' if args.headless else 'GUI'}"
                      f"{', CSI' if is_csi_source else ''}) "
                      f"— setting --num-gpu {num_gpu}")
        except Exception:
            pass

    if not ensure_ollama_running(args.ollama_url, args.model):
        print("FATAL: Cannot reach Ollama API. Exiting.", file=sys.stderr)
        sys.exit(1)

    main(video_source, args.ollama_url, args.model, final_prompt, args.interval,
         args.motion_threshold, args.headless, args.enhance_video, args.timeout,
         args.denoise, args.piper_model, args.memory_path, args.zone, args.memory_context,
         args.record_dir, args.vosk_model, num_gpu)
