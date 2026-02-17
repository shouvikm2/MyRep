import subprocess
import requests
import base64
import time
import json
import os
import numpy as np
import argparse
import cv2
import queue
import threading
import sys
import platform
from collections import deque
from datetime import datetime
import signal

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

# Configuration
API_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava-jetson"  # ~2.9GB, Phi-3 backbone — best reasoning/memory/context on Jetson Orin 8GB
DEFAULT_PROMPT = """In one sentence, describe exactly what you see. Name specific objects (e.g. 'Samsung earbud box', 'laptop', 'coffee mug') and actions. Note anything new or missing compared to previous observations if provided. Output only the single sentence, no preamble."""
CHECK_INTERVAL = 10  # seconds between periodic checks when no motion
MOTION_THRESHOLD = 5000  # changed pixels to trigger motion-based analysis
MEMORY_MAX_PER_ZONE = 100  # max stored observations per zone before pruning
MOTION_DOWNSCALE = 2  # downscale factor for motion detection (saves CPU on ARM)
JPEG_QUALITY = 70  # lower = smaller payload for VLM, 70 balances detail vs size
FRAME_BUFFER_SIZE = 5  # ring buffer for sharpest-frame selection at analysis time


def _frame_sharpness(frame) -> float:
    """Laplacian variance — higher means sharper. Used to pick best frame for VLM."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- Signal Handling for Forceful Exit ---
def setup_signals(stop_event, voice_stop_func=None):
    def handle_exit(sig, frame):
        print("\n[SHUTDOWN] Exit signal received. Cleaning up threads...")
        stop_event.set()
        if voice_stop_func:
            voice_stop_func() # If you use the speech_recognition background function
        # We don't sys.exit here; we let the main loop break and clean up naturally
    
   
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

def _build_gstreamer_pipeline(source, width: int, height: int) -> str | None:
    src = str(source)
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
    # Analysis
    "camera what do you see": "analyze_now",
    "camera look again": "analyze_now",
    "camera repeat": "speak_last",
    # Recording
    "camera record on": "record_on",
    "camera record off": "record_off",
    "camera are you recording": "recording_status",
    # Memory
    "camera remember this": "remember",
    "camera what have you seen": "recall",
    # Status & control
    "camera status": "status",
    "camera go to sleep": "sleep",
    "camera wake up": "wake",
    "camera mute": "mute",
    "camera unmute": "unmute",
    # Help
    "camera help": "help",
}

# --- TTS state ---
_tts_processes: list = []  # all active TTS subprocesses (piper+aplay or espeak-ng)
_tts_busy = threading.Event()  # set while TTS audio is playing (any platform)
_tts_lock = threading.Lock()


def stop_speaking():
    """Interrupt any in-progress TTS output."""
    with _tts_lock:
        for proc in _tts_processes:
            if proc and proc.poll() is None:
                proc.terminate()
        _tts_processes.clear()
        _tts_busy.clear()


# --- Analysis state ---
analysis_lock = threading.Lock()
analysis_in_progress = False


class MemoryStore:
    """Persistent scene memory backed by JSON on SSD.
    Zone-aware: v1 uses 'default', v1.1 will use GPS-derived zones, v2 room zones.
    """

    def __init__(self, path: str, max_per_zone: int = MEMORY_MAX_PER_ZONE):
        self.path = path
        self.max_per_zone = max_per_zone
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Memory load error: {e}", file=sys.stderr)
        return {}

    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"Memory save error: {e}", file=sys.stderr)

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
            self._save()

    def get_context(self, zone: str = "default", n: int = 3) -> str:
        with self._lock:
            entries = self._data.get(zone, [])
            if not entries:
                return ""
            recent = entries[-n:][::-1]  # most recent first
            lines = [f"- [{e['ts'][11:16]}] {e['desc']}" for e in recent]
            return "Recent observations:\n" + "\n".join(lines)


def speak(text: str, piper_model: str | None = None):
    """Non-blocking TTS output. Interrupts any in-progress speech (barge-in)."""
    stop_speaking()  # kill current speech before starting new
    with _tts_lock:
        try:
            if piper_model:
                _tts_busy.set()
                piper = subprocess.Popen(
                    ['piper', '--model', piper_model, '--output_raw'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                aplay = subprocess.Popen(
                    ['aplay', '-r', '22050', '-f', 'S16_LE', '-c', '1'],
                    stdin=piper.stdout,
                    stderr=subprocess.DEVNULL
                )
                assert piper.stdin is not None
                piper.stdin.write(text.encode('utf-8'))
                piper.stdin.close()
                _tts_processes.extend([piper, aplay])

                def _monitor(proc):
                    proc.wait()
                    _tts_busy.clear()
                threading.Thread(target=_monitor, args=(aplay,), daemon=True).start()
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


def get_ai_analysis(image_data: str, ollama_url: str, model: str, prompt: str, timeout: int) -> str | None:
    """Sends image to Ollama API and returns response text.
    UPDATED: Now includes context limits and keep-alive to prevent Jetson crashes.
    """
    try:
        # CRITICAL FIX 1: Limit context window to 1024 tokens to save VRAM (avoids OOM).
        # CRITICAL FIX 2: Set keep_alive to -1 to prevent model unloading/reloading loop.
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "keep_alive": -1,  
            "options": {
                "num_ctx": 1024
            }
        }
        
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get('response')
    except requests.exceptions.RequestException as e:
        print(f"Ollama API error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected API error: {e}", file=sys.stderr)
        return None


def _execute_analysis(image_data: str, ollama_url: str, model: str, prompt: str,
                      shared_state: dict, timeout: int, piper_model: str | None,
                      memory: MemoryStore | None, zone: str,
                      speak_once_event: threading.Event | None = None):
    """Runs blocking API call in a background thread, stores in memory, speaks only when asked."""
    global analysis_in_progress

    result = get_ai_analysis(image_data, ollama_url, model, prompt, timeout)
    current_time = time.strftime('%H:%M:%S')

    with shared_state['lock']:
        shared_state['latest_status'] = result.strip() if result else "Analysis failed."
        print(f"[{current_time}] {shared_state['latest_status']}")

    if result:
        if speak_once_event is not None and speak_once_event.is_set():
            speak(result.strip(), piper_model)
            speak_once_event.clear()
        if memory is not None:
            memory.add(result.strip(), zone)

    with analysis_lock:
        analysis_in_progress = False


def ai_worker(ollama_url: str, model: str, interval: int, motion_threshold: int,
              shared_state: dict, stop_event: threading.Event, timeout: int,
              piper_model: str | None, memory: MemoryStore | None, zone: str,
              base_prompt: str, memory_context_n: int,
              force_analyze_event: threading.Event | None = None,
              speak_once_event: threading.Event | None = None,
              sleep_event: threading.Event | None = None):
    """Worker thread: motion detection loop that triggers AI analysis."""
    global analysis_in_progress
    print("AI worker started.")
    previous_frame_gray = None
    last_ai_analysis_time = time.time()

    while not stop_event.is_set():
        if sleep_event is not None and sleep_event.is_set():
            stop_event.wait(0.25)
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
            # Downscale for motion detection — saves CPU on ARM cores
            mh, mw = frame_to_process.shape[:2]
            motion_small = cv2.resize(frame_to_process,
                                      (mw // MOTION_DOWNSCALE, mh // MOTION_DOWNSCALE),
                                      interpolation=cv2.INTER_NEAREST)
            current_frame_gray = cv2.cvtColor(motion_small, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (9, 9), 0)

            trigger_analysis = False
            is_motion_trigger = False
            motion_area = 0
            # Scale threshold to match reduced resolution
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

            if not trigger_analysis and force_analyze_event is not None and force_analyze_event.is_set():
                trigger_analysis = True
                force_analyze_event.clear()

            with analysis_lock:
                can_start_analysis = not analysis_in_progress

            if trigger_analysis and can_start_analysis:
                with analysis_lock:
                    analysis_in_progress = True
                last_ai_analysis_time = time.time()

                if is_motion_trigger:
                    print(f"[{time.strftime('%H:%M:%S')}] Motion detected (area: {motion_area:.0f}), analysing...")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Periodic check...")

                # Build memory-augmented prompt
                prompt = base_prompt
                if memory is not None:
                    ctx = memory.get_context(zone, n=memory_context_n)
                    if ctx:
                        prompt = f"{ctx}\n\n{base_prompt}"

                # Pick sharpest frame from buffer for VLM
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
                          piper_model, memory, zone, speak_once_event),
                    daemon=True
                )
                analysis_thread.start()

            elif trigger_analysis and not can_start_analysis and not is_motion_trigger:
                # Periodic timer fired but analysis is busy — reset so we don't re-fire immediately
                last_ai_analysis_time = time.time()

            previous_frame_gray = current_frame_gray

        stop_event.wait(0.25)
    print("AI worker stopped.")


def start_recording(record_dir: str, resolution: tuple) -> cv2.VideoWriter:
    """Opens a new timestamped video file for recording."""
    os.makedirs(record_dir, exist_ok=True)
    filename = os.path.join(record_dir, f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, 20.0, resolution)
    print(f"[{time.strftime('%H:%M:%S')}] Recording started: {filename}")
    return writer


def stop_recording(writer: cv2.VideoWriter | None):
    """Releases the video writer."""
    if writer is not None:
        writer.release()
    print(f"[{time.strftime('%H:%M:%S')}] Recording stopped.")


def voice_listener(command_queue: queue.Queue, stop_event: threading.Event, vosk_model_path: str,
                   ready_event: threading.Event | None = None):
    """Listens to microphone and pushes recognized commands to command_queue.
    Requires: pip install vosk pyaudio
    Model: download from https://alphacephei.com/vosk/models (e.g. vosk-model-small-en-us-0.15)
    """
    try:
        import vosk
        import pyaudio
    except ImportError as e:
        print(f"Voice listener disabled ({e}). Run: pip install vosk pyaudio", file=sys.stderr)
        return

    abs_model_path = os.path.abspath(vosk_model_path)
    if not os.path.isdir(abs_model_path):
        print(f"Voice listener disabled: model directory not found: {abs_model_path}", file=sys.stderr)
        print("  Download a model from https://alphacephei.com/vosk/models and extract it.", file=sys.stderr)
        return

    try:
        model = vosk.Model(abs_model_path)
        rec = vosk.KaldiRecognizer(model, 48000)
        pa = pyaudio.PyAudio()

        try:
            # We explicitly want index 0 (CA Essential SP)
            dev_info = pa.get_device_info_by_index(0)
            print(f"  Audio input device: {dev_info['name']}")
        except IOError:
            print("Voice listener Error: Could not find Device Index 0", file=sys.stderr)
            pa.terminate()
            return

        # UPDATED: Force device_index=0 and rate=48000
        stream = pa.open(format=pyaudio.paInt16, 
                         channels=1, 
                         rate=48000, 
                         input=True, 
                         input_device_index=0, 
                         frames_per_buffer=8000)
        stream.start_stream()
    except OSError as e:
        print(f"Voice listener init failed (audio device): {e}", file=sys.stderr)
        if sys.platform == "win32":
            print("  Hint: Check Windows Settings > Privacy > Microphone.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Voice listener init failed: {e}", file=sys.stderr)
        return

    print("Voice listener started.")
    if ready_event is not None:
        ready_event.set()
    tts_was_active = False
    while not stop_event.is_set():
        # Pause mic during TTS to prevent echo/self-hearing
        if _tts_busy.is_set():
            try:
                stream.read(4000, exception_on_overflow=False)  # drain, discard
            except Exception:
                pass
            tts_was_active = True
            continue

        # Cooldown after TTS ends — flush residual TTS audio from mic buffer
        if tts_was_active:
            tts_was_active = False
            time.sleep(0.3)
            rec.FinalResult()  # flush recognizer state
            continue

        try:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").lower().strip()
                if text:
                    print(f"[{time.strftime('%H:%M:%S')}] Heard: \"{text}\"")
                    for phrase, action in VOICE_COMMANDS.items():
                        if phrase in text:
                            command_queue.put(action)
                            print(f"[{time.strftime('%H:%M:%S')}] Voice command: {action}")
                            break
        except Exception as e:
            print(f"Voice listener error: {e}", file=sys.stderr)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("Voice listener stopped.")


def main(video_source: str | int, ollama_url: str, model: str, initial_prompt: str,
         interval: int, motion_threshold: int, headless: bool, enhance_video: bool,
         timeout: int, denoise: bool, piper_model: str | None,
         memory_path: str | None, zone: str, memory_context_n: int,
         record_dir: str, vosk_model: str | None):
    """Main capture loop."""
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
          f" | CUDA-CV={'yes' if HW_CAPS['has_cuda_cv'] else 'no'}")
    if vosk_model and not os.path.isdir(vosk_model):
        print(f" *** WARNING: vosk model path '{os.path.abspath(vosk_model)}' does not exist!")
    print("-" * 30)

    memory = MemoryStore(memory_path) if memory_path else None
    command_queue: queue.Queue = queue.Queue()
    force_analyze_event = threading.Event()
    speak_once_event = threading.Event()
    sleep_event = threading.Event()
    mute_flag = threading.Event()

    resolution = (480, 360)  # Balanced: detail for object ID + manageable payload

    # --- Capture init: prefer GStreamer + NVDEC on Jetson, fallback to default ---
    hw_resize = False
    if HW_CAPS['has_gstreamer'] and HW_CAPS['is_jetson']:
        gst_pipe = _build_gstreamer_pipeline(video_source, resolution[0], resolution[1])
        if gst_pipe:
            cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                hw_resize = True  # GStreamer nvvidconv handles resize
                print(" - Capture: GStreamer (NVDEC decode + VIC resize)")
            else:
                print(" - GStreamer open failed, falling back to default", file=sys.stderr)
                cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Cannot open source {video_source}", file=sys.stderr)
        return

    # --- CLAHE: use CUDA when available ---
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
    }
    stop_event = threading.Event()

    worker_thread = threading.Thread(
        target=ai_worker,
        args=(ollama_url, model, interval, motion_threshold, shared_state, stop_event,
              timeout, piper_model, memory, zone, initial_prompt, memory_context_n,
              force_analyze_event, speak_once_event, sleep_event),
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
        # Wait for voice listener to finish initializing before entering main loop
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

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Stream lost, reconnecting...", file=sys.stderr)
                cap.release()
                time.sleep(2)
                if hw_resize and HW_CAPS['has_gstreamer']:
                    gst_pipe = _build_gstreamer_pipeline(video_source, resolution[0], resolution[1])
                    cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER) if gst_pipe else cv2.VideoCapture(video_source)
                else:
                    cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    stop_event.set()
                    break
                continue

            # Skip software resize when GStreamer nvvidconv already delivered target resolution
            if not hw_resize:
                frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)

            # Denoise: fast GaussianBlur on Jetson, full bilateralFilter on Windows
            if denoise:
                if HW_CAPS['is_jetson']:
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)
                else:
                    frame = cv2.bilateralFilter(frame, 9, 75, 75)

            # CLAHE: CUDA path on Jetson, CPU path on Windows
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
                shared_state['latest_frame'] = frame
                shared_state['frame_buffer'].append(frame)
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
                time.sleep(0.1)  # 10fps headless — motion worker polls at 4Hz anyway

            # Process voice commands (works in both headless and windowed modes)
            try:
                while True:
                    cmd = command_queue.get_nowait()
                    stop_speaking()  # barge-in: interrupt current speech on any new command
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
            print(f"Main loop error: {e}", file=sys.stderr)

    if recording and video_writer is not None:
        stop_recording(video_writer)
    cap.release()
    if not headless:
        cv2.destroyAllWindows()


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
    # CRITICAL FIX 3: Increased default timeout from 30s to 300s (5 minutes) for Jetson
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

    args = parser.parse_args()

    final_prompt = args.prompt if args.prompt else DEFAULT_PROMPT

    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source

    main(video_source, args.ollama_url, args.model, final_prompt, args.interval,
         args.motion_threshold, args.headless, args.enhance_video, args.timeout,
         args.denoise, args.piper_model, args.memory_path, args.zone, args.memory_context,
         args.record_dir, args.vosk_model)
