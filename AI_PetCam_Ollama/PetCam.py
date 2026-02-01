import requests
import base64
import time
import numpy as np
import argparse
import cv2
import threading
import sys

# Configuration
API_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llama3.2-vision"
MY_PROMPT = """You are a meticulous AI visual analyst. Your task is to observe a hamster and report its activity with high accuracy.

**Step 1: Analyze the Scene (Internal Monologue).**
First, describe what you see in the image. Is the hamster present? Where is it located in the cage? What object is it nearest to? The cage contains a white playing bowl in which the hamster sits when it wants to be picked up by its owner, a red sand bath, a blue water bottle and feeder setup, a seesaw, a tunnel, somne small toys and a white mouse-shaped house.

**Step 2: Determine the Activity (Internal Monologue).**
Based on the hamster's posture and location from Step 1, determine its most likely activity (e.g., eating, drinking, sleeping, running, climbing, bathing, exploring, or wanting to play and interact with owner).

**Step 3: Formulate the Final Summary.**
Synthesize your analysis from the previous steps into a single, concise summary sentence.

**Output Rules:**
- Your final output MUST be only the single summary sentence from Step 3.
- If the hamster is interacting with a known object, mention it.
- If the hamster is not clearly visible or its activity cannot be determined, your summary must be: "The hamster's activity is currently unclear."
- Do not guess or invent details. Stick strictly to visual evidence.

Begin your analysis."""
DEFAULT_INTERVAL = 10  # seconds between checks

def get_hamster_status(image_data: str, ollama_url: str, model: str, prompt: str) -> str | None:
    """
    Sends an image to the Ollama API for analysis and returns the response.
    Args:
        image_data: The base64 encoded image string.
        ollama_url: The URL of the Ollama generate endpoint.
        model: The name of the vision model to use.
        prompt: The prompt to send to the model.
    Returns:
        The text response from the model, or None if an error occurs.
    """
    try:
        response = requests.post(ollama_url, json={
            "model": model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False
        }, timeout=30)  # Added a timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get('response')
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}", file=sys.stderr)
        return None

def ai_worker(ollama_url: str, model: str, prompt: str, interval: int, shared_state: dict, stop_event: threading.Event):
    """
    A worker thread that periodically takes the latest frame and sends it for AI analysis.
    This runs in the background to avoid blocking the main UI thread.
    """
    print("AI worker thread started.")
    while not stop_event.is_set():
        frame_to_process = None
        # Safely get the latest frame from the shared state
        with shared_state['lock']:
            if shared_state['latest_frame'] is not None:
                # Make a copy to process outside the lock, so the UI thread isn't blocked
                frame_to_process = shared_state['latest_frame'].copy()

        if frame_to_process is not None:
            # Add a message to show that analysis is starting
            print(f"[{time.strftime('%H:%M:%S')}] Sending frame to AI for analysis...")

            # --- Performance Optimization: Resize & Compress Image ---
            # Vision models don't need high-res images. Resizing reduces data size and speeds up analysis.
            max_dim = 1024
            h, w, _ = frame_to_process.shape
            if h > max_dim or w > max_dim:
                if h > w:
                    new_h, new_w = max_dim, int(w * (max_dim / h))
                else:
                    new_h, new_w = int(h * (max_dim / w)), max_dim
                resized_frame = cv2.resize(frame_to_process, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized_frame = frame_to_process

            # 1. Encode the resized frame with lower quality to further reduce size. 85 is a good balance.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # 2. Get AI analysis (this is a blocking call, but it's in a separate thread)
            status = get_hamster_status(img_base64, ollama_url, model, prompt)
            current_time = time.strftime('%H:%M:%S')

            # 3. Safely update the shared status
            with shared_state['lock']:
                if status:
                    shared_state['latest_status'] = f"AI: {status.strip()}"
                    print(f"[{current_time}] Hamster AI says: {status.strip()}")
                else:
                    shared_state['latest_status'] = "AI: Analysis failed."
                    print(f"[{current_time}] Failed to get analysis.")

        # Wait for the next interval, but check for the stop event periodically to exit quickly
        stop_event.wait(interval)
    print("AI worker thread stopped.")

def main(stream_url: str, ollama_url: str, model: str, prompt: str, interval: int):
    """
    Main loop to capture frames, get analysis, and print status.
    """
    print("Starting Hamster Cam...")
    print(f" - Video Stream: {stream_url}")
    print(f" - Ollama URL: {ollama_url}")
    print(f" - Model: {model}")
    print(f" - Interval: {interval}s")
    print("-" * 20)

    # Create a window that can be resized
    window_name = "Hamster Cam Feed (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- Video Capture Setup ---
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {stream_url}", file=sys.stderr)
        print("Please check the URL and ensure the camera is streaming.", file=sys.stderr)
        return

    # --- Threading Setup ---
    shared_state = {
        'latest_frame': None,
        'latest_status': "Initializing...",
        'lock': threading.Lock()  # A lock to prevent race conditions between threads
    }
    stop_event = threading.Event()

    # Create and start the AI worker thread.
    # It's a 'daemon' so it automatically exits when the main program does.
    worker_thread = threading.Thread(
        target=ai_worker,
        args=(ollama_url, model, prompt, interval, shared_state, stop_event),
        daemon=True
    )
    worker_thread.start()

    while True:
        try:
            # 1. Grab frame from the video stream using OpenCV
            ret, frame = cap.read()

            if not ret:
                print("Stream ended or failed to grab frame. Attempting to reconnect...", file=sys.stderr)
                # Release the old capture object and try to create a new one
                cap.release()
                time.sleep(2) # Wait a moment before reconnecting
                cap = cv2.VideoCapture(stream_url)
                continue

            # 2. Safely update the latest frame for the worker and get the latest status for display
            with shared_state['lock']:
                shared_state['latest_frame'] = frame
                display_text = shared_state['latest_status']

            # --- Display Logic ---
            # Add a black background rectangle for better text visibility
            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (0, frame.shape[0] - text_height - 20), (10 + text_width, frame.shape[0]), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, display_text, (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 3. Show the frame. This loop runs quickly, keeping the UI responsive.
            cv2.imshow(window_name, frame)

            # Wait for 30ms for a key press. This also allows the window to be refreshed (~33 FPS).
            if cv2.waitKey(30) & 0xFF == ord('q'):
                stop_event.set()  # Signal the worker thread to stop
                break

        except KeyboardInterrupt:
            stop_event.set()
            print("\nExiting Hamster Cam.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}", file=sys.stderr)
            time.sleep(interval)

    # Clean up the capture object
    cap.release()
    # Clean up OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An AI-powered camera to monitor your pet.")
    parser.add_argument("stream_url", help="The URL of the IP camera stream (e.g., http://192.168.1.109:8080/video).")
    parser.add_argument("--ollama-url", default=API_URL, help=f"The Ollama API URL (default: {API_URL}).")
    parser.add_argument("--model", default=VISION_MODEL, help=f"The vision model to use (default: {VISION_MODEL}).")
    parser.add_argument("--prompt", default=MY_PROMPT, help="The prompt for the AI model.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help=f"The interval in seconds between checks (default: {DEFAULT_INTERVAL}).")

    args = parser.parse_args()

    main(args.stream_url, args.ollama_url, args.model, args.prompt, args.interval)