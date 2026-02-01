import requests
import base64
import time
import numpy as np
import argparse
import cv2
import threading
import sys

# --- Default Configuration ---
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2-vision"
DEFAULT_PROMPT = "You are a hamster behavior expert. Look at this image. Is the hamster: 1. On the wheel? 2. Eating? 3. Sleeping? 4. Climbing? Provide a 1-sentence summary."
DEFAULT_INTERVAL = 15

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
            # 1. Encode the captured frame for the API
            _, buffer = cv2.imencode('.jpg', frame_to_process)
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
            # 1. Grab frame data from phone
            img_resp = requests.get(stream_url, timeout=10)
            img_resp.raise_for_status()

            # Decode image for display
            img_arr = np.frombuffer(img_resp.content, np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Could not decode image from stream. Is the URL correct and the stream active?", file=sys.stderr)
                time.sleep(5)
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

        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from stream: {e}", file=sys.stderr)
            print("Retrying in 15 seconds...", file=sys.stderr)
            time.sleep(15)
        except KeyboardInterrupt:
            stop_event.set()
            print("\nExiting Hamster Cam.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}", file=sys.stderr)
            time.sleep(interval)

    # Clean up OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An AI-powered camera to monitor your pet.")
    parser.add_argument("stream_url", help="The URL of the IP camera stream (e.g., http://192.168.1.109:8080/video).")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"The Ollama API URL (default: {DEFAULT_OLLAMA_URL}).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"The vision model to use (default: {DEFAULT_MODEL}).")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="The prompt for the AI model.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help=f"The interval in seconds between checks (default: {DEFAULT_INTERVAL}).")

    args = parser.parse_args()

    main(args.stream_url, args.ollama_url, args.model, args.prompt, args.interval)