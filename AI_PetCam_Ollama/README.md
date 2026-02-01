 AI Hamster Cam

A proof-of-concept (POC) project that uses a mobile phone as an IP camera and a local Ollama vision model to monitor and analyze a hamster's behavior in real-time.

The script displays a live video feed on your computer and overlays a 1-sentence summary of the hamster's current activity, as determined by the AI.

## Features

- **Live Video Feed**: Streams video from any IP camera source (like a phone app).
- **Real-time AI Analysis**: Uses a local Ollama vision model (e.g., `llama3.2-vision`) to analyze the hamster's behavior.
- **Responsive UI**: A multithreaded design ensures the video feed remains smooth and the application window doesn't freeze during AI processing.
- **On-Screen Display**: The AI's analysis is displayed directly on the video feed.
- **Configurable**: Easily change the AI model, prompt, and analysis interval via command-line arguments.

## How It Works

The application is built with a multithreaded architecture to ensure a smooth user experience:

1.  **Main Thread (UI & Capture)**: This thread is responsible for continuously fetching new frames from the IP camera stream and displaying them in an OpenCV window. This keeps the video fluid and the window responsive.
2.  **Worker Thread (AI Analysis)**: This thread runs in the background. At a set interval (e.g., every 15 seconds), it takes the most recent video frame, sends it to the local Ollama API for analysis, and updates the status text.

This separation prevents the time-consuming AI analysis from blocking the video feed.

## Requirements

### Hardware
- A computer to run the script.
- A mobile phone or any IP camera.

### Software
- Python 3.8+
- An IP Camera App for your phone (e.g., "IP Webcam" on Android or "EpocCam" on iOS/Android).
- Ollama installed and running on your computer.
- A downloaded Ollama vision model.

## Setup and Installation

1.  **Set up Ollama:**
    - Install and run Ollama from the official website.
    - Pull the vision model. The script defaults to `llama3.2-vision`.
      > ollama pull llama3.2-vision


2.  **Set up the Python Environment:**


3.  **Set up your IP Camera:**
    - Install an IP camera app on your phone.
    - Start the video stream and note the **video stream URL**. This is often different from the main IP address and might end in `/video` or `/shot.jpg`.

## Usage

Run the script from your terminal with the virtual environment activated. You must provide the video stream URL as an argument.
> python PetCam.py <your_video_stream_url>

A window will open showing the live feed with the AI analysis overlaid. Press `q` with the window in focus to quit.


