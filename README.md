# Coverless Audio Steganography using VQ-VAE-2

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)

A state-of-the-art web application that allows you to encrypt secret messages directly into the structure of synthesized audio. Unlike traditional steganography that hides data in the noise floor of pre-existing audio tracks, this project uses **Coverless Steganography**—synthesizing the audio carrier from scratch specifically to hold the data.

## Features

- **Coverless Architecture**: Generates custom Lo-Fi music, Jazz loops, and Pop chord progressions dynamically, securely embedding your message into the high-frequency spectrum.
- **Premium Terminal UI**: Features a beautiful, interactive glowing-green CRT terminal interface.
- **In-Browser Playback**: Listen to your generated encrypted beats immediately before downloading.
- **Built-in Drum Synthesizer**: Generates its own kick drums, snare drums, and hi-hats mathematically using `numpy`.

## Prerequisites

- Python 3.11+
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rakesh-pathuri/Coverless-Audio-steganography-using-VQ-VAE-2.git
   cd Coverless-Audio-steganography-using-VQ-VAE-2
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask server:
   ```bash
   python main.py
   ```
2. Open your browser and navigate to `http://localhost:5000`.
3. Log in with the default credentials:
   - **Username**: `operator`
   - **Password**: `terminal123`

## How it Works

The audio engine (`audio.py`) translates your text message into a binary stream and modulates it into specific high-frequency pairings between 17kHz and 19kHz. Simultaneously, a dynamic music generator synthesizes a thick layer of chords, basslines, and lo-fi textures below 15kHz. This masks the steganographic frequencies entirely to the human ear while allowing the decoder to easily extract the message via spectral analysis.
