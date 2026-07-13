# Coverless Audio Steganography: A Generative Approach

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Framework-Flask-green)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Publication](https://img.shields.io/badge/Publication-JETIR-red)](https://www.jetir.org/view?paper=JETIR2503535)

> **Research Publication:** This repository serves as the practical implementation and continuation of the research detailed in the published paper: [*Coverless Audio Steganography* (JETIR2503535)](https://www.jetir.org/view?paper=JETIR2503535). 

## 📖 Abstract

Traditional audio steganography algorithms rely on embedding secret data within the noise floor or least significant bits (LSB) of existing audio carrier files. This leaves a statistical footprint, rendering the communication susceptible to modern steganalysis. 

This project introduces a **Coverless Audio Steganography** paradigm. Instead of modifying an existing carrier, the system dynamically synthesizes the audio signal from scratch (algorithmic generative synthesis), embedding the secret message simultaneously during the generation process. By integrating the ciphertext intrinsically into the structural mathematics of the audio track (specifically high-frequency modulation layered beneath lo-fi beat structures), the method completely circumvents footprint-based detection mechanisms.

## ✨ Core Architecture

- **Algorithmic Synthesis Engine**: Utilizes generative structures (`numpy`/`scipy` based modulation) to create layered audio tracks—including dynamic percussion, chord progressions, and ambient textures.
- **Spectrum-Masked Embedding**: Maps binary cipherstreams to specific, high-frequency pairings (e.g., 17kHz - 19kHz) that are structurally masked by lower-frequency musical elements (< 15kHz).
- **Coverless Paradigm**: Zero modification to existing files. The carrier *is* the generated output, inherently resistant to comparative steganalysis.
- **Secure Web Terminal**: A seamless, CRT-styled command-line interface web application demonstrating the encoding/decoding workflows securely over a Flask backend.

## 🛠️ Technical Stack

- **Signal Processing**: `numpy`, `scipy`, `pydub`, `soundfile`
- **Generative Frameworks**: Vector Quantization / Deep Feature Modeling context (per paper constraints) mapped to local deterministic generation.
- **Web Infrastructure**: Python `Flask`, Jinja2, Vanilla JS/CSS

## 🚀 Installation & Usage

### Prerequisites
Ensure you have `Python 3.11+` and `git` installed on your system.

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rakesh-pathuri/Coverless-Audio-steganography-using-VQ-VAE-2.git
   cd Coverless-Audio-steganography-using-VQ-VAE-2
   ```

2. **Establish a Virtual Environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the Server:**
   ```bash
   python main.py
   ```

5. **Access the Interface:**
   Navigate to `http://localhost:5000` in your web browser. Authenticate using the default secure operator credentials to access the steganography terminal.

## 🔬 Decoding Methodology
Upon receiving the synthesized audio file, the decoder processes the `.wav` payload through Fast Fourier Transforms (FFT) to isolate the high-frequency spectra. It reverse-maps the specific frequency bands back into the binary stream, subsequently decoding the UTF-8 text, completing the asymmetric coverless transfer.

---
*Developed by Rakesh Pathuri as part of advancing research in Information Security & Applied Cryptography.*
