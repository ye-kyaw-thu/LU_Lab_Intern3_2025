# Speech Training Recorder (LU Lab., Myanmar) 🎤

A Python-based tool for recording prompted speech data, specifically designed for ASR/TTS corpus creation with Myanmar language support.

![Application Screenshot](screenshot.png)

## Features ✨

- **Multiple Prompt Modes**: 
  - Random (for ASR training)
  - Ordered (preserve original sequence)
  - Sequential (cycle through prompts)
- **Smart Output Handling**:
  - Auto-generated timestamped folders (e.g., `rec_1439_14Jun2025`)
  - WAV audio + TSV metadata recording
- **Efficient Workflow**:
  - Keyboard shortcuts for all actions
  - Built-in playback verification
  - Automatic Myanmar font detection
- **Customizable Audio**:
  - Adjustable sample rates (8kHz-48kHz)
  - 16/32-bit depth options

## Installation 💻

```bash
pip install PyQt6 sounddevice numpy
