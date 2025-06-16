# Speech Training Recorder (LU Lab., Myanmar) 🎤

A Python-based tool for recording prompted speech data, specifically designed for ASR/TTS corpus creation with Myanmar language support.  

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/tools/recorder/recorder_UI.png" alt="UI" width="380"/>  
</p>  
<div align="center">
  Fig. UI of recorder  
</div> 

<br />  

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
```

## Usage 🚀

### Basic Recording

```
python recorder.py -p prompts.txt
```

### Advanced Options

```
# Sequential mode with auto-advance
python recorder.py -p prompts.txt -m sequential -a

# Custom output folder and high-quality audio
python recorder.py -p prompts.txt -d my_recordings -sr 48000 -b 32
```

### Full Help

```
$python recorder.py --help
usage: recorder.py [-h] [-p PROMPTS_FILENAME] [-d SAVE_DIR] [-m {random,ordered,sequential}] [-c PROMPTS_COUNT]
                   [-l PROMPT_LEN_SOFT_MAX] [-a] [-sr {8000,16000,44100,48000}] [-b {16,32}]

Speech Training Recorder (LU Lab., Myanmar) - Record prompted speech

optional arguments:
  -h, --help            show this help message and exit
  -p PROMPTS_FILENAME, --prompts_filename PROMPTS_FILENAME
                        text file containing prompts (one per line)
  -d SAVE_DIR, --save_dir SAVE_DIR
                        custom output directory (default: auto-generated)
  -m {random,ordered,sequential}, --prompt_selection {random,ordered,sequential}
                        prompt selection mode (default: random)
  -c PROMPTS_COUNT, --prompts_count PROMPTS_COUNT
                        max prompts to use (default: 100)
  -l PROMPT_LEN_SOFT_MAX, --prompt_len_soft_max PROMPT_LEN_SOFT_MAX
                        maximum prompt length in characters (0=no limit)
  -a, --auto_next       auto-advance to next prompt after save
  -sr {8000,16000,44100,48000}, --sample_rate {8000,16000,44100,48000}
                        sample rate in Hz (default: 16000)
  -b {16,32}, --bit_depth {16,32}
                        bit depth (16 or 32, default: 16)

Example usages:
  recorder.py -p prompts.txt -m random
  recorder.py -p script.txt -m ordered -d custom_folder
  recorder.py -p phrases.txt -m sequential -a

```

### Key Controls ⌨️  

### Prompt File Format 📝  

Example ```prompts.txt```: 

```
နေကောင်းလား
ထမင်း စားပြီးပြီလား
သတိရလို့ ဖုန်းလှမ်းခေါ်လိုက်တာပါ
ငါတို့ မတွေ့ဖြစ်တာတောင် ၅နှစ်ကျော်သွားပြီလားလို့
အလုပ်အကိုင်ကော အဆင်ပြေရဲ့လား
```

### Output Structure 📂  

```
rec_1439_14Jun2025/
├── recordings.tsv
├── recording_20250616_123825.wav
├── recording_20250616_123838.wav
└── ...
```

