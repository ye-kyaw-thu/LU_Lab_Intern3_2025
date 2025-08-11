"""
Myanmar specific IPA (e.g. myG2P) to espeak sound converter.
Written by Ye Kyaw Thu.
Last updated: 9 Aug 2025

Usage:
    python ./ipa_my2speech.py --input ./test1.ipa.txt --output 8_ethnics.wav

ye@lst-hpc3090:~/exp/vs/ipa2speech$ cat test1.ipa.txt
kə tɕʰɪ̀ɴ
kə já
kə jɪ̀ɴ
tɕʰɪ́ɴ
mʊ̀ɴ
bə mà
jə kʰàɪɴ
ʃáɴ
"""

import argparse
import subprocess
import sys
import tempfile
import os
import soundfile as sf
import numpy as np
from pathlib import Path

# Comprehensive IPA to eSpeak mapping with Myanmar-specific symbols
IPA_TO_ESPEAK = {
    # Standard IPA consonants
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
    'm': 'm', 'n': 'n', 'ŋ': 'N', 'f': 'f', 'v': 'v', 'θ': 'T',
    'ð': 'D', 's': 's', 'z': 'z', 'ʃ': 'S', 'ʒ': 'Z', 'h': 'h',
    'tʃ': 'tS', 'dʒ': 'dZ', 'l': 'l', 'r': 'r', 'j': 'j', 'w': 'w',
    'ɕ': 's', 'ʑ': 'z', 'ɲ': 'J', 'ɬ': 'K', 'ɮ': 'H', 'ʍ': 'W',
    
    # Myanmar-specific consonants
    'ɴ': 'n',  # Myanmar final nasal
    'ɹ': 'r',   # Alternative rhotic
    'ʈ': 't', 'ɖ': 'd',  # Retroflex
    'ɽ': 'r',   # Retroflex flap
    'ɸ': 'f', 'β': 'v',  # Bilabial fricatives
    'ʂ': 'S', 'ʐ': 'Z',  # Retroflex fricatives
    'ʕ': '?', 'ħ': 'h',  # Pharyngeal
    'ʘ': 'p', 'ǀ': 't', 'ǃ': '!', 'ǂ': 'c', 'ǁ': 'l',  # Clicks
    
    # Vowels
    'i': 'i', 'ɪ': 'I', 'e': 'e', 'ɛ': 'E', 'æ': '&',
    'a': 'a', 'ɑ': 'A', 'ɔ': 'O', 'o': 'o', 'ʊ': 'U', 'u': 'u',
    'ʌ': 'V', 'ə': '@', 'ɚ': '@`', 'ɝ': '3', 'ɜ': '3',
    
    # Myanmar-specific vowels
    'ɯ': 'M', 'ɤ': '7', 'ɨ': '1', 'ʉ': '2',
    
    # Tone markers (Myanmar uses these as diacritics)
    '̀': '_L',  # Low tone
    '́': '_H',  # High tone
    '̂': '_R',  # Rising tone
    '̌': '_F',  # Falling tone
    '̄': '_M',  # Mid tone
    '̋': '_HH', # Extra high
    '̏': '_LL', # Extra low
    
    # Diacritics
    'ʰ': '_h',  # Aspirated
    'ʲ': '_j',  # Palatalized
    'ʷ': '_w',  # Labialized
    'ˠ': '_G',  # Velarized
    'ˤ': '_?',  # Pharyngealized
    '̃': '~',    # Nasalized
    '̥': '_0',   # Voiceless
    '̤': '_t',   # Breathy voice
    '̪': '_d',   # Dental
    '̺': '_a',   # Apical
    '̻': '_c',   # Laminal
    '̟': '+',    # Advanced
    '̠': '-',    # Retracted
    '̈': '"',     # Centralized
    '̽': 'x',    # Mid-centralized
    '̩': '=',    # Syllabic
    '̯': '^',    # Non-syllabic
    '̚': '.',    # No audible release
    '̰': '_c',   # Creaky voice
    '̱': '_',    # Weak articulation
    
    # Length markers
    'ː': ':',   # Long
    'ˑ': ':',   # Half-long
    '̆': '',     # Extra-short
    
    # Special Myanmar symbols (precomposed characters)
    'á': 'a_H', 'à': 'a_L', 'ā': 'a_M', 'â': 'a_R', 'ǎ': 'a_F',
    'a̰': 'a_c', 'a̱': 'a_', 'a̋': 'a_HH', 'ȁ': 'a_LL',
    
    'é': 'e_H', 'è': 'e_L', 'ē': 'e_M', 'ê': 'e_R', 'ě': 'e_F',
    'ḛ': 'e_c', 'e̱': 'e_', 'e̋': 'e_HH', 'ȅ': 'e_LL',
    'ὲ': 'E_L', 'έ': 'E_H', 'ɛ̄': 'E_M', 'ɛ̂': 'E_R', 'ɛ̌': 'E_F',
    'ɛ̰': 'E_c', 'ḛ': 'E_c',
    
    'í': 'i_H', 'ì': 'i_L', 'ī': 'i_M', 'î': 'i_R', 'ǐ': 'i_F',
    'ḭ': 'i_c', 'i̱': 'i_', 'i̋': 'i_HH', 'ȉ': 'i_LL',
    'ḭ': 'i_c',
    
    'ó': 'o_H', 'ò': 'o_L', 'ō': 'o_M', 'ô': 'o_R', 'ǒ': 'o_F',
    'o̰': 'o_c', 'o̱': 'o_', 'ő': 'o_HH', 'ȍ': 'o_LL',
    'ɔ́': 'O_H', 'ɔ̀': 'O_L', 'ɔ̄': 'O_M', 'ɔ̂': 'O_R', 'ɔ̌': 'O_F',
    'ɔ̰': 'O_c',
    
    'ú': 'u_H', 'ù': 'u_L', 'ū': 'u_M', 'û': 'u_R', 'ǔ': 'u_F',
    'ṵ': 'u_c', 'u̱': 'u_', 'ű': 'u_HH', 'ȕ': 'u_LL',
    'ʊ́': 'U_H', 'ʊ̀': 'U_L', 'ʊ̄': 'U_M', 'ʊ̂': 'U_R', 'ʊ̌': 'U_F',
    'ʊ̰': 'U_c', 'ṵ': 'u_c', 'ṷ': 'u_w',
    
    # Affricates and other complex symbols
    'tɕ': 'ts', 'dʑ': 'dz', 'tɕʰ': 'ts_h', 'ʥ': 'dz',
    'pf': 'pf', 'bv': 'bv', 'ts': 'ts', 'dz': 'dz',
    'tɬ': 'tl', 'dɮ': 'dl', 'kx': 'kx', 'gɣ': 'gG',
    
    # Punctuation and special characters
    ' ': ' ', '.': '.', ',': ',', 'ʔ': '?', "'": "'",
    'ˈ': "'",   # Primary stress
    'ˌ': ",",   # Secondary stress
    '‖': '|',   # Minor break
    '‿': '_',   # Linking
}

def convert_ipa_to_espeak(ipa_text):
    """Convert Myanmar IPA to eSpeak NG notation with tone handling"""
    converted = []
    i = 0
    while i < len(ipa_text):
        # Check for multi-character sequences first (up to 3 chars)
        found = False
        for length in [3, 2, 1]:
            if i + length <= len(ipa_text):
                current = ipa_text[i:i+length]
                if current in IPA_TO_ESPEAK:
                    converted.append(IPA_TO_ESPEAK[current])
                    i += length
                    found = True
                    break
        
        if not found:
            # Handle unknown characters
            char = ipa_text[i]
            # Check if it's a combining character
            if ord(char) >= 0x300 and ord(char) <= 0x36F:
                # Try to find a similar diacritic
                base = char
                if base in ['̀', '́', '̂', '̌', '̄']:
                    converted.append('_' + {'̀':'L', '́':'H', '̂':'R', '̌':'F', '̄':'M'}[base])
                else:
                    converted.append(char)
                    print(f"Warning: No mapping for diacritic '{char}'", file=sys.stderr)
            else:
                converted.append(char)
                print(f"Warning: No mapping for symbol '{char}'", file=sys.stderr)
            i += 1
    
    return ''.join(converted)

def check_espeak_installed():
    try:
        result = subprocess.run(['espeak', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=True)
        return True, result.stdout.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, str(e)

def ipa_to_speech(ipa_text, voice='en-us', speed=120, output_file=None):
    """
    Convert IPA text to speech using espeak command-line tool
    
    Args:
        ipa_text (str): IPA phonetic text
        voice (str): Voice to use (default 'en-us')
        speed (int): Speech speed in words per minute
        output_file (str): Optional path to save WAV file
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    # Convert IPA to eSpeak notation
    espeak_text = convert_ipa_to_espeak(ipa_text)
    print(f"Converted IPA to eSpeak: {ipa_text} -> {espeak_text}", file=sys.stderr)
    
    # Create temporary file
    temp_path = Path(tempfile.mktemp(suffix='.wav'))
    
    try:
        # Build espeak command
        cmd = [
            'espeak',
            '-v', voice,
            '-s', str(speed),
            '-w', str(temp_path),
            '[[%s]]' % espeak_text  # Use phoneme input format
        ]
        
        # Run espeak
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        
        # Read the generated WAV file
        data, sample_rate = sf.read(temp_path)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Save to output file if requested
        if output_file:
            sf.write(output_file, data, sample_rate)
            
        return data, sample_rate
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"espeak failed: {e.stderr.decode().strip()}") from e
    except Exception as e:
        raise RuntimeError(f"Error during conversion: {str(e)}") from e
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(
        description='Convert IPA (International Phonetic Alphabet) to speech using espeak',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, 
                       help='Input file containing IPA text (one per line)')
    parser.add_argument('--output', type=str, 
                       help='Output WAV file (if not specified, returns audio data)')
    parser.add_argument('--voice', type=str, default='en-us',
                       help='Voice to use for speech synthesis')
    parser.add_argument('--speed', type=int, default=120,
                       help='Speech speed in words per minute')
    
    args = parser.parse_args()
    
    # Check dependencies
    espeak_installed, espeak_version = check_espeak_installed()
    if not espeak_installed:
        print("Error: espeak is not installed or not found in PATH.", file=sys.stderr)
        print("On Ubuntu/Debian: sudo apt-get install espeak-ng", file=sys.stderr)
        print("On macOS: brew install espeak-ng", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using espeak version: {espeak_version}", file=sys.stderr)
    
    # Read input
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                ipa_lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading input file: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Enter IPA text (press Ctrl+D to finish):", file=sys.stderr)
        ipa_lines = [line.strip() for line in sys.stdin if line.strip()]
    
    if not ipa_lines:
        print("Error: No IPA text provided.", file=sys.stderr)
        sys.exit(1)
    
    ipa_text = ' '.join(ipa_lines)
    print(f"Processing IPA text: {ipa_text}", file=sys.stderr)
    
    try:
        audio_data, sample_rate = ipa_to_speech(
            ipa_text, 
            voice=args.voice, 
            speed=args.speed, 
            output_file=args.output
        )
        
        if args.output:
            print(f"Successfully generated speech and saved to {args.output}", file=sys.stderr)
        else:
            print(f"Successfully generated speech (sample rate: {sample_rate} Hz)", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

