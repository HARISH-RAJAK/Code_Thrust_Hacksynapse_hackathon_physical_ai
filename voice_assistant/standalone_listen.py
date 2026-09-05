#!/usr/bin/env python3
"""
Standalone Voice Pipeline (Mic -> FunASR -> Ollama Qwen3)
Can be run directly with Python: python standalone_listen.py
"""

import time
import queue
import requests
import numpy as np
import sounddevice as sd

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False

try:
    import speech_recognition as sr
    import io
    import scipy.io.wavfile as wavfile
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


def find_bluetooth_mic():
    """Auto-detects Bluetooth microphone index."""
    try:
        devices = sd.query_devices()
        bt_keywords = ['bluetooth', 'headset', 'hands-free', 'bt', 'wireless', 'airpods']
        for idx, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:
                dev_name = dev.get('name', '').lower()
                for kw in bt_keywords:
                    if kw in dev_name:
                        print(f"[+] Detected Bluetooth Mic: '{dev['name']}' (Device Index {idx})")
                        return idx
        
        default_dev = sd.default.device[0]
        dev_name = devices[default_dev].get('name', 'Default') if default_dev is not None else 'Default'
        print(f"[*] Bluetooth mic not found. Fallback to default mic: '{dev_name}' (Device Index {default_dev})")
        return default_dev
    except Exception as e:
        print(f"[!] Warning checking audio devices: {e}")
        return None


def query_ollama(text, model="qwen3:8b", url="http://localhost:11434/api/generate"):
    """Queries local Ollama API."""
    print(f"\n[*] Sending text to Ollama ({model}): '{text}'...")
    payload = {
        "model": model,
        "prompt": text,
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            reply = resp.json().get('response', '').strip()
            print(f"[+] Qwen3 Response: '{reply}'\n")
            return reply
        else:
            print(f"[-] Ollama HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[-] Could not connect to Ollama: {e}")
    return None


def start_listening(sample_rate=16000, model_name="paraformer-zh"):
    print("=" * 60)
    print("        Standalone Voice Listener (Mic -> STT -> Ollama)")
    print("=" * 60)

    model = None
    sr_recognizer = None

    mic_index = find_bluetooth_mic()

    if FUNASR_AVAILABLE:
        print(f"[*] Loading FunASR model ('{model_name}')...")
        try:
            model = AutoModel(model=model_name, vad_model="fsmn-vad", punc_model="ct-punc", disable_update=True)
            print("[+] FunASR STT Model loaded successfully!")
        except Exception as e:
            print(f"[!] FunASR model load warning: {e}")
            if SR_AVAILABLE:
                sr_recognizer = sr.Recognizer()
                print("[+] Fallback to SpeechRecognition STT Engine.")
    elif SR_AVAILABLE:
        sr_recognizer = sr.Recognizer()
        print("[+] SpeechRecognition STT Engine loaded (FunASR unavailable).")
    else:
        print("[-] Neither 'funasr' nor 'SpeechRecognition' is installed.")
        return

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[!] Stream status: {status}")
        audio_q.put(indata.copy())

    print("\n[*] Listening continuously... Speak into your microphone!")
    print("[*] Press Ctrl+C to stop.\n")

    silence_threshold = 0.015
    audio_buffer = []
    is_speaking = False
    last_speech_time = time.time()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32',
                        blocksize=1600, device=mic_index, callback=audio_callback):
        try:
            while True:
                while not audio_q.empty():
                    chunk = audio_q.get().flatten()
                    rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0

                    if rms > silence_threshold:
                        if not is_speaking:
                            print("[*] Voice activity detected...")
                        is_speaking = True
                        last_speech_time = time.time()
                        audio_buffer.extend(chunk)
                    elif is_speaking:
                        audio_buffer.extend(chunk)
                        if time.time() - last_speech_time > 0.8:
                            is_speaking = False

                buffer_duration = len(audio_buffer) / float(sample_rate)
                if (not is_speaking and buffer_duration >= 0.8) or (buffer_duration >= 5.0):
                    audio_data = np.array(audio_buffer, dtype=np.float32)
                    audio_buffer.clear()

                    clean_text = ""
                    if model is not None:
                        print("[*] Transcribing with FunASR...")
                        res = model.generate(input=audio_data, sample_rate=sample_rate)
                        if isinstance(res, list) and len(res) > 0:
                            clean_text = res[0].get('text', '').strip()
                        elif isinstance(res, dict):
                            clean_text = res.get('text', '').strip()
                    elif sr_recognizer is not None:
                        print("[*] Transcribing with SpeechRecognition...")
                        pcm_int16 = (audio_data * 32767).astype(np.int16)
                        byte_io = io.BytesIO()
                        wavfile.write(byte_io, sample_rate, pcm_int16)
                        byte_io.seek(0)
                        with sr.AudioFile(byte_io) as source:
                            audio_listened = sr_recognizer.record(source)
                            try:
                                clean_text = sr_recognizer.recognize_google(audio_listened, language="hi-IN")
                            except Exception:
                                try:
                                    clean_text = sr_recognizer.recognize_google(audio_listened)
                                except Exception:
                                    clean_text = ""

                    if clean_text:
                        print(f"[+] Recognized Speech: '{clean_text}'")
                        query_ollama(clean_text)
                    else:
                        print("[*] No clear speech recognized.")

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[*] Stopping listener.")


if __name__ == '__main__':
    start_listening()
