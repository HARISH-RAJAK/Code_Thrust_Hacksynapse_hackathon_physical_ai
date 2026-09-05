#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import threading
import time

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


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')

        # Parameters
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('silence_threshold', 0.015)
        self.declare_parameter('max_buffer_seconds', 5.0)
        self.declare_parameter('min_speech_duration', 0.8)
        self.declare_parameter('model_name', 'paraformer-zh')

        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.silence_threshold = self.get_parameter('silence_threshold').get_parameter_value().double_value
        self.max_buffer_seconds = self.get_parameter('max_buffer_seconds').get_parameter_value().double_value
        self.min_speech_duration = self.get_parameter('min_speech_duration').get_parameter_value().double_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value

        # Publisher & Subscriber
        self.text_pub = self.create_publisher(String, '/speech_text', 10)
        self.audio_sub = self.create_subscription(Float32MultiArray, '/audio', self.audio_callback, 10)

        # Internal audio buffer & lock
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.last_speech_time = time.time()
        self.is_speaking = False

        # Load STT Engine (FunASR or SpeechRecognition fallback)
        self.model = None
        self.sr_recognizer = None

        if FUNASR_AVAILABLE:
            self.get_logger().info(f"[*] Initializing FunASR model ('{self.model_name}')...")
            try:
                self.model = AutoModel(
                    model=self.model_name,
                    vad_model="fsmn-vad",
                    punc_model="ct-punc",
                    disable_update=True
                )
                self.get_logger().info("[+] FunASR STT Model loaded successfully!")
            except Exception as e:
                self.get_logger().warn(f"[!] FunASR load error: {e}")
                if SR_AVAILABLE:
                    self.sr_recognizer = sr.Recognizer()
                    self.get_logger().info("[+] Fallback to SpeechRecognition STT Engine.")
        elif SR_AVAILABLE:
            self.sr_recognizer = sr.Recognizer()
            self.get_logger().info("[+] SpeechRecognition STT Engine initialized (FunASR unavailable).")
        else:
            self.get_logger().error("[-] Neither FunASR nor SpeechRecognition installed.")

        # Worker thread for processing audio buffers
        self.processing_thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        self.processing_thread.start()

    def audio_callback(self, msg: Float32MultiArray):
        """Receives float32 PCM audio chunks from /audio topic."""
        chunk = np.array(msg.data, dtype=np.float32)
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0

        with self.buffer_lock:
            if rms > self.silence_threshold:
                if not self.is_speaking:
                    self.get_logger().info("[*] Speech activity detected...")
                self.is_speaking = True
                self.last_speech_time = time.time()
                self.audio_buffer.extend(chunk)
            elif self.is_speaking:
                self.audio_buffer.extend(chunk)
                # If silence for > 0.8s, trigger STT
                if time.time() - self.last_speech_time > 0.8:
                    self.is_speaking = False

    def process_audio_loop(self):
        """Periodic loop to run FunASR on accumulated audio buffers."""
        while rclpy.ok():
            time.sleep(0.1)

            audio_data = None
            with self.buffer_lock:
                buffer_duration = len(self.audio_buffer) / float(self.sample_rate)
                
                # Process if speech ended OR max buffer duration reached
                if (not self.is_speaking and buffer_duration >= self.min_speech_duration) or (buffer_duration >= self.max_buffer_seconds):
                    audio_data = np.array(self.audio_buffer, dtype=np.float32)
                    self.audio_buffer.clear()

            if audio_data is not None and len(audio_data) > 0:
                self.transcribe_and_publish(audio_data)

    def transcribe_and_publish(self, audio_data: np.ndarray):
        """Passes audio array to STT model (FunASR or SpeechRecognition) and publishes clean text to /speech_text."""
        if self.model is None and self.sr_recognizer is None:
            self.get_logger().error("[-] No STT model (FunASR or SpeechRecognition) is initialized.")
            return

        clean_text = ""
        try:
            if self.model is not None:
                self.get_logger().info("[*] Running FunASR Speech-to-Text inference...")
                res = self.model.generate(input=audio_data, sample_rate=self.sample_rate)
                if isinstance(res, list) and len(res) > 0:
                    clean_text = res[0].get('text', '').strip()
                elif isinstance(res, dict):
                    clean_text = res.get('text', '').strip()

            elif self.sr_recognizer is not None:
                self.get_logger().info("[*] Running SpeechRecognition STT inference...")
                pcm_int16 = (audio_data * 32767).astype(np.int16)
                byte_io = io.BytesIO()
                wavfile.write(byte_io, self.sample_rate, pcm_int16)
                byte_io.seek(0)
                
                with sr.AudioFile(byte_io) as source:
                    audio_listened = self.sr_recognizer.record(source)
                    try:
                        clean_text = self.sr_recognizer.recognize_google(audio_listened, language="hi-IN")
                    except sr.UnknownValueError:
                        clean_text = ""
                    except Exception:
                        try:
                            clean_text = self.sr_recognizer.recognize_google(audio_listened)
                        except Exception:
                            clean_text = ""

            if clean_text:
                self.get_logger().info(f"[+] Recognized Text: '{clean_text}'")
                msg = String()
                msg.data = clean_text
                self.text_pub.publish(msg)
            else:
                self.get_logger().info("[*] STT: No clear speech recognized in audio chunk.")

        except Exception as e:
            self.get_logger().error(f"[-] Error during STT transcription: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
