#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import sounddevice as sd
import numpy as np
import sys


class MicNode(Node):
    def __init__(self):
        super().__init__('mic_node')

        # Declare parameters
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('chunk_size', 1600)
        self.declare_parameter('device_index', -1)

        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.channels = self.get_parameter('channels').get_parameter_value().integer_value
        self.chunk_size = self.get_parameter('chunk_size').get_parameter_value().integer_value
        self.device_index = self.get_parameter('device_index').get_parameter_value().integer_value

        # Publisher for raw audio chunks
        self.audio_pub = self.create_publisher(Float32MultiArray, '/audio', 10)

        # Detect mic device
        if self.device_index < 0:
            self.device_index = self.find_bluetooth_mic()

        self.get_logger().info(f"[*] Initializing MicNode | Device Index: {self.device_index} | Sample Rate: {self.sample_rate}Hz")

        # Start audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.chunk_size,
                device=self.device_index if self.device_index >= 0 else None,
                callback=self.audio_callback
            )
            self.stream.start()
            self.get_logger().info("[+] Continuous Bluetooth Mic Listening started -> Publishing to /audio")
        except Exception as e:
            self.get_logger().error(f"[-] Failed to start audio stream on device {self.device_index}: {e}")

    def find_bluetooth_mic(self):
        """Scans input devices to automatically detect Bluetooth microphone."""
        try:
            devices = sd.query_devices()
            bt_keywords = ['bluetooth', 'headset', 'hands-free', 'bt', 'wireless', 'airpods']
            
            for idx, dev in enumerate(devices):
                if dev.get('max_input_channels', 0) > 0:
                    dev_name = dev.get('name', '').lower()
                    for kw in bt_keywords:
                        if kw in dev_name:
                            self.get_logger().info(f"[+] Found Bluetooth Mic: '{dev['name']}' (Index {idx})")
                            return idx
            
            # Fallback to default input device
            default_dev = sd.default.device[0]
            if default_dev is not None and default_dev >= 0:
                dev_name = devices[default_dev].get('name', 'Default Mic')
                self.get_logger().info(f"[*] Bluetooth mic not found. Fallback to default mic: '{dev_name}' (Index {default_dev})")
                return default_dev

        except Exception as e:
            self.get_logger().warn(f"[!] Error scanning audio devices: {e}")

        return -1

    def audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback for streaming audio chunks."""
        if status:
            self.get_logger().warn(f"[!] Audio stream status: {status}")

        msg = Float32MultiArray()
        # Flatten input numpy float32 array
        msg.data = indata.flatten().tolist()
        self.audio_pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
