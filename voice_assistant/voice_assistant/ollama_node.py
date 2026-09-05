#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json


class OllamaNode(Node):
    def __init__(self):
        super().__init__('ollama_node')

        # Parameters
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/generate')
        self.declare_parameter('model_name', 'qwen3:8b')
        self.declare_parameter('system_prompt', 'You are a helpful ROS 2 Physical AI Robot Assistant. Give short, direct answers.')

        self.ollama_url = self.get_parameter('ollama_url').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.system_prompt = self.get_parameter('system_prompt').get_parameter_value().string_value

        # Publisher & Subscriber
        self.response_pub = self.create_publisher(String, '/llm_response', 10)
        self.speech_sub = self.create_subscription(String, '/speech_text', self.speech_callback, 10)

        self.get_logger().info(f"[*] Initialized OllamaNode | URL: {self.ollama_url} | Model: {self.model_name}")

    def speech_callback(self, msg: String):
        """Called when clean text is published to /speech_text."""
        user_text = msg.data.strip()
        if not user_text:
            return

        self.get_logger().info(f"[*] Received Speech Text: '{user_text}' -> Querying Ollama ({self.model_name})...")

        payload = {
            "model": self.model_name,
            "prompt": user_text,
            "system": self.system_prompt,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                llm_reply = data.get('response', '').strip()
                
                self.get_logger().info(f"[+] Ollama ({self.model_name}) Response: '{llm_reply}'")

                # Publish to /llm_response
                out_msg = String()
                out_msg.data = llm_reply
                self.response_pub.publish(out_msg)
            else:
                self.get_logger().error(f"[-] Ollama API HTTP {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            self.get_logger().error(f"[-] Could not connect to Ollama at {self.ollama_url}. Make sure 'ollama serve' is running!")
        except Exception as e:
            self.get_logger().error(f"[-] Error calling Ollama API: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OllamaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
