from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voice_assistant',
            executable='mic_node',
            name='mic_node',
            output='screen',
            parameters=[
                {'sample_rate': 16000},
                {'channels': 1},
                {'chunk_size': 1600},
                {'device_index': -1}  # -1 for Auto-detect Bluetooth Mic
            ]
        ),
        Node(
            package='voice_assistant',
            executable='stt_node',
            name='stt_node',
            output='screen',
            parameters=[
                {'sample_rate': 16000},
                {'silence_threshold': 0.015},
                {'max_buffer_seconds': 5.0},
                {'min_speech_duration': 0.8},
                {'model_name': 'paraformer-zh'}
            ]
        ),
        Node(
            package='voice_assistant',
            executable='ollama_node',
            name='ollama_node',
            output='screen',
            parameters=[
                {'ollama_url': 'http://localhost:11434/api/generate'},
                {'model_name': 'qwen3:8b'},
                {'system_prompt': 'You are a helpful ROS 2 Physical AI Robot Assistant. Give short, direct answers.'}
            ]
        )
    ])
