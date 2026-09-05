import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'voice_assistant'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'sounddevice', 'numpy', 'requests', 'scipy', 'funasr'],
    zip_safe=True,
    maintainer='Team Code Thrust',
    maintainer_email='harish@code-thrust.com',
    description='ROS 2 Modular Voice Assistant with Bluetooth Mic, FunASR STT, and Ollama Qwen3 LLM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mic_node = voice_assistant.mic_node:main',
            'stt_node = voice_assistant.stt_node:main',
            'ollama_node = voice_assistant.ollama_node:main',
        ],
    },
)
