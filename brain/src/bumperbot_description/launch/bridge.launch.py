import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # 1. Package path find karein
    pkg_share = get_package_share_directory("bumperbot_description")

    # 2. YAML file ka path define karein
    bridge_config_path = os.path.join(pkg_share, "config", "ros_gz_bridge.yaml")

    # 3. Launch Argument declare karein
    bridge_config_arg = DeclareLaunchArgument(
        name="bridge_config", 
        default_value=bridge_config_path,
        description="Absolute path to the bridge configuration YAML file"
    )

    # 4. Node define karein (arguments ki jagah parameters use karein)
    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            'config_file': LaunchConfiguration('bridge_config')
        }],
        output='screen'
    )


    # Includes optimizations to minimize latency and bandwidth when streaming image data
    start_gazebo_ros_image_bridge_cmd = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=[
            '/camera/image',
            '/camera/depth_image'
        ],
        remappings=[
            ('/camera/image', '/camera/color/image_raw'),
            ('/camera/depth_image' ,'/camera/depth_image')
        ])    

    return LaunchDescription([
        bridge_config_arg,
        gz_ros2_bridge,
        start_gazebo_ros_image_bridge_cmd
    ])