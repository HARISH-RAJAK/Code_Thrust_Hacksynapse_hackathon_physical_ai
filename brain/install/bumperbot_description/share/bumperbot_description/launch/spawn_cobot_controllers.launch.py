from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )
    
    hiwonder_xarm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['hiwonder_xarm_controller'],
        output='screen'
    )
    
    
    bumperbot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['bumperbot_controller'],
        output='screen'
    )
    return LaunchDescription([
        joint_state_broadcaster_spawner,
        hiwonder_xarm_controller_spawner,
        bumperbot_controller_spawner
    ])
