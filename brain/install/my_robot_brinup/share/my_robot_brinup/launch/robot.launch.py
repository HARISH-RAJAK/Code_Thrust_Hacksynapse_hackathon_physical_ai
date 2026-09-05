import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    gazebo = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("bumperbot_description"),
            "launch",
            "cobot.launch.py"
        ),
    )
    
    controller = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("bumperbot_description"),
            "launch",
            "spawn_cobot_controllers.launch.py"
        ),
    )
    

    bridge = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("bumperbot_description"),
            "launch",
            "bridge.launch.py"
        ),
    )


    amcl = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("bumperbot_description"),
            "launch",
            "amcl.launch.py"
        ),
    )

    navigation = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("bumperbot_description"),
            "launch",
            "navigation.launch.py"
        ),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(
                get_package_share_directory("nav2_bringup"),
                "rviz",
                "nav2_default_view.rviz"
            )
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    repub_node = Node(
        package='simple_bumperbot_py_examples',
        executable='vel_repub',
        name='relay_node',
        output='screen'
    )    
    
    return LaunchDescription([
        gazebo,
        amcl,
        controller,
        bridge,
        navigation,
        rviz,
        repub_node,
    ])