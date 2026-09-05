import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bumperbot_description = get_package_share_directory("bumperbot_description")

    model_arg = DeclareLaunchArgument(
        name="model", default_value=os.path.join(
                bumperbot_description, "cobot_urdf", "ebot.urdf.xacro"
            ),
        description="Absolute path to robot urdf file"
    )

    world_name_arg = DeclareLaunchArgument(name="world_name", default_value="small_house")

    world_path = PathJoinSubstitution([
            bumperbot_description,
            "worlds",
            PythonExpression(expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"])
        ]
    )

    model_path = str(Path(bumperbot_description).parent.resolve())
    model_path += pathsep + os.path.join(get_package_share_directory("bumperbot_description"), 'models')

    gazebo_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        model_path
        )

    ros_distro = os.environ.get("ROS_DISTRO", "jazzy")
    is_ignition = "True" if ros_distro == "humble" else "False"

    robot_description = ParameterValue(Command([
            "xacro ",
            LaunchConfiguration("model"),
            " is_ignition:=",
            is_ignition
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}]
    )

    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory("ros_gz_sim"), "launch"), "/gz_sim.launch.py"]),
                launch_arguments={
                    "gz_args": PythonExpression(["'", world_path, " -v 4 -r'"])
                }.items()
             )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description",
                   "-name", "my_robot",
                    "-x", "2.0",      # 2 meters in X
                    "-y", "1.0",      # 1 meter in Y
                    "-z", "0.2",      # 20 cm above ground

                    "-R", "0.0",      # Roll (radians)
                    "-P", "0.0",      # Pitch (radians)
                    "-Y", "0.0"      # Yaw (1.57 rad ≈ 90°)
                    ],
        parameters=[{"use_sim_time": False}]
    )


    # Delay spawn until Gazebo has had time to start (5 seconds)
    gz_spawn_entity_delayed = TimerAction(
        period=5.0,
        actions=[gz_spawn_entity]
    )

    return LaunchDescription([
        model_arg,
        world_name_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity_delayed,
    ])
