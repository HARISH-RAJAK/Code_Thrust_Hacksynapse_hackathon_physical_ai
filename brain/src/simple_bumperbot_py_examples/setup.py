from setuptools import find_packages, setup

package_name = 'simple_bumperbot_py_examples'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ram2',
    maintainer_email='ram2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ["simple_turtlesim_kinematics = simple_bumperbot_py_examples.simple_turtlesim_kinematics:main",
        "gripper_mimic = simple_bumperbot_py_examples.mimic:main",
        "camera_echo = simple_bumperbot_py_examples.camera_echo:main",
        "gui = simple_bumperbot_py_examples.gui:main",
        "camera_perception = simple_bumperbot_py_examples.camera_perception:main",
        "inverse = simple_bumperbot_py_examples.inverse:main",
        "inverse_with_orientation = simple_bumperbot_py_examples.inverse_with_orient:main",
        "vel_repub = simple_bumperbot_py_examples.vel_republisher:main",
        "tf_to_ik = simple_bumperbot_py_examples.tf_to_ik:main",
        "command = simple_bumperbot_py_examples.command:main",
        "imu_calibration = simple_bumperbot_py_examples.imu_callibration:main",
        "interactive_waypoint_follower = simple_bumperbot_py_examples.interactive_waypoint_follower:main",
        "goal_pose_publisher = simple_bumperbot_py_examples.goal_pose_publisher:main"
        ],
    },
)
