#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tkinter as tk
from threading import Thread


# Joint order MUST match controller order
JOINT_CONFIG = [
    ("Link1_to_Link2", -1.57, 1.57),
    ("Link2_to_Link3", -2.26, 2.26),
    ("Link5_to_gripperbase", -2.09, 2.09),
    ("Linkt_to_Link5", -2.26, 2.26),
    ("base_to_Link1", -2.35, 2.35),
    ("gripper_finger1_joint", 0.0, 0.87),
]


class ArmGUI(Node):

    def __init__(self):
        super().__init__('arm_gui_controller')

        self.publisher = self.create_publisher(
            JointTrajectory,
            '/hiwonder_xarm_controller/joint_trajectory',
            10
        )

        self.current_positions = [0.0] * len(JOINT_CONFIG)

        gui_thread = Thread(target=self.start_gui)
        gui_thread.daemon = True
        gui_thread.start()

    def send_trajectory(self):
        msg = JointTrajectory()
        msg.joint_names = [j[0] for j in JOINT_CONFIG]

        point = JointTrajectoryPoint()
        point.positions = self.current_positions
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        msg.points.append(point)
        self.publisher.publish(msg)

    def update_joint(self, index, value):
        self.current_positions[index] = float(value)
        self.send_trajectory()

    def start_gui(self):
        root = tk.Tk()
        root.title("Arm Joint Control")

        for i, (name, lower, upper) in enumerate(JOINT_CONFIG):
            frame = tk.Frame(root)
            frame.pack(padx=10, pady=5)

            label = tk.Label(frame, text=name)
            label.pack()

            slider = tk.Scale(
                frame,
                from_=lower,
                to=upper,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                length=500,
                command=lambda val, idx=i: self.update_joint(idx, val)
            )
            slider.set(0.0)
            slider.pack()

        root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = ArmGUI()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()