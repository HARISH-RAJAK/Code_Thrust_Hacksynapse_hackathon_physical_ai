#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tkinter as tk
from threading import Thread

JOINT_CONFIG = [
    ("xarm_6_joint", -2.35, 2.35),
    ("xarm_5_joint", -1.62, 1.62),
    ("xarm_4_joint", -2.2, 2.2),
    ("xarm_3_joint", -2.05, 2.35),
    ("xarm_2_joint", -2.35, 2.35),
    ("xarm_1_joint", 0.0, 0.028),
]

class XarmGUI(Node):

    def __init__(self):
        super().__init__('xarm_gui')

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

        msg.points.append(point)

        self.publisher.publish(msg)

    def update_joint(self, index, value):
        self.current_positions[index] = float(value)
        self.send_trajectory()

    def start_gui(self):
        root = tk.Tk()
        root.title("XArm Joint Control")

        for i, (name, lower, upper) in enumerate(JOINT_CONFIG):
            frame = tk.Frame(root)
            frame.pack(padx=10, pady=5)

            label = tk.Label(frame, text=f"{name}")
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
    node = XarmGUI()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()