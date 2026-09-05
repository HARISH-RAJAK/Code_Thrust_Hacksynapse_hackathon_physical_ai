#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tkinter as tk
from tkinter import ttk
from threading import Thread
import math


# Joint configuration with accurate limits from URDF
# Format: (joint_name, lower_limit, upper_limit)
JOINT_CONFIG = [
    ("base_to_Link1", -2.0943, 2.0943),           # ~-120 to 120 degrees
    ("Link1_to_Link2", -1.570796, 1.570796),      # -90 to 90 degrees (-π/2 to π/2)
    ("Link2_to_Link3", -2.35619, 2.35619),        # ~-135 to 135 degrees
    ("Linkt_to_Link5", -1.8849, 1.8849),          # ~-108 to 108 degrees
    ("Link5_to_gripperbase", -1.9198, 1.9198),    # ~-110 to 110 degrees
    ("gripper_finger1_joint", 0.0, 1.25654),      # Gripper
]


class ArmGUIController(Node):
    """ROS2 node for controlling robot arm via GUI sliders"""

    def __init__(self):
        super().__init__('arm_gui_controller_robot1')

        # Publisher for joint trajectories
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/robot1/hiwonder_xarm_controller/joint_trajectory',
            10
        )

        # Store current joint positions
        self.current_positions = [0.0] * len(JOINT_CONFIG)
        self.slider_vars = []
        self.position_labels = []
        
        # Control parameters
        self.trajectory_time = 1.0  # seconds

        # Start GUI in separate thread
        gui_thread = Thread(target=self.start_gui)
        gui_thread.daemon = True
        gui_thread.start()

    def send_trajectory(self):
        """Publish joint trajectory message"""
        msg = JointTrajectory()
        msg.joint_names = [j[0] for j in JOINT_CONFIG]

        point = JointTrajectoryPoint()
        point.positions = self.current_positions.copy()
        point.time_from_start.sec = int(self.trajectory_time)
        point.time_from_start.nanosec = int((self.trajectory_time - int(self.trajectory_time)) * 1e9)

        msg.points.append(point)
        self.publisher.publish(msg)
        self.get_logger().info(f"Published trajectory: {self.current_positions}")

    def update_joint(self, index, value):
        """Update joint position and publish trajectory"""
        try:
            self.current_positions[index] = float(value)
            # Update label with current value
            if index < len(self.position_labels):
                joint_name = JOINT_CONFIG[index][0]
                self.position_labels[index].config(
                    text=f"{joint_name}: {float(value):.4f} rad ({math.degrees(float(value)):.2f}°)"
                )
            self.send_trajectory()
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f"Error updating joint {index}: {e}")

    def reset_positions(self):
        """Reset all joints to zero position"""
        for i, var in enumerate(self.slider_vars):
            var.set(0.0)

    def update_trajectory_time(self, value):
        """Update trajectory execution time"""
        try:
            self.trajectory_time = float(value)
        except ValueError:
            pass

    def start_gui(self):
        """Build and start the Tkinter GUI"""
        root = tk.Tk()
        root.title("HiWonder XArm Controller - /robot1/")
        root.geometry("800x900")

        # Main frame with padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="HiWonder XArm Joint Control",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        # Topic info frame
        info_frame = ttk.LabelFrame(main_frame, text="Publishing Info", padding="5")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        topic_label = ttk.Label(
            info_frame,
            text="Topic: /robot1/hiwonder_xarm_controller/joint_trajectory",
            font=("Arial", 10)
        )
        topic_label.pack()

        msg_type_label = ttk.Label(
            info_frame,
            text="Message Type: trajectory_msgs/JointTrajectory",
            font=("Arial", 10)
        )
        msg_type_label.pack()

        # Trajectory time frame
        time_frame = ttk.Frame(main_frame)
        time_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Label(time_frame, text="Trajectory Time (s):", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        time_var = tk.DoubleVar(value=self.trajectory_time)
        time_spinbox = ttk.Spinbox(
            time_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=time_var,
            width=10,
            command=lambda: self.update_trajectory_time(time_var.get())
        )
        time_spinbox.pack(side=tk.LEFT, padx=5)

        # Joint sliders canvas with scrollbar
        canvas_frame = ttk.LabelFrame(main_frame, text="Joint Sliders", padding="5")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create slider for each joint
        for i, (name, lower, upper) in enumerate(JOINT_CONFIG):
            # Joint frame
            joint_frame = ttk.LabelFrame(scrollable_frame, text=f"{name}", padding="10")
            joint_frame.pack(fill=tk.X, padx=5, pady=5)

            # Position label
            position_label = ttk.Label(
                joint_frame,
                text=f"{name}: 0.0000 rad (0.00°)",
                font=("Arial", 9)
            )
            position_label.pack(anchor=tk.W, pady=3)
            self.position_labels.append(position_label)

            # Slider with variable
            var = tk.DoubleVar(value=0.0)
            self.slider_vars.append(var)

            slider_frame = ttk.Frame(joint_frame)
            slider_frame.pack(fill=tk.X, pady=5)

            # Lower limit label
            ttk.Label(slider_frame, text=f"{lower:.2f}", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)

            slider = ttk.Scale(
                slider_frame,
                from_=lower,
                to=upper,
                variable=var,
                orient=tk.HORIZONTAL,
                command=lambda val, idx=i: self.update_joint(idx, val)
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

            # Upper limit label
            ttk.Label(slider_frame, text=f"{upper:.2f}", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)

            # Min/Max/Home buttons
            button_frame = ttk.Frame(joint_frame)
            button_frame.pack(fill=tk.X, pady=5)

            def create_button_cmd(idx, val):
                return lambda: self.slider_vars[idx].set(val)

            ttk.Button(
                button_frame,
                text="Min",
                width=8,
                command=create_button_cmd(i, lower)
            ).pack(side=tk.LEFT, padx=5)

            ttk.Button(
                button_frame,
                text="Zero",
                width=8,
                command=create_button_cmd(i, 0.0)
            ).pack(side=tk.LEFT, padx=5)

            ttk.Button(
                button_frame,
                text="Max",
                width=8,
                command=create_button_cmd(i, upper)
            ).pack(side=tk.LEFT, padx=5)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(
            control_frame,
            text="Reset All to Zero",
            command=self.reset_positions
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="Send Trajectory",
            command=self.send_trajectory
        ).pack(side=tk.LEFT, padx=5)

        root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = ArmGUIController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
