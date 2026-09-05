import rclpy
from rclpy.node import Node
import ikpy.chain
import numpy as np
import tempfile
from scipy.spatial.transform import Rotation as R


# ================== URDF WITH target_link ==================
URDF_CONTENT = """<?xml version="1.0"?>
<robot name="my_robot">
    <link name="base_link"/>
    <link name="Link_1_1"/>
    <link name="Link_2_1"/>
    <link name="Link_3_1"/>
    <link name="Link_4_1"/>
    <link name="Link_5_1"/>
    <link name="tool_link"/>
    <link name="gripper_base_1"/>
    <link name="target_link"/>

    <joint name="base_to_Link1" type="revolute">
        <parent link="base_link"/>
        <child link="Link_1_1"/>
        <origin xyz="0 0 0.05157"/>
        <axis xyz="0 0 1"/>
        <limit upper="2.09" lower="-2.01"/>
    </joint>

    <joint name="Link1_to_Link2" type="revolute">
        <parent link="Link_1_1"/>
        <child link="Link_2_1"/>
        <origin xyz="-0.011 0.019 0.03"/>
        <axis xyz="0 1 0"/>
        <limit upper="1.57" lower="-1.57"/>
    </joint>

    <joint name="Link2_to_Link3" type="revolute">
        <parent link="Link_2_1"/>
        <child link="Link_3_1"/>
        <origin xyz="0 -0.0004 0.096"/>
        <axis xyz="0 1 0"/>
        <limit upper="2.35" lower="-2.35"/>
    </joint>

    <joint name="Link3_4_Bridge" type="fixed">
        <parent link="Link_3_1"/>
        <child link="Link_4_1"/>
        <origin xyz="3.8e-05 -0.019 0.058"/>
    </joint>

    <joint name="Linkt_to_Link5" type="revolute">
        <parent link="Link_4_1"/>
        <child link="Link_5_1"/>
        <origin xyz="3.8e-05 0.019 0.037"/>
        <axis xyz="0 1 0"/>
        <limit upper="1.88" lower="-1.88"/>
    </joint>

    <joint name="Link5_to_gripperbase" type="revolute">
        <parent link="Link_5_1"/>
        <child link="tool_link"/>
        <origin xyz="0.0003 -0.019 0.055" rpy="0 0 1.57"/>
        <axis xyz="0 0 1"/>
        <limit upper="1.91" lower="-1.91"/>
    </joint>

    <joint name="gripper_base_joint" type="fixed">
        <parent link="tool_link"/>
        <child link="gripper_base_1"/>
    </joint>

    <!-- Tool offset -->
    <joint name="target_joint" type="fixed">
        <parent link="gripper_base_1"/>
        <child link="target_link"/>
        <origin xyz="0 0 0.07"/>
    </joint>

</robot>
"""


class IKCalculator(Node):

    def __init__(self):
        super().__init__('ik_calculator')

        # Save URDF to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as tmp:
            tmp.write(URDF_CONTENT)
            urdf_path = tmp.name

        # Build IK chain (last link = target_link)
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[False, True, True, True, False, True, True, False, False]
        )

        # ===============================
        # 🔵 YOUR TARGET POSE
        # ===============================
        target_xyz = [-0.22132, -1.3e-05, 0.040954]

        # Quaternion (x, y, z, w)
        target_quat = [-0.6792, 0.69611, 0.16245, -0.1665]

        self.calculate_ik(target_xyz, target_quat)


    def calculate_ik(self, target_xyz, target_quat):

        # Convert quaternion → rotation
        rotation = R.from_quat(target_quat)

        # Extract Z-axis direction (5 DOF compatible)
        z_axis = rotation.as_matrix()[:, 2]

        initial_position = [0] * len(self.chain.links)

        # Solve IK (position + Z-axis alignment)
        ik_result = self.chain.inverse_kinematics(
            target_position=target_xyz,
            target_orientation=z_axis,
            orientation_mode="Z",
            initial_position=initial_position
        )

        print("\n==============================")
        print("POSE IK RESULT (Radians)")
        print("==============================")
        print(f"Target Position: {target_xyz}")
        print(f"Target Quaternion: {target_quat}\n")

        joint_names = [
            "base_to_Link1",
            "Link1_to_Link2",
            "Link2_to_Link3",
            "Linkt_to_Link5",
            "Link5_to_gripperbase"
        ]

        indices = [1, 2, 3, 5, 6]

        for name, idx in zip(joint_names, indices):
            print(f"{name:25s}: {ik_result[idx]:.6f} rad")

        # Forward kinematics verification
        fk = self.chain.forward_kinematics(ik_result)
        reached_position = fk[:3, 3]

        print("\nReached target_link position (FK):")
        print(reached_position)

        error = np.linalg.norm(np.array(target_xyz) - reached_position)
        print("\nPosition error:", error)


def main(args=None):
    rclpy.init(args=args)
    node = IKCalculator()
    rclpy.shutdown()


if __name__ == '__main__':
    main()