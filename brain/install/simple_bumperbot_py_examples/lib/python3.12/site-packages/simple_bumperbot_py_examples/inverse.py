import rclpy
from rclpy.node import Node
import ikpy.chain
import numpy as np
import tempfile

# ================== UPDATED URDF ==================
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
        <origin xyz="0.0 0.0 0.05157"/>
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

    <!-- 🔵 NEW TOOL OFFSET -->
    <joint name="target_joint" type="fixed">
        <parent link="gripper_base_1"/>
        <child link="target_link"/>
        <origin xyz="0 0 0.07" rpy="0 0 0"/>
    </joint>

</robot>
"""


class IKCalculator(Node):

    def __init__(self):
        super().__init__('ik_calculator')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as tmp:
            tmp.write(URDF_CONTENT)
            urdf_path = tmp.name

        # 🔴 Chain now has one extra link
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[False, True, True, True, False, True, True, False, False]
        )

        # =========================
        # 🔵 TARGET (wrt base_link)
        # This will place target_link origin here
        # =========================
        #target_xyz = [-0.21, 0.0, 0.03]
        # target_xyz = [-0.169, 0.100, 0.035]
        target_xyz = [0.13, 0.15, 0.15]


        self.calculate_ik(target_xyz)


    def calculate_ik(self, target_xyz):

        initial_position = [0] * len(self.chain.links)

        ik_result = self.chain.inverse_kinematics(
            target_position=target_xyz,
            initial_position=initial_position
        )

        joint_angles = {
            "base_to_Link1": ik_result[1],
            "Link1_to_Link2": ik_result[2],
            "Link2_to_Link3": ik_result[3],
            "Linkt_to_Link5": ik_result[5],
            "Link5_to_gripperbase": ik_result[6],
        }

        print("\n==============================")
        print("IK RESULT (wrt target_link)")
        print("==============================")
        print(f"Target Position (m): {target_xyz}\n")

        for joint, angle in joint_angles.items():
            print(f"{joint:25s} : {angle:.6f} rad")

        # 🔎 Verify
        fk = self.chain.forward_kinematics(ik_result)
        reached_position = fk[:3, 3]

        print("\nReached target_link position from FK:")
        print(reached_position)

        print("\nPosition Error:")
        print(np.linalg.norm(np.array(target_xyz) - reached_position))


def main(args=None):
    rclpy.init(args=args)
    node = IKCalculator()
    rclpy.shutdown()


if __name__ == '__main__':
    main()