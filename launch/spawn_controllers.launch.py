from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 状態読み取り（共通）
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster"],
        ),
        # 2. MoveIt用コントローラー（アクティブで起動）
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["arm_controller"],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["gripper_controller"],
        ),
        # 3. AI用コントローラー（非アクティブで待機！）
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["ai_position_controller", "--inactive"],
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["ai_gripper_controller", "--inactive"],
        ),
    ])