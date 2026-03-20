from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch

def generate_launch_description():
    hardware_type = LaunchConfiguration("hardware_type")
    
    moveit_config = (
        MoveItConfigsBuilder("so101_new_calib", package_name="so101_moveit_jazzy")
        .robot_description(mappings={"hardware_type": hardware_type})
        .to_moveit_configs()
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "hardware_type",
            default_value="mock",
            description="'mock' (ダミー), 'real' (実機), or 'gazebo' (シミュレータ)"
        ),
        generate_demo_launch(moveit_config),
    ])