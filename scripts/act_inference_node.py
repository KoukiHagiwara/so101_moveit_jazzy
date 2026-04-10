import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 
from sensor_msgs.msg import JointState

import cv2
import torch
import numpy as np
import math
import os

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import build_inference_frame

class ActInferenceNode(Node):
    def __init__(self):
        super().__init__('act_inference_node')
       
        self.arm_publisher_ = self.create_publisher(Float64MultiArray, '/ai_position_controller/commands', 10)
        self.gripper_publisher_ = self.create_publisher(Float64MultiArray, '/ai_gripper_controller/commands', 10)

        self.current_joints_deg = [0.0] * 6
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # ROS 2からデータが届いたかどうかのフラグ
        self.valid_joints_received = False

        # 🌟 最新の完璧なオフセット
        self.offsets = [3.83, 8.40, -15.16, 15.87, -1.71, 5.20]

        top_cam_path = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_200901010001-video-index0"
        wrist_cam_path = "/dev/v4l/by-id/usb-4K_USB_Camera_4K_USB_Camera_01.00.00-video-index0"

        self.cap_top = cv2.VideoCapture(top_cam_path)
        self.cap_wrist = cv2.VideoCapture(wrist_cam_path)

        for cap in [self.cap_top, self.cap_wrist]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap_top.isOpened() or not self.cap_wrist.isOpened():
            self.get_logger().error("❌ カメラのオープンに失敗しました。パスを確認してください。")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🌟 APIで成功した完全版モデルのパスに修正
        model_id = "/home/kouki/lerobot/outputs/train/act_slide_pick_final/checkpoints/100000/pretrained_model"
        # 🌟 50エピソード学習に使った本番データセット（ここから正確な正規化Statsを取得します）
        dataset_id = "KoukiHagiwara/slide_and_pick_place_final"
       
        self.get_logger().info(f"🧠 モデルを読み込んでいます: {model_id}")
        self.model = ACTPolicy.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # 🌟 正規化用のメタデータを取得
        self.dataset_metadata = LeRobotDatasetMetadata(dataset_id)
        
        # 🌟 APIコードに合わせて初期化（policy_cfg=self.modelに変更）
        self.preprocess, self.postprocess = make_pre_post_processors(
            policy_cfg=self.model,
            pretrained_path=model_id,
            dataset_stats=self.dataset_metadata.stats
        )

        # 🌟 LeRobot本来の学習速度と同じ 30Hz 駆動
        self.timer = self.create_timer(0.033, self.run_inference)
        self.get_logger().info("==================================================")
        self.get_logger().info("🚀 API完全準拠版 ACT 推論ノード開始！")
        self.get_logger().info("==================================================")

    def joint_callback(self, msg):
        try:
            temp_dict = dict(zip(msg.name, msg.position))
            r2d = 180.0 / math.pi
            # ROS2のラジアンを度に変換し、オフセットを足して「AIが学習した状態」に戻す
            self.current_joints_deg = [
                (temp_dict['shoulder_pan'] * r2d) + self.offsets[0],
                (temp_dict['shoulder_lift'] * r2d) + self.offsets[1],
                (temp_dict['elbow_flex'] * r2d) + self.offsets[2],
                (temp_dict['wrist_flex'] * r2d) + self.offsets[3],
                (temp_dict['wrist_roll'] * r2d) + self.offsets[4],
                (temp_dict['gripper'] * r2d) + self.offsets[5]
            ]
            self.valid_joints_received = True
        except KeyError:
            pass

    def send_command(self, action_array):
        # AIの推論（度数法）をROS 2用（ラジアン＋オフセット抜き）に変換して送信
        current_str = ", ".join([f"{a:>6.1f}" for a in self.current_joints_deg])
        target_str  = ", ".join([f"{a:>6.1f}" for a in action_array])
        self.get_logger().info(f"📊 現在地(度): [{current_str}] \n🎯 AI目標(度): [{target_str}]")
        self.get_logger().info("-" * 40)

        d2r = math.pi / 180.0
        arm_msg = Float64MultiArray()
        arm_msg.data = [
            (action_array[0] - self.offsets[0]) * d2r,
            (action_array[1] - self.offsets[1]) * d2r,
            (action_array[2] - self.offsets[2]) * d2r,
            (action_array[3] - self.offsets[3]) * d2r,
            (action_array[4] - self.offsets[4]) * d2r
        ]
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [(action_array[5] - self.offsets[5]) * d2r]
       
        self.arm_publisher_.publish(arm_msg)
        self.gripper_publisher_.publish(gripper_msg)

    def run_inference(self):
        ret_top, frame_top = self.cap_top.read()
        ret_wrist, frame_wrist = self.cap_wrist.read()
        if not ret_top or not ret_wrist: return

        # ROS 2から正しい現在地が届くまでは推論を実行しない
        if not self.valid_joints_received:
            self.get_logger().info("⏳ ロボットの初期座標を取得中...", throttle_duration_sec=1.0)
            return

        # AIに入力するためのデータ辞書を作成
        obs_dict = {
            "top_camera": cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB), 
            "wrist_camera": cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB),
            "state": np.array(self.current_joints_deg, dtype=np.float32)
        }

        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        for i, name in enumerate(joint_names):
            obs_dict[f"{name}.pos"] = self.current_joints_deg[i]
            obs_dict[f"{name}.vel"] = 0.0
            
        # build_inference_frame で Numpy配列をPyTorchテンソルに正確に変換
        obs_frame = build_inference_frame(
            observation=obs_dict, 
            ds_features=self.dataset_metadata.features, 
            device=self.device
        )
        
        # 正規化（Statsの適用）
        obs = self.preprocess(obs_frame)

        # 推論
        with torch.no_grad():
            action = self.model.select_action(obs)

        # 逆正規化してリストに戻す
        action_list = self.postprocess(action).squeeze().cpu().numpy().tolist()
        self.send_command(action_list)

def main(args=None):
    rclpy.init(args=args)
    node = ActInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap_top.release()
        node.cap_wrist.release()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()