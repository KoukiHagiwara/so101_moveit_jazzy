import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray  
from sensor_msgs.msg import JointState

import cv2
import torch
import numpy as np
import math

# LeRobotのモジュール
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame

class SmolVlaInferenceNode(Node):
    def __init__(self):
        super().__init__('smolvla_inference_node')
        
        # --- 1. ROS 2 Publisher / Subscriber ---
        self.arm_publisher_ = self.create_publisher(Float64MultiArray, '/ai_position_controller/commands', 10)
        self.gripper_publisher_ = self.create_publisher(Float64MultiArray, '/ai_gripper_controller/commands', 10)

        self.current_joints = [0.0] * 6
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # --- 2. カメラの初期化 ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # --- 3. smolVLAモデルの読み込み ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "outputs/train/smolvla_test_task_05/checkpoints/050000/pretrained_model" 
        dataset_id = "local_workspace/test_task_05"
        
        self.get_logger().info(f"smolVLAモデルを {model_id} から読み込み中...")
        
        self.model = SmolVLAPolicy.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.dataset_metadata = LeRobotDatasetMetadata(dataset_id)
        self.preprocess, self.postprocess = make_pre_post_processors(self.model.config, dataset_stats=self.dataset_metadata.stats)

        # 🌟 VLA特有：言語指示の定義
        self.instruction = "Grab the object."

        # --- 4. 推論ループ用タイマーの作成 (0.05秒 = 20Hz) ---
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.run_inference)

    def joint_callback(self, msg):
        """ROS 2から送られてくる最新の関節角度を保存する"""
        try:
            temp_dict = dict(zip(msg.name, msg.position))
            self.current_joints = [
                temp_dict['shoulder_pan'],
                temp_dict['shoulder_lift'],
                temp_dict['elbow_flex'],
                temp_dict['wrist_flex'],
                temp_dict['wrist_roll'],
                temp_dict['gripper']
            ]
            # 現在の角度が正しく取れているか、たまに表示して確認
            if np.random.rand() < 0.1: 
                self.get_logger().info(f"現在位置: {[round(j, 3) for j in self.current_joints]}")
        except KeyError:
            pass 

    def send_command(self, action_array):
        self.get_logger().info(f"AI出力: {[round(a, 3) for a in action_array]}")
        
        # --- 腕の命令を作成 (度をラジアンに変換) ---
        arm_msg = Float64MultiArray()
        arm_msg.data = [float(p) * (math.pi / 180.0) for p in action_array[:5]]

        # --- グリッパーの命令を作成 (度をラジアンに変換) ---
        gripper_msg = Float64MultiArray()
        gripper_msg.data = [float(action_array[5]) * (math.pi / 180.0)]
        
        # 送信
        self.arm_publisher_.publish(arm_msg)
        self.gripper_publisher_.publish(gripper_msg)

    def run_inference(self):
        # --- 1. カメラ画像取得 ---
        ret, frame = self.cap.read()
        if not ret: return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 2. 状態の配列化 ---
        state_array = np.array(self.current_joints, dtype=np.float32)

        # --- 3. Observationの作成 ---
        obs_dict = {
            "top_camera": frame_rgb,  
            "state": state_array,
            "text": self.instruction
        }

        # 個別の関節データ（.pos）を辞書に追加
        joint_names = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex', 
            'wrist_flex', 'wrist_roll', 'gripper'
        ]
        for i, name in enumerate(joint_names):
            obs_dict[f"{name}.pos"] = self.current_joints[i]
            obs_dict[f"{name}.vel"] = 0.0

        # --- 4. 前処理と推論 ---
        obs_frame = build_inference_frame(
            observation=obs_dict, 
            ds_features=self.dataset_metadata.features, 
            device=self.device
        )
        obs = self.preprocess(obs_frame)

        # 学習時の rename_map を推論用に手動適用
        if 'observation.images.top_camera' in obs:
            obs['observation.images.camera1'] = obs.pop('observation.images.top_camera')

        with torch.no_grad():
            action = self.model.select_action(obs)

        # --- 5. 後処理と送信 ---
        action = self.postprocess(action)
        action_list = action.squeeze().cpu().numpy().tolist()
        if isinstance(action_list[0], list):
            action_list = action_list[0] 

        self.send_command(action_list)

def main(args=None):
    rclpy.init(args=args)
    
    node = SmolVlaInferenceNode()
    node.get_logger().info("🚀 SmolVLA 推論ノードを開始します！(停止は Ctrl+C)")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ユーザー操作により推論を終了します。")
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()