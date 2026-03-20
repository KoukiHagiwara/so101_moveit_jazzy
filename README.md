# 実行方法

## moveitとso-101で同じ動きをさせる
- デジタルツインさせるコマンド
```
$ sudo chmod 666 /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AF7133194-if00

```
```
$ ros2 launch so101_moveit_jazzy demo.launch.py hardware_type:=real

```
## 学習データをもとに推論させる
- それぞれ数字ごとに別のターミナルを開いて実行

- 1.MoveItの起動(実機が壊れないよう安全性をあげるため)
```
$  cd ros2_ws
```
```
$  ros2 launch so101_moveit_jazzy demo.launch.py hardware_type:=real
```

- 2.ros2_controlの切り替え
```
$  ros2 control switch_controllers \
  --deactivate arm_controller gripper_controller \
  --activate ai_position_controller ai_gripper_controller
```
- 3.推論ノード(仮想環境pixi)
```
$  cd lerobot
```
```
$  pixi shell
```
```
$  pixi run python act_inference_node.py
```
- or

```
$  python3 smolvla_inference_node.py
```
- 4.実機の現在位置とAIの命令値をグラフでリアルタイムで確認(必要なら)
```
$  ros2 run plotjuggler plotjuggler
```

## 補足
- plan&executeを廃止し、AIから直接モータの値を読み込むコマンド
- これを実行することでJointTrajectoryControllerからJointGroupPositionControllerに切り替わる
- 元に戻したい時は、--deactivate と --activate の名前を逆にすればOK
```
$  ros2 control switch_controllers \
  --deactivate arm_controller gripper_controller \
  --activate ai_position_controller ai_gripper_controller
```


- pcとシミュレーションだけで確認したい場合(vla_manipulator_simパッケージのもの)
```
$ ros2 launch vla_manipulator_sim spawn_robot_gazebo.launch.py

```

