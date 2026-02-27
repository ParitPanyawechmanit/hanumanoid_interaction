# In Terminal Moveit
cd hanumanoid_interaction
source /opt/ros/humble/setup.bash
source ~/hanumanoid_interaction/install/setup.bash
ros2 launch hanumanoid_config demo.launch.py

# In Terminal YOLO
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run realsense_listener yolo_node

# In Terminal COLLISION
source /opt/ros/humble/setup.bash
source ~/hanumanoid_interaction/install/setup.bash
python3 /home/parit/hanumanoid_interaction/src/target_collision_publisher_dynamic.py --ros-args \
  -p planning_frame:=base_link \
  -p target_frame:=Target_object \
  -p object_id:=target_object \
  -p padding:=0.005

# In Terminal for Run Reach_tf.py
cd hanumanoid_interaction
source /opt/ros/humble/setup.bash
source ~/hanumanoid_interaction/install/setup.bash

ros2 run hanumanoid_interaction_description reach_tf_py --ros-args \
  -p use_orientation:=true \
  -p ori_mode:=lookat \
  -p avoid_backhand:=true \
  -p palm_axis:=+x \
  -p lookat_yaw_sweep:=0.0

try this for limit the search space + reduce planning time

ros2 run hanumanoid_interaction_description reach_tf_py --ros-args \
  -p use_orientation:=true \
  -p ori_mode:=lookat \
  -p avoid_backhand:=true \
  -p palm_axis:=+x \
  -p lookat_forward_axis:=+x \
  -p lookat_up_axis:=+z \
  -p planning_time:=3.0 \
  -p planning_attempts:=2 \
  -p min_offset_z:=0.10 -p max_offset_z:=0.15 -p offset_step_z:=0.02 \
  -p lookat_yaw_sweep:=3.14 \
  -p lookat_yaw_step:=0.35 \
  -p ori_tol_x:=0.35 -p ori_tol_y:=0.35 -p ori_tol_z:=0.60

# Config for Isaac sim

-Choosing Tool -> Robotics -> Ros 2 OmniGraphs -> Joint States
<img width="1452" height="947" alt="image" src="https://github.com/user-attachments/assets/85cb403b-359c-4ae5-8494-879ef1796528" />

-Set Articulation Root to "/World/hanumanoid_interaction/base_link"
-Set Publisher Topic to "/isaac_joint_states"
-Set Subscriber Topic to "/isaac_joint_command"
<img width="1452" height="947" alt="image" src="https://github.com/user-attachments/assets/718ee28a-9ea5-4415-bb2a-b1665404ee37" />

# Add Palm TF to URDF

In "/home/parit/hanumanoid_interaction/src/hanumanoid_interaction_description/urdf/hanumanoid_interaction.urdf" 
add r_palm and l_palm to make palm tf

<img width="1905" height="1195" alt="image" src="https://github.com/user-attachments/assets/d77ac7e2-d520-4310-b915-d0c34d36f40e" />







