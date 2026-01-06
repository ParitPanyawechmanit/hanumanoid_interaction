#include <chrono>
#include <thread>
#include <future>
#include <algorithm>


#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>

#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

using MoveGroup = moveit_msgs::action::MoveGroup;
using namespace std::chrono_literals;


static moveit_msgs::msg::Constraints make_position_goal(
  const std::string& frame,
  const std::string& link,
  double x, double y, double z,
  double radius_m)
{
  moveit_msgs::msg::Constraints c;

  moveit_msgs::msg::PositionConstraint pc;
  pc.header.frame_id = frame;
  pc.link_name = link;

  shape_msgs::msg::SolidPrimitive prim;
  prim.type = shape_msgs::msg::SolidPrimitive::SPHERE;
  prim.dimensions = {radius_m};

  geometry_msgs::msg::Pose p;
  p.position.x = x;
  p.position.y = y;
  p.position.z = z;
  p.orientation.w = 1.0;

  pc.constraint_region.primitives.push_back(prim);
  pc.constraint_region.primitive_poses.push_back(p);
  pc.weight = 1.0;

  c.position_constraints.push_back(pc);
  return c;
}

static std::vector<std::string> group_joint_names(const std::string& group)
{
  if (group == "left_arm") {
    return {
      "Joint_l_shoulder_pitch",
      "Joint_l_shoulder_roll",
      "Joint_l_shoulder_yaw",
      "Joint_l_elbow_pitch",
      "Joint_l_wrist_yaw",
      "Joint_l_wrist_pitch",
      "Joint_l_wrist_roll"
    };
  }
  if (group == "right_arm") {
    return {
      "Joint_r_shoulder_pitch",
      "Joint_r_shoulder_roll",
      "Joint_r_shoulder_yaw",
      "Joint_r_elbow_pitch",
      "Joint_r_wrist_yaw",
      "Joint_r_wrist_pitch",
      "Joint_r_wrist_roll"
    };
  }
  // fallback: empty
  return {};
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("reach_tf");

  node->declare_parameter<std::string>("group", "left_arm");
  node->declare_parameter<std::string>("target_frame", "Target_object");
  node->declare_parameter<std::string>("planning_frame", "base_link");
  node->declare_parameter<std::string>("eef_link", "l_hand_roll_1");
  node->declare_parameter<double>("offset_x", 0.0);
  node->declare_parameter<double>("offset_y", 0.0);
  node->declare_parameter<double>("offset_z", 0.12);
  // When auto-searching offset_z, try the requested (closest) first, then back off until a plan succeeds.
  node->declare_parameter<double>("min_offset_z", 0.02);
  node->declare_parameter<double>("max_offset_z", 0.20);
  node->declare_parameter<double>("offset_step_z", 0.01);
  node->declare_parameter<double>("radius", 0.02);          // goal tolerance sphere (m)
  node->declare_parameter<double>("planning_time", 5.0);
  node->declare_parameter<bool>("execute", true);
  node->declare_parameter<bool>("return_to_start", false);

  const auto group = node->get_parameter("group").as_string();
  const auto target_frame = node->get_parameter("target_frame").as_string();
  const auto planning_frame = node->get_parameter("planning_frame").as_string();
  auto eef_link = node->get_parameter("eef_link").as_string();

  const double ox = node->get_parameter("offset_x").as_double();
  const double oy = node->get_parameter("offset_y").as_double();
  const double oz = node->get_parameter("offset_z").as_double();
  const double min_oz = node->get_parameter("min_offset_z").as_double();
  const double max_oz = node->get_parameter("max_offset_z").as_double();
  const double step_oz = node->get_parameter("offset_step_z").as_double();
  const double radius = node->get_parameter("radius").as_double();
  const double planning_time = node->get_parameter("planning_time").as_double();
  const bool execute = node->get_parameter("execute").as_bool();
  const bool return_to_start = node->get_parameter("return_to_start").as_bool();

  if (group == "right_arm" && eef_link == "l_hand_roll_1") eef_link = "r_hand_roll_1";

  // Spin thread (TF + subscriptions)
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);


  RCLCPP_INFO(node->get_logger(), "Group: %s", group.c_str());
  RCLCPP_INFO(node->get_logger(), "Planning frame: %s", planning_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "EEF link: %s", eef_link.c_str());
  RCLCPP_INFO(node->get_logger(), "Target frame: %s", target_frame.c_str());

  // Grab one JointState as "start" (for undo)
  sensor_msgs::msg::JointState start_js;
  bool got_js = false;
  std::promise<void> js_promise;
  auto js_future = js_promise.get_future();

  using namespace std::chrono_literals;

    auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [&](sensor_msgs::msg::JointState::SharedPtr msg) {
        if (!got_js) {
        start_js = *msg;
        got_js = true;
        }
    });

    for (int i = 0; i < 100 && !got_js; i++) { // 2s
    exec.spin_some();
    std::this_thread::sleep_for(20ms);
    }

    if (!got_js) {
    RCLCPP_WARN(node->get_logger(), "Did not capture /joint_states quickly (undo may be unavailable).");
    }


  // TF lookup: planning_frame -> Target_object
    tf2_ros::Buffer tf_buffer(node->get_clock());
    tf2_ros::TransformListener tf_listener(tf_buffer, node, false);

    // Wait until TF is available (up to 5 seconds)
    bool tf_ok = false;
    for (int i = 0; i < 100; i++) { // 5s
    if (tf_buffer.canTransform(planning_frame, target_frame, tf2::TimePointZero)) {
        tf_ok = true;
        break;
    }
    exec.spin_some();
    std::this_thread::sleep_for(50ms);
    }

    if (!tf_ok) {
    RCLCPP_ERROR(node->get_logger(), "TF not ready: %s -> %s",
                planning_frame.c_str(), target_frame.c_str());
    rclcpp::shutdown();
    exec.cancel();
    return 1;
    }


  geometry_msgs::msg::TransformStamped t;
  try {
    t = tf_buffer.lookupTransform(planning_frame, target_frame, tf2::TimePointZero);
  } catch (const tf2::TransformException& ex) {
    RCLCPP_ERROR(node->get_logger(), "TF lookup failed: %s", ex.what());
    rclcpp::shutdown();
    exec.cancel();
    return 1;
  }

    const double gx0 = t.transform.translation.x + ox;
    const double gy0 = t.transform.translation.y + oy;
    const double gz0 = t.transform.translation.z;  // offset_z will be added by the search


  // MoveGroup action client
  auto client = rclcpp_action::create_client<MoveGroup>(node, "move_action");
  if (!client->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_ERROR(node->get_logger(), "move_action server not available. Check `ros2 action list`.");
    rclcpp::shutdown();
    exec.cancel();
    return 2;
  }

    auto send_goal = [&](const moveit_msgs::msg::Constraints& goal_constraints, bool plan_only) -> bool {
    MoveGroup::Goal goal;
    goal.request.group_name = group;
    goal.request.num_planning_attempts = 5;
    goal.request.allowed_planning_time = planning_time;
    goal.request.max_velocity_scaling_factor = 0.2;
    goal.request.max_acceleration_scaling_factor = 0.2;
    goal.request.goal_constraints.clear();
    goal.request.goal_constraints.push_back(goal_constraints);

    goal.planning_options.plan_only = plan_only;   // <<< here
    goal.planning_options.look_around = false;
    goal.planning_options.replan = false;
    goal.planning_options.planning_scene_diff.is_diff = true;

    auto gh = client->async_send_goal(goal);
    if (exec.spin_until_future_complete(gh, 10s) != rclcpp::FutureReturnCode::SUCCESS) return false;

    auto goal_handle = gh.get();
    if (!goal_handle) return false;

    auto res_f = client->async_get_result(goal_handle);
    if (exec.spin_until_future_complete(res_f, 30s) != rclcpp::FutureReturnCode::SUCCESS) return false;

    auto res = res_f.get();
    return res.result && (res.result->error_code.val == res.result->error_code.SUCCESS);
    };

        // Convenience wrappers
    auto plan_goal = [&](const moveit_msgs::msg::Constraints& c) -> bool {
        return send_goal(c, /*plan_only=*/true);
    };

    auto exec_goal = [&](const moveit_msgs::msg::Constraints& c) -> bool {
        return send_goal(c, /*plan_only=*/!execute);  // if execute=false -> plan_only=true
    };


  // 1) Go to cube
    // --- Auto-find an offset_z that can plan ---
    // Try the requested offset_z first (closest), then increase until we find one that can plan.
    // This prevents the old behavior that always jumped to 0.20m even if 0.12m worked.
    const double oz_start = std::clamp(oz, min_oz, max_oz);
    double chosen_oz = oz_start;
    bool found = false;

    // Sanity: avoid infinite loop / bad step
    const double dz = (step_oz > 1e-6) ? step_oz : 0.01;

    for (double oz_try = oz_start; oz_try <= max_oz + 1e-9; oz_try += dz) {
      auto goal_try = make_position_goal(planning_frame, eef_link, gx0, gy0, gz0 + oz_try, radius);
      bool ok_plan = plan_goal(goal_try);
      RCLCPP_INFO(node->get_logger(), "Test offset_z=%.3f -> %s", oz_try, ok_plan ? "OK" : "FAIL");
      if (ok_plan) {
        chosen_oz = oz_try;
        found = true;
        break; // first success is the closest (smallest) that works
      }
    }

    if (!found) {
    RCLCPP_ERROR(node->get_logger(), "No valid offset_z found (try larger radius or planning_time).");
    rclcpp::shutdown();
    return 4;
    }

    // Execute the chosen one (or plan only if execute==false)
    auto goal_final = make_position_goal(planning_frame, eef_link, gx0, gy0, gz0 + chosen_oz, radius);
    bool ok = exec_goal(goal_final);
    RCLCPP_INFO(node->get_logger(), "Go-to-target (offset_z=%.3f): %s", chosen_oz, ok ? "SUCCESS" : "FAILED");


  // 2) Optional undo: return to start joint positions
  if (ok && return_to_start && got_js) {
    auto needed = group_joint_names(group);
    moveit_msgs::msg::Constraints jc;

    for (size_t i = 0; i < start_js.name.size(); i++) {
      const auto& name = start_js.name[i];
      if (!needed.empty() && std::find(needed.begin(), needed.end(), name) == needed.end()) continue;

      moveit_msgs::msg::JointConstraint j;
      j.joint_name = name;
      j.position = start_js.position[i];
      j.tolerance_above = 0.02;
      j.tolerance_below = 0.02;
      j.weight = 1.0;
      jc.joint_constraints.push_back(j);
    }

    bool back = exec_goal(jc);
    RCLCPP_INFO(node->get_logger(), "Return-to-start: %s", back ? "SUCCESS" : "FAILED");
  }


  rclcpp::shutdown();
  return ok ? 0 : 4;
}