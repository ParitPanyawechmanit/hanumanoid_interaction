#include <chrono>
#include <thread>
#include <future>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <string>
#include <vector>
#include <utility>
#include <limits>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/exceptions.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit_msgs/msg/joint_constraint.hpp>

#include <shape_msgs/msg/solid_primitive.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

using MoveGroup = moveit_msgs::action::MoveGroup;
using namespace std::chrono_literals;

// ------------------------ small helpers ------------------------

static inline bool is_finite(double v) {
  return std::isfinite(v);
}

static inline std::string trim_copy(std::string s) {
  auto not_space = [](unsigned char c){ return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  // strip quotes if user passes "-z" with quotes
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front()=='\'' && s.back()=='\''))) {
    s = s.substr(1, s.size()-2);
    s = trim_copy(s);
  }
  return s;
}

static inline tf2::Vector3 axis_from_string(std::string s, bool* ok_out=nullptr) {
  s = trim_copy(s);
  // accept: x, +x, -x, y, z, etc. case-insensitive
  for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (ok_out) *ok_out = true;

  if (s == "x" || s == "+x") return tf2::Vector3(1,0,0);
  if (s == "-x") return tf2::Vector3(-1,0,0);
  if (s == "y" || s == "+y") return tf2::Vector3(0,1,0);
  if (s == "-y") return tf2::Vector3(0,-1,0);
  if (s == "z" || s == "+z") return tf2::Vector3(0,0,1);
  if (s == "-z") return tf2::Vector3(0,0,-1);

  if (ok_out) *ok_out = false;
  return tf2::Vector3(0,0,0);
}

static inline geometry_msgs::msg::Quaternion toMsgQuat(const tf2::Quaternion& q_in) {
  tf2::Quaternion q = q_in;
  q.normalize();
  geometry_msgs::msg::Quaternion m;
  m.x = q.x(); m.y = q.y(); m.z = q.z(); m.w = q.w();
  return m;
}

static inline bool quat_ok(const tf2::Quaternion& q) {
  return is_finite(q.x()) && is_finite(q.y()) && is_finite(q.z()) && is_finite(q.w())
      && !(std::abs(q.x()) < 1e-12 && std::abs(q.y()) < 1e-12 && std::abs(q.z()) < 1e-12 && std::abs(q.w()) < 1e-12);
}

// Project v onto plane perpendicular to n (n must be unit)
static inline tf2::Vector3 project_to_plane(const tf2::Vector3& v, const tf2::Vector3& n_unit) {
  return v - n_unit * v.dot(n_unit);
}

// Build a world_R_ee such that:
// - ee forward axis (one of ±X ±Y ±Z) points along dir_world (unit)
// - ee up axis (one of ±X ±Y ±Z, different from forward) is as close as possible to world_up_ref
// Then optionally add twist around forward axis.
static tf2::Quaternion make_lookat_quat(
  const tf2::Vector3& dir_world_unit,
  const tf2::Vector3& world_up_ref_in,
  const tf2::Vector3& ee_forward_axis, // in ee coords: one-hot ±
  const tf2::Vector3& ee_up_axis,      // in ee coords: one-hot ± (must be different)
  double yaw_add_rad,
  bool* ok_out=nullptr)
{
  if (ok_out) *ok_out = false;

  tf2::Vector3 d = dir_world_unit;
  if (d.length2() < 1e-12) return tf2::Quaternion(0,0,0,0);
  d.normalize();

  // Determine which ee basis vector is forward and its sign
  // ex,ey,ez are world directions of ee axes
  bool have_ex=false, have_ey=false, have_ez=false;
  tf2::Vector3 ex, ey, ez;

  auto set_axis = [&](const tf2::Vector3& ee_axis, const tf2::Vector3& world_vec_unit) {
    // ee_axis is one-hot ±
    if (std::abs(ee_axis.x()) > 0.5) { ex = (ee_axis.x() > 0 ? world_vec_unit : -world_vec_unit); have_ex=true; }
    else if (std::abs(ee_axis.y()) > 0.5) { ey = (ee_axis.y() > 0 ? world_vec_unit : -world_vec_unit); have_ey=true; }
    else if (std::abs(ee_axis.z()) > 0.5) { ez = (ee_axis.z() > 0 ? world_vec_unit : -world_vec_unit); have_ez=true; }
  };

  // forward points to target
  set_axis(ee_forward_axis, d);

  // choose world up reference (avoid collinear)
  tf2::Vector3 up_ref = world_up_ref_in;
  if (up_ref.length2() < 1e-12) up_ref = tf2::Vector3(0,0,1);
  up_ref.normalize();

  // if up_ref almost collinear with forward, pick another
  tf2::Vector3 f_world = d;
  double col = std::abs(up_ref.dot(f_world));
  if (col > 0.95) up_ref = tf2::Vector3(0,1,0);

  // Build an "up" candidate that is perpendicular to forward
  tf2::Vector3 up_perp = project_to_plane(up_ref, f_world);
  if (up_perp.length2() < 1e-12) up_perp = project_to_plane(tf2::Vector3(1,0,0), f_world);
  if (up_perp.length2() < 1e-12) return tf2::Quaternion(0,0,0,0);
  up_perp.normalize();

  set_axis(ee_up_axis, up_perp);

  // If forward and up axes accidentally map to the same ee axis, invalid
  if ((std::abs(ee_forward_axis.x()) > 0.5 && std::abs(ee_up_axis.x()) > 0.5) ||
      (std::abs(ee_forward_axis.y()) > 0.5 && std::abs(ee_up_axis.y()) > 0.5) ||
      (std::abs(ee_forward_axis.z()) > 0.5 && std::abs(ee_up_axis.z()) > 0.5)) {
    return tf2::Quaternion(0,0,0,0);
  }

  // Compute missing axis to ensure right-handed frame
  if (have_ex && have_ey && !have_ez) { ez = ex.cross(ey); have_ez=true; }
  if (have_ey && have_ez && !have_ex) { ex = ey.cross(ez); have_ex=true; }
  if (have_ez && have_ex && !have_ey) { ey = ez.cross(ex); have_ey=true; }

  if (!have_ex || !have_ey || !have_ez) return tf2::Quaternion(0,0,0,0);

  // Normalize and orthogonalize lightly
  ex.normalize();
  ey = (ey - ex * ey.dot(ex));
  if (ey.length2() < 1e-12) return tf2::Quaternion(0,0,0,0);
  ey.normalize();
  ez = ex.cross(ey);
  if (ez.length2() < 1e-12) return tf2::Quaternion(0,0,0,0);
  ez.normalize();

  // Build rotation matrix with columns = ee axes in world
  tf2::Matrix3x3 R(
    ex.x(), ey.x(), ez.x(),
    ex.y(), ey.y(), ez.y(),
    ex.z(), ey.z(), ez.z()
  );

  tf2::Quaternion q_base;
  R.getRotation(q_base);
  q_base.normalize();

  // Add yaw twist around forward direction in world (which is d)
  if (std::abs(yaw_add_rad) > 1e-9) {
    tf2::Quaternion q_tw;
    q_tw.setRotation(d, yaw_add_rad);
    q_tw.normalize();
    q_base = q_tw * q_base; // world-axis twist
    q_base.normalize();
  }

  if (ok_out) *ok_out = quat_ok(q_base);
  return q_base;
}

// ------------------------ constraints builders ------------------------

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

static void add_orientation_constraint(
  moveit_msgs::msg::Constraints& c,
  const std::string& frame,
  const std::string& link,
  const tf2::Quaternion& q_world,
  double tol_x, double tol_y, double tol_z,
  double weight = 1.0)
{
  moveit_msgs::msg::OrientationConstraint oc;
  oc.header.frame_id = frame;
  oc.link_name = link;
  oc.orientation = toMsgQuat(q_world);

  oc.absolute_x_axis_tolerance = tol_x;
  oc.absolute_y_axis_tolerance = tol_y;
  oc.absolute_z_axis_tolerance = tol_z;
  oc.weight = weight;

  c.orientation_constraints.push_back(oc);
}

// left/right joints list (for return_to_start)
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
  return {};
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("reach_tf");

  // -------- params --------
  node->declare_parameter<std::string>("group", "left_arm");
  node->declare_parameter<std::string>("target_frame", "Target_object");
  node->declare_parameter<std::string>("planning_frame", "base_link");
  node->declare_parameter<std::string>("eef_link", "l_palm");
  node->declare_parameter<std::string>("move_action", "move_action");

  node->declare_parameter<bool>("offset_in_target_frame", true);
  node->declare_parameter<double>("offset_x", 0.0);
  node->declare_parameter<double>("offset_y", 0.0);
  node->declare_parameter<double>("offset_z", 0.12);

  // offset_z scan
  node->declare_parameter<double>("min_offset_z", 0.02);
  node->declare_parameter<double>("max_offset_z", 0.20);
  node->declare_parameter<double>("offset_step_z", 0.01);

  node->declare_parameter<double>("radius", 0.06);
  node->declare_parameter<double>("planning_time", 10.0);
  node->declare_parameter<int>("planning_attempts", 5);
  node->declare_parameter<bool>("execute", true);
  node->declare_parameter<bool>("return_to_start", false);

  // orientation
  node->declare_parameter<bool>("use_orientation", false);
  node->declare_parameter<std::string>("ori_mode", "fixed"); // fixed | target | lookat
  node->declare_parameter<double>("ori_tol_x", 1.57);
  node->declare_parameter<double>("ori_tol_y", 1.57);
  node->declare_parameter<double>("ori_tol_z", 3.14);

  node->declare_parameter<double>("ori_offset_roll", 0.0);
  node->declare_parameter<double>("ori_offset_pitch", 0.0);
  node->declare_parameter<double>("ori_offset_yaw", 0.0);

  // lookat specific
  node->declare_parameter<std::string>("lookat_forward_axis", "-z");
  node->declare_parameter<std::string>("lookat_up_axis", "+y");
  node->declare_parameter<double>("lookat_yaw_sweep", 0.0);     // total sweep (rad), e.g. pi
  node->declare_parameter<double>("lookat_yaw_step", 0.349066); // 20deg

  // -------- read params --------
  const auto group = node->get_parameter("group").as_string();
  const auto target_frame = node->get_parameter("target_frame").as_string();
  const auto planning_frame = node->get_parameter("planning_frame").as_string();
  std::string eef_link = node->get_parameter("eef_link").as_string();
  const auto move_action_name = node->get_parameter("move_action").as_string();

  const bool offset_in_target_frame = node->get_parameter("offset_in_target_frame").as_bool();
  const double ox = node->get_parameter("offset_x").as_double();
  const double oy = node->get_parameter("offset_y").as_double();
  const double oz_req = node->get_parameter("offset_z").as_double();
  const double min_oz = node->get_parameter("min_offset_z").as_double();
  const double max_oz = node->get_parameter("max_offset_z").as_double();
  const double step_oz = node->get_parameter("offset_step_z").as_double();

  const double radius = node->get_parameter("radius").as_double();
  const double planning_time = node->get_parameter("planning_time").as_double();
  const int planning_attempts = node->get_parameter("planning_attempts").as_int();
  const bool execute = node->get_parameter("execute").as_bool();
  const bool return_to_start = node->get_parameter("return_to_start").as_bool();

  const bool use_orientation = node->get_parameter("use_orientation").as_bool();
  const auto ori_mode = trim_copy(node->get_parameter("ori_mode").as_string());
  const double tol_x = node->get_parameter("ori_tol_x").as_double();
  const double tol_y = node->get_parameter("ori_tol_y").as_double();
  const double tol_z = node->get_parameter("ori_tol_z").as_double();

  const double off_r = node->get_parameter("ori_offset_roll").as_double();
  const double off_p = node->get_parameter("ori_offset_pitch").as_double();
  const double off_y = node->get_parameter("ori_offset_yaw").as_double();

  const auto fwd_axis_str = node->get_parameter("lookat_forward_axis").as_string();
  const auto up_axis_str  = node->get_parameter("lookat_up_axis").as_string();
  const double yaw_sweep = node->get_parameter("lookat_yaw_sweep").as_double();
  const double yaw_step  = node->get_parameter("lookat_yaw_step").as_double();

  if (group == "right_arm" && eef_link == "l_palm") eef_link = "r_palm";

  RCLCPP_INFO(node->get_logger(), "Group: %s", group.c_str());
  RCLCPP_INFO(node->get_logger(), "EEF link: %s", eef_link.c_str());
  RCLCPP_INFO(node->get_logger(), "Target frame: %s", target_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "Requested planning_frame: %s", planning_frame.c_str());
  RCLCPP_INFO(node->get_logger(), "move_action: %s", move_action_name.c_str());
  RCLCPP_INFO(node->get_logger(), "offset_in_target_frame: %s", offset_in_target_frame ? "true" : "false");
  RCLCPP_INFO(node->get_logger(), "use_orientation: %s, ori_mode: %s", use_orientation ? "true" : "false", ori_mode.c_str());

  // -------- executor --------
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);

  // -------- capture start joint state (optional undo) --------
  sensor_msgs::msg::JointState start_js;
  bool got_js = false;

  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [&](sensor_msgs::msg::JointState::SharedPtr msg) {
      if (!got_js) {
        start_js = *msg;
        got_js = true;
      }
    });

  for (int i = 0; i < 100 && rclcpp::ok() && !got_js; i++) {
    exec.spin_some();
    std::this_thread::sleep_for(20ms);
  }
  if (!got_js) {
    RCLCPP_WARN(node->get_logger(), "Did not capture /joint_states quickly (return_to_start may be unavailable).");
  }

  // -------- TF lookup planning_frame -> target_frame --------
  tf2_ros::Buffer tf_buffer(node->get_clock());
  tf2_ros::TransformListener tf_listener(tf_buffer, node, false);

  geometry_msgs::msg::TransformStamped t;
  bool tf_ok = false;
  for (int i = 0; i < 200 && rclcpp::ok(); i++) { // up to ~10s
    try {
      t = tf_buffer.lookupTransform(planning_frame, target_frame, tf2::TimePointZero);
      tf_ok = true;
      break;
    } catch (const tf2::TransformException&) {
      exec.spin_some();
      std::this_thread::sleep_for(50ms);
    }
  }

  if (!tf_ok) {
    RCLCPP_ERROR(node->get_logger(), "TF not ready: %s -> %s", planning_frame.c_str(), target_frame.c_str());
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Using planning_frame: %s", planning_frame.c_str());

  // target position in planning frame
  tf2::Vector3 p_t(
    t.transform.translation.x,
    t.transform.translation.y,
    t.transform.translation.z
  );

  double dist = std::sqrt(p_t.length2());
  RCLCPP_INFO(node->get_logger(), "Target in %s: xyz = [%.3f %.3f %.3f], dist=%.3f m",
              planning_frame.c_str(), p_t.x(), p_t.y(), p_t.z(), dist);

  // target rotation (for offset_in_target_frame or ori_mode=target)
  tf2::Quaternion q_t(
    t.transform.rotation.x,
    t.transform.rotation.y,
    t.transform.rotation.z,
    t.transform.rotation.w
  );
  q_t.normalize();

  // -------- MoveGroup action client --------
  auto client = rclcpp_action::create_client<MoveGroup>(node, move_action_name);
  if (!client->wait_for_action_server(5s)) {
    RCLCPP_ERROR(node->get_logger(), "MoveGroup action server not available: %s", move_action_name.c_str());
    rclcpp::shutdown();
    return 2;
  }

  auto send_goal = [&](const moveit_msgs::msg::Constraints& goal_constraints, bool plan_only) -> std::pair<bool,int> {
    MoveGroup::Goal goal;
    goal.request.group_name = group;
    goal.request.num_planning_attempts = std::max(1, planning_attempts);
    goal.request.allowed_planning_time = planning_time;
    goal.request.max_velocity_scaling_factor = 0.2;
    goal.request.max_acceleration_scaling_factor = 0.2;
    goal.request.goal_constraints.clear();
    goal.request.goal_constraints.push_back(goal_constraints);

    goal.planning_options.plan_only = plan_only;
    goal.planning_options.look_around = false;
    goal.planning_options.replan = false;
    goal.planning_options.planning_scene_diff.is_diff = true;

    auto gh_f = client->async_send_goal(goal);
    if (exec.spin_until_future_complete(gh_f, 10s) != rclcpp::FutureReturnCode::SUCCESS) return {false, -1000};

    auto goal_handle = gh_f.get();
    if (!goal_handle) return {false, -1001};

    auto res_f = client->async_get_result(goal_handle);
    if (exec.spin_until_future_complete(res_f, std::chrono::seconds((int)std::ceil(planning_time) + 10)) != rclcpp::FutureReturnCode::SUCCESS) {
      return {false, -1002};
    }

    auto res = res_f.get();
    if (!res.result) return {false, -1003};

    int code = res.result->error_code.val;
    bool ok = (code == res.result->error_code.SUCCESS);
    RCLCPP_INFO(node->get_logger(), "MoveIt result: %s (code=%d), planning_time=%.3f",
                ok ? "SUCCESS" : "FAILURE", code, res.result->planning_time);
    return {ok, code};
  };

  auto exec_goal = [&](const moveit_msgs::msg::Constraints& c) -> std::pair<bool,int> {
    return send_goal(c, /*plan_only=*/!execute);
  };

  // -------- build candidate offset_z list (center-first, then +/-) --------
  const double oz0 = std::clamp(oz_req, min_oz, max_oz);
  const double dz = (step_oz > 1e-6) ? step_oz : 0.01;

  std::vector<double> oz_list;
  oz_list.reserve(64);
  oz_list.push_back(oz0);
  for (int k = 1; k < 100; k++) {
    double a = oz0 - k*dz;
    double b = oz0 + k*dz;
    bool added = false;
    if (a >= min_oz - 1e-9) { oz_list.push_back(a); added = true; }
    if (b <= max_oz + 1e-9) { oz_list.push_back(b); added = true; }
    if (!added) break;
  }

  // -------- yaw list (0, +step, -step, +2step, -2step...) --------
  std::vector<double> yaw_list;
  yaw_list.reserve(64);
  yaw_list.push_back(0.0);
  if (yaw_sweep > 1e-6 && yaw_step > 1e-6) {
    int N = (int)std::floor((yaw_sweep * 0.5) / yaw_step);
    for (int k = 1; k <= N; k++) {
      yaw_list.push_back(+k * yaw_step);
      yaw_list.push_back(-k * yaw_step);
    }
  }

  // orientation offset quaternion (local ee)
  tf2::Quaternion q_off;
  q_off.setRPY(off_r, off_p, off_y);
  q_off.normalize();

  // parse lookat axes
  bool f_ok=false, u_ok=false;
  tf2::Vector3 ee_fwd = axis_from_string(fwd_axis_str, &f_ok);
  tf2::Vector3 ee_up  = axis_from_string(up_axis_str,  &u_ok);
  if (!f_ok) {
    RCLCPP_WARN(node->get_logger(), "lookat_forward_axis parse failed: '%s' (use -z, +z, +x, etc). Using -z.",
                fwd_axis_str.c_str());
    ee_fwd = tf2::Vector3(0,0,-1);
  }
  if (!u_ok) {
    RCLCPP_WARN(node->get_logger(), "lookat_up_axis parse failed: '%s'. Using +y.", up_axis_str.c_str());
    ee_up = tf2::Vector3(0,1,0);
  }

  // -------- search: offset_z (and yaw if lookat) --------
  bool found = false;
  double chosen_oz = oz0;
  double chosen_yaw = 0.0;
  int last_code = -1;

  for (double oz_try : oz_list) {
    if (!rclcpp::ok()) break;

    // goal position
    tf2::Vector3 goal_pos = p_t;
    if (offset_in_target_frame) {
      // apply offset in target frame using target rotation
      tf2::Matrix3x3 Rt(q_t);
      tf2::Vector3 off_t(ox, oy, oz_try);
      tf2::Vector3 off_w = Rt * off_t;
      goal_pos = p_t + off_w;
    } else {
      goal_pos = p_t + tf2::Vector3(ox, oy, oz_try);
    }

    // Without orientation: just test once
    if (!use_orientation) {
      auto c = make_position_goal(planning_frame, eef_link, goal_pos.x(), goal_pos.y(), goal_pos.z(), radius);
      auto [ok, code] = exec_goal(c);
      last_code = code;
      RCLCPP_INFO(node->get_logger(), "Test offset_z=%.3f -> %s(code=%d)",
                  oz_try, ok ? "SUCCESS" : "FAILURE", code);
      if (ok) { chosen_oz = oz_try; found = true; break; }
      continue;
    }

    // With orientation:
    for (double yaw_add : yaw_list) {
      if (!rclcpp::ok()) break;

      tf2::Quaternion q_des(0,0,0,0);
      bool q_ok = false;

      if (ori_mode == "fixed") {
        // fixed orientation = only offsets (in planning frame)
        q_des.setRPY(0,0,0);
        q_des.normalize();
        q_des = q_des * q_off;
        q_des.normalize();
        q_ok = quat_ok(q_des);
      }
      else if (ori_mode == "target") {
        // use target frame orientation, then apply offsets
        q_des = q_t;
        q_des.normalize();
        q_des = q_des * q_off;
        q_des.normalize();
        q_ok = quat_ok(q_des);
      }
      else if (ori_mode == "lookat") {
        // look direction: from base (planning origin) to target (works even if eef TF isn't published)
        tf2::Vector3 dir = p_t;
        if (dir.length2() < 1e-12) dir = tf2::Vector3(0,0,1);
        dir.normalize();

        tf2::Vector3 world_up(0,0,1);
        q_des = make_lookat_quat(dir, world_up, ee_fwd, ee_up, yaw_add, &q_ok);
        if (q_ok) {
          // apply local rpy offsets (ee frame)
          q_des = q_des * q_off;
          q_des.normalize();
          q_ok = quat_ok(q_des);
        }
      }
      else {
        RCLCPP_WARN(node->get_logger(), "Unknown ori_mode='%s' (use fixed|target|lookat). Treating as fixed.",
                    ori_mode.c_str());
        q_des.setRPY(0,0,0);
        q_des.normalize();
        q_des = q_des * q_off;
        q_des.normalize();
        q_ok = quat_ok(q_des);
      }

      if (!q_ok) {
        RCLCPP_WARN(node->get_logger(), "Skip candidate: invalid quaternion (oz=%.3f yaw_add=%.3f)", oz_try, yaw_add);
        continue;
      }

      auto c = make_position_goal(planning_frame, eef_link, goal_pos.x(), goal_pos.y(), goal_pos.z(), radius);
      add_orientation_constraint(c, planning_frame, eef_link, q_des, tol_x, tol_y, tol_z, 1.0);

      auto [ok, code] = exec_goal(c);
      last_code = code;
      RCLCPP_INFO(node->get_logger(), "Test offset_z=%.3f yaw_add=%.3f -> %s(code=%d)",
                  oz_try, yaw_add, ok ? "SUCCESS" : "FAILURE", code);

      if (ok) {
        chosen_oz = oz_try;
        chosen_yaw = yaw_add;
        found = true;
        break;
      }
    }

    if (found) break;
  }

  if (!found) {
    RCLCPP_ERROR(node->get_logger(), "No valid plan found. Last error code=%d", last_code);
    RCLCPP_ERROR(node->get_logger(),
      "Most common fixes:\n"
      "  - eef_link must be a real LINK in the robot model\n"
      "  - group must include that link as tip (or chain reaches it)\n"
      "  - kinematics.yaml must define IK solver for the group\n"
      "  - if using lookat: try lookat_forward_axis +/-z and yaw sweep\n"
      "  - if MoveIt returns 99999 immediately: quaternion/constraint was invalid");
    rclcpp::shutdown();
    return 5;
  }

  RCLCPP_INFO(node->get_logger(), "Go-to-target: chosen offset_z=%.3f, chosen yaw_add=%.3f",
              chosen_oz, chosen_yaw);

  // -------- optional return-to-start --------
  if (return_to_start && got_js && rclcpp::ok()) {
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

    auto [back_ok, back_code] = exec_goal(jc);
    RCLCPP_INFO(node->get_logger(), "Return-to-start: %s(code=%d)", back_ok ? "SUCCESS" : "FAILURE", back_code);
  }

  rclcpp::shutdown();
  return 0;
}
