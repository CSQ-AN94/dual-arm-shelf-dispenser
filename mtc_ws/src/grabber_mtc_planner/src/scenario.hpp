// Scenario / arm-config loading and machine-readable result writing for the
// dual RM75 shelf-to-shelf MTC prototype.  Deliberately free of any MTC
// include so the data model can be read (and reused) without MoveIt.
#pragma once

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <Eigen/Geometry>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace grabber_mtc
{

/// One arm's identity.  Everything the planner needs to build a branch, plus
/// the execution gate that keeps an un-calibrated arm out of Gate C.
struct ArmConfig
{
	std::string arm_id;
	std::string planning_group;
	std::string end_effector;   // SRDF end-effector name, "" when the model has none
	std::string ik_link;        // link the IK frame is expressed relative to
	std::string hand_group;     // "" while the gripper is driven by the RealMan SDK
	std::vector<std::string> touch_links;
	std::string home_state;     // SRDF named state used as the safe/compact pose
	std::string tool_id;
	std::string tool_version;
	std::string calibration_version;
	/// Full rigid pose of the TCP in ik_link.  This is a calibration input: do
	/// not collapse a measured tool rotation into a scalar offset.
	Eigen::Isometry3d tcp_transform_from_ik_link{ Eigen::Isometry3d::Identity() };
	bool execution_eligible{ false };
	std::string execution_block_reason;
};

struct BoxObject
{
	std::string id;
	std::array<double, 3> size{ { 0.0, 0.0, 0.0 } };
	geometry_msgs::msg::Pose pose;
};

struct GraspCandidate
{
	std::string id;
	geometry_msgs::msg::Pose pose;
};

struct Scenario
{
	std::string scenario_id;
	std::string frame_id;  // must equal the robot model frame; checked at runtime
	bool pick_only{ false };
	bool place_only{ false };
	std::string planning_arm_id;
	std::string source_layer_id;
	std::string target_layer_id;
	std::string lift_state_id;
	std::string source_support_surface_id;
	std::string target_support_surface_id;

	geometry_msgs::msg::Pose source_grasp_pose;   // desired TCP pose at the grasp
	std::vector<GraspCandidate> source_grasp_candidates;
	geometry_msgs::msg::Pose target_place_pose;   // desired TCP pose at the place

	geometry_msgs::msg::Vector3 source_approach_direction;
	geometry_msgs::msg::Vector3 source_lift_direction;
	geometry_msgs::msg::Vector3 source_retreat_direction;
	geometry_msgs::msg::Vector3 target_insert_direction;
	geometry_msgs::msg::Vector3 target_retreat_direction;

	double source_pregrasp_offset_m{ 0.085 };
	/// Only this final part of the source approach may use bottle↔touch-link ACM.
	double source_contact_distance_m{ 0.020 };
	double source_lift_distance_m{ 0.050 };
	double source_retreat_distance_m{ 0.150 };
	double target_preplace_offset_m{ 0.085 };
	double target_contact_distance_m{ 0.010 };
	double target_retreat_distance_m{ 0.150 };
	/// Cartesian segments are accepted when at least this fraction of the
	/// requested distance is reachable.
	double cartesian_min_fraction{ 0.6 };
	bool cartesian_transport{ false };
	/// Per-joint weights for path cost, base joint first.  Unit weights make a
	/// wrist roll look as expensive as a shoulder sweep, so "shortest" paths
	/// came out visually uncoordinated: the arm would swing its whole shoulder
	/// where a wrist turn would do.  Proximal joints move the most mass and the
	/// most volume, so they cost more here.
	/// TUNING KNOB — raise the leading entries to keep the shoulder stiller.
	std::vector<double> planning_joint_weights{ 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0 };
	/// Optional arm posture the operator actually used to reach this grasp.
	/// A 7-DoF arm reaches one TCP pose from many configurations, and IK plus
	/// RRTConnect pick among them at random: replaying the 2026-08-02 shelf
	/// scene eight times returned the demonstrated posture four times, a
	/// posture 160-210 deg away twice, and nothing twice.  When set, this
	/// biases the grasp IK toward the branch a person actually used, which is
	/// the one known to clear the shelf in practice rather than only on paper.
	/// Empty leaves selection unbiased.
	std::vector<double> source_grasp_reference_joints_deg;
	/// Optional collision-planned right-arm carry pose reached after pick-only
	/// retreat while the bottle remains attached.
	std::vector<double> post_pick_carry_joints_deg;
	/// Collision-planned right-arm home pose after place-only release.
	std::vector<double> post_place_home_joints_deg;

	// Bottle modelled as an upright cylinder at the grasp pose's position.
	std::string bottle_id{ "bottle" };
	double bottle_radius_m{ 0.033 };
	double bottle_height_m{ 0.21 };
	geometry_msgs::msg::Pose bottle_pose;

	/// Extra shelf boxes injected on top of the live PlanningScene.  Leave empty
	/// when the live scene already carries the shelf.
	std::vector<BoxObject> shelf_boxes;
	std::string dynamic_obstacle_id{ "head_rgbd_non_target" };
	double obstacle_voxel_size_m{ 0.065 };
	std::vector<std::array<double, 3>> obstacle_voxels;
	bool spawn_scene_objects{ true };
	BoxObject tcp_path_workspace;
	bool has_tcp_path_workspace{ false };

	std::string start_state_source{ "current_state" };
	std::string scene_version;
	std::string target_captured_at_utc;
	std::string scene_captured_at_utc;
	double freshness_max_age_s{ 45.0 };
	bool fixture_source{ true };

	// Planner knobs.
	std::string planner_id{ "RRTConnectkConfigDefault" };
	std::string local_motion_planner{ "cartesian" };
	double planning_timeout_s{ 5.0 };
	size_t max_ik_solutions{ 8 };
	size_t max_solutions{ 10 };
};

/// Per-arm outcome of one plan-only run.
struct ArmResult
{
	std::string arm_id;
	std::string branch_id;
	std::string grasp_candidate_id;
	bool solved{ false };
	size_t complete_solution_count{ 0 };
	std::string earliest_failure_stage{ "" };
	double best_total_cost{ -1.0 };
	std::string selected_solution_id;
	bool execution_eligible{ false };
	std::string execution_block_reason;
	/// stage name -> best cost among that stage's solutions (-1 when none).
	std::vector<std::pair<std::string, double>> stage_costs;
};

struct RunResult
{
	std::string scenario_id;
	bool solved{ false };
	std::string selected_arm;
	std::string selected_grasp_candidate;
	std::string selected_solution_id;
	std::string mode{ "full_transfer" };
	std::string trajectory_export_path;
	bool execution_eligible{ false };
	std::string execution_block_reason;
	double planning_wall_time_s{ 0.0 };
	std::string robot_model_name;
	std::vector<std::string> planning_group_names;
	std::string scene_version;
	std::string tool_version;
	std::string calibration_version;
	bool fixture_source{ true };
	// The state MTC's stages::CurrentState actually started from, read back out
	// of that stage's own solution after planning.  Recorded so a run can be
	// audited. The global all-zero bit is retained for old result readers; live
	// execution gates use the selected-arm completeness, stamp and age below.
	std::vector<std::pair<std::string, double>> start_state_joints;
	bool start_state_all_zero{ true };
	std::string start_state_selected_arm;
	bool start_state_selected_arm_complete{ false };
	std::int64_t start_state_joint_state_stamp_ns{ 0 };
	double start_state_joint_state_age_s_at_planning{ -1.0 };
	std::vector<ArmResult> arms;
	const Scenario* scenario{ nullptr };
};

struct TrajectoryPoint
{
	double time_from_start_s{ 0.0 };
	std::vector<double> positions_deg;
	std::vector<double> velocities_deg_s;
	std::vector<double> accelerations_deg_s2;
};

struct PhaseBoundary
{
	std::string name;
	std::size_t start_index{ 0 };
	std::size_t end_index{ 0 };
};

struct PickTrajectoryExport
{
	std::string scenario_id;
	std::string arm_id;
	std::string execution_block_reason;
	std::string grasp_candidate_id;
	std::string target_captured_at_utc;
	std::string scene_captured_at_utc;
	double freshness_max_age_s{ 0.0 };
	std::vector<std::string> joint_names;
	std::vector<TrajectoryPoint> points;
	std::vector<PhaseBoundary> phases;
	std::size_t attach_point_index{ 0 };
};

struct PlaceTrajectoryExport
{
	std::string scenario_id;
	std::string arm_id;
	std::string scene_captured_at_utc;
	double freshness_max_age_s{ 0.0 };
	std::vector<std::string> joint_names;
	std::vector<TrajectoryPoint> points;
	std::vector<PhaseBoundary> phases;
	std::size_t release_point_index{ 0 };
};

struct FullTransferTrajectoryExport
{
	std::string scenario_id;
	std::string arm_id;
	std::string grasp_candidate_id;
	std::vector<std::string> joint_names;
	std::vector<TrajectoryPoint> points;
	std::vector<PhaseBoundary> phases;
	std::size_t attach_point_index{ 0 };
	std::size_t release_point_index{ 0 };
};

/// Throws std::runtime_error with a human-readable message on any problem.
Scenario loadScenario(const std::string& path);
std::vector<ArmConfig> loadArmConfigs(const std::string& path);

/// Collision objects for the bottle and the scenario's shelf boxes.
std::vector<moveit_msgs::msg::CollisionObject> buildSceneObjects(const Scenario& scenario);

void writeResultJson(const std::string& path, const RunResult& result);
void writePickTrajectoryJson(const std::string& path, const PickTrajectoryExport& trajectory);
void writePlaceTrajectoryJson(const std::string& path, const PlaceTrajectoryExport& trajectory);
void writeFullTransferTrajectoryJson(const std::string& path,
                                     const FullTransferTrajectoryExport& trajectory);
std::string formatResultSummary(const RunResult& result);

}  // namespace grabber_mtc
