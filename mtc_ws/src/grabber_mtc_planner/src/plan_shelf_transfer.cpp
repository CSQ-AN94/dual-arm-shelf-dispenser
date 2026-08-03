// RM75 bottle task planned with MoveIt Task Constructor.
//
// PLAN ONLY.  This node never calls Task::execute(), never publishes a
// FollowJointTrajectory goal, and never links against the RealMan SDK.
// Full-transfer remains the dual-arm default. Pick-only plans the configured
// right-arm grasp candidate from live CurrentState through source retreat.

#include "scenario.hpp"

#include <Eigen/SVD>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <moveit/task_constructor/container.h>
#include <moveit/task_constructor/cost_terms.h>
#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/compute_ik.h>
#include <moveit/task_constructor/stages/connect.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/generate_pose.h>
#include <moveit/task_constructor/stages/modify_planning_scene.h>
#include <moveit/task_constructor/stages/move_relative.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <moveit_task_constructor_msgs/msg/solution.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mtc = moveit::task_constructor;
using grabber_mtc::ArmConfig;
using grabber_mtc::ArmResult;
using grabber_mtc::RunResult;
using grabber_mtc::Scenario;

namespace
{

const rclcpp::Logger LOGGER = rclcpp::get_logger("grabber_mtc_planner");
constexpr double EXECUTION_J4_SINGULARITY_DEG = 8.0;
constexpr double EXECUTION_MAX_JOINT_RANGE_DEG = 300.0;
constexpr double EXECUTION_DENSE_FK_JOINT_STEP_DEG = 1.5;
constexpr std::size_t EXECUTION_CONTROLLER_MAX_COMMANDS = 30;
// Live feedback differs from the sampled CurrentState by a few thousandths of
// a degree.  That can split the first compressed chord, so never select a plan
// that already consumes all 30 controller slots.
constexpr std::size_t EXECUTION_PLANNER_MAX_COMMANDS =
    EXECUTION_CONTROLLER_MAX_COMMANDS - 1;
constexpr double EXECUTION_CONTROLLER_MAX_STEP_DEG = 15.0;
constexpr double EXECUTION_CONTROLLER_MAX_ERROR_DEG = 0.02;
// Scale translational Jacobian rows by a representative RM75 reach so all six
// rows are dimensionless before the SVD.  The independent J4 band remains the
// calibrated elbow-specific guard; this catches the other singular families.
constexpr double EXECUTION_JACOBIAN_LENGTH_SCALE_M = 0.5;
constexpr double EXECUTION_MIN_JACOBIAN_SINGULAR_VALUE = 0.02;
constexpr double EXECUTION_MAX_JACOBIAN_CONDITION_NUMBER = 100.0;
constexpr double EXECUTION_MAX_FINGER_ROLL_DEG = 0.25;

struct Options
{
	std::string scenario_path;
	std::string arms_path;
	std::string result_path{ "mtc_plan_result.json" };
	double hold_seconds{ 0.0 };
	bool plan_only{ false };
};

Options parseArgs(const std::vector<std::string>& args)
{
	Options opt;
	for (std::size_t i = 1; i < args.size(); ++i)
	{
		const std::string& a = args[i];
		auto next = [&](const char* what) {
			if (i + 1 >= args.size())
				throw std::runtime_error(std::string("missing value for ") + what);
			return args[++i];
		};
		if (a == "--scenario")
			opt.scenario_path = next("--scenario");
		else if (a == "--arms")
			opt.arms_path = next("--arms");
		else if (a == "--out")
			opt.result_path = next("--out");
		else if (a == "--hold-seconds")
			opt.hold_seconds = std::stod(next("--hold-seconds"));
		else if (a == "--plan-only")
			opt.plan_only = true;
		else
			throw std::runtime_error("unknown argument: " + a);
	}
	if (opt.scenario_path.empty())
		throw std::runtime_error("--scenario <file.yaml> is required");
	// The flag is mandatory and there is no other mode: it exists so a reader of
	// the command line can see that no motion is possible.
	if (!opt.plan_only)
		throw std::runtime_error("--plan-only is required; this package cannot execute motion");
	if (opt.arms_path.empty())
		opt.arms_path = ament_index_cpp::get_package_share_directory("grabber_mtc_planner") +
		                "/config/dual_rm75_arms.yaml";
	return opt;
}

Eigen::Isometry3d tcpFrame(const ArmConfig& arm)
{
	return arm.tcp_transform_from_ik_link;
}

Eigen::Isometry3d poseTransform(const geometry_msgs::msg::Pose& pose)
{
	Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
	transform.translation() =
	    Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
	transform.linear() =
	    Eigen::Quaterniond(
	        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
	        .normalized()
	        .toRotationMatrix();
	return transform;
}

/// Scenario weights keyed by the group's own joint names, base joint first.
/// Used for both the planner's cost term and the post-export ranking so the two
/// layers prefer the same thing instead of pulling against each other.
std::map<std::string, double> jointCostWeights(const Scenario& s,
                                               const moveit::core::RobotModel& model,
                                               const ArmConfig& arm)
{
	const auto* group = model.getJointModelGroup(arm.planning_group);
	if (!group)
		throw std::runtime_error("joint weights need planning group '" + arm.planning_group + "'");
	const auto& active = group->getActiveJointModels();
	if (active.size() != s.planning_joint_weights.size())
		throw std::runtime_error("planning_joint_weights size does not match group '" +
		                         arm.planning_group + "'");
	std::map<std::string, double> weights;
	for (std::size_t i = 0; i < active.size(); ++i)
		weights.emplace(active[i]->getName(), s.planning_joint_weights[i]);
	return weights;
}

/// Cost that prefers the arm posture a person actually used for this grasp.
///
/// A 7-DoF arm reaches one TCP pose from many joint configurations.  IK returns
/// several and RRTConnect explores them in whatever order costs allow, so with
/// nothing to separate them the choice is effectively random: replaying the
/// 2026-08-02 shelf scene eight times gave the demonstrated posture four times,
/// a posture 160-210 deg away twice, and no solution twice.  All of those pass
/// every geometric gate -- they reach the same grasp -- but only the
/// demonstrated one is known to work on the real shelf.
///
/// Weighted by the same table as the path-length term so the two agree about
/// which joints are expensive to move.
std::unique_ptr<mtc::LambdaCostTerm> referencePostureCost(
    const std::vector<double>& reference_deg, const std::string& group,
    const std::map<std::string, double>& weights)
{
	return std::make_unique<mtc::LambdaCostTerm>(
	    [reference_deg, group, weights](const mtc::SubTrajectory& solution,
	                                    std::string& comment) {
		    const auto& state = solution.end()->scene()->getCurrentState();
		    const auto* jmg = state.getRobotModel()->getJointModelGroup(group);
		    if (!jmg)
			    return 0.0;
		    const auto& active = jmg->getActiveJointModels();
		    double cost = 0.0;
		    for (std::size_t i = 0; i < active.size() && i < reference_deg.size(); ++i)
		    {
			    const std::string& name = active[i]->getName();
			    const double* position = state.getJointPositions(name);
			    if (!position)
				    continue;
			    const auto weight = weights.find(name);
			    cost += (weight == weights.end() ? 1.0 : weight->second) *
			            std::abs(*position - reference_deg[i] * M_PI / 180.0);
		    }
		    comment = "demonstrated-posture distance";
		    return cost;
	    });
}

std::vector<std::string> activeJointNames(const moveit::core::RobotModel& model,
                                          const ArmConfig& arm)
{
	const auto* group = model.getJointModelGroup(arm.planning_group);
	if (!group)
		throw std::runtime_error("joint export needs planning group '" +
		                         arm.planning_group + "'");
	std::vector<std::string> names;
	for (const auto* joint : group->getActiveJointModels())
	{
		if (joint->getVariableCount() != 1)
			throw std::runtime_error("pick-only export only supports single-variable arm joints");
		names.push_back(joint->getName());
	}
	if (names.size() != 7)
		throw std::runtime_error("pick-only export requires exactly seven active arm joints");
	return names;
}

constexpr char TCP_WORKSPACE_SHELL_ID[] = "tcp_path_workspace_shell";
/// Slabs are deliberately thick.  A thin wall is tunnellable: MoveIt validates
/// a motion at discrete interpolation steps, so a fast segment can step clean
/// over a few-centimetre obstacle without ever sampling inside it.
constexpr double TCP_WORKSPACE_SHELL_THICKNESS_M = 0.5;
/// How far a wrist link may sit outside the TCP's own bound before the shell
/// should still tolerate it.  Added to the measured ik_link->TCP distance.
constexpr double TCP_WORKSPACE_SHELL_LINK_REACH_M = 0.15;

/// The same volume the post-export audit checks, expressed as geometry the
/// sampler already understands.  A TCP *pose* constraint would force OMPL into
/// projection sampling (measured: 80 s, no solution); collision geometry is
/// checked natively at full speed.
///
/// Deliberately LOOSER than the audit.  Collision checking acts on links, but
/// the audit bounds the TCP point, and `r_hand` reaches past the TCP — placing
/// the walls exactly on the box would reject paths the audit accepts. Offset by
/// the tool reach so the shell can only ever prune excursions far worse than
/// the audit threshold (the observed runaway solutions left it by 300-600 mm,
/// the marginal ones by 2-9 mm). The audit stays authoritative.
moveit_msgs::msg::CollisionObject buildTcpWorkspaceShell(const Scenario& s, const ArmConfig& arm)
{
	const double margin = arm.tcp_transform_from_ik_link.translation().norm() +
	                      TCP_WORKSPACE_SHELL_LINK_REACH_M;
	const auto& pose = s.tcp_path_workspace.pose;
	const Eigen::Quaterniond rotation(pose.orientation.w, pose.orientation.x,
	                                  pose.orientation.y, pose.orientation.z);
	if (rotation.angularDistance(Eigen::Quaterniond::Identity()) > 1e-6)
		throw std::runtime_error(
		    "tcp_path_workspace must stay axis-aligned in the scenario frame to be "
		    "expressed as a collision shell");
	moveit_msgs::msg::CollisionObject shell;
	shell.id = TCP_WORKSPACE_SHELL_ID;
	shell.header.frame_id = s.frame_id;
	shell.operation = moveit_msgs::msg::CollisionObject::ADD;
	const double t = TCP_WORKSPACE_SHELL_THICKNESS_M;
	const std::array<double, 3> centre{ pose.position.x, pose.position.y, pose.position.z };
	const std::array<double, 3> half{ s.tcp_path_workspace.size[0] / 2.0 + margin,
		                              s.tcp_path_workspace.size[1] / 2.0 + margin,
		                              s.tcp_path_workspace.size[2] / 2.0 + margin };
	for (std::size_t axis = 0; axis < 3; ++axis)
		for (const int side : { -1, 1 })
		{
			shape_msgs::msg::SolidPrimitive prim;
			prim.type = shape_msgs::msg::SolidPrimitive::BOX;
			prim.dimensions.resize(3);
			geometry_msgs::msg::Pose slab;
			slab.orientation.w = 1.0;
			for (std::size_t other = 0; other < 3; ++other)
			{
				// Every slab overhangs its neighbours by one thickness so the
				// eight corners stay sealed instead of leaving diagonal gaps.
				prim.dimensions[other] =
				    other == axis ? t : 2.0 * half[other] + 2.0 * t;
			}
			const double offset = half[axis] + t / 2.0;
			const std::array<double, 3> position{
				centre[0] + (axis == 0 ? side * offset : 0.0),
				centre[1] + (axis == 1 ? side * offset : 0.0),
				centre[2] + (axis == 2 ? side * offset : 0.0)
			};
			slab.position.x = position[0];
			slab.position.y = position[1];
			slab.position.z = position[2];
			shell.primitives.push_back(std::move(prim));
			shell.primitive_poses.push_back(std::move(slab));
		}
	return shell;
}

/// One state's TCP in the workspace box's own frame, so callers compare against
/// half-size directly.  Takes its own copy because FK may need an update.
Eigen::Vector3d tcpInWorkspaceFrame(const Scenario& s, const ArmConfig& arm,
                                    const moveit::core::RobotState& reference)
{
	moveit::core::RobotState state(reference);
	state.update();
	Eigen::Isometry3d model_from_frame = Eigen::Isometry3d::Identity();
	if (s.frame_id != state.getRobotModel()->getModelFrame())
		model_from_frame = state.getGlobalLinkTransform(s.frame_id);
	const Eigen::Isometry3d workspace_from_model =
	    (model_from_frame * poseTransform(s.tcp_path_workspace.pose)).inverse();
	return workspace_from_model *
	       (state.getGlobalLinkTransform(arm.ik_link) * arm.tcp_transform_from_ik_link)
	           .translation();
}

struct ExecutionTrajectoryAudit
{
	bool controller_safe{ true };
	bool joint_bounds_safe{ true };
	bool timing_safe{ true };
	bool jacobian_safe{ true };
	std::size_t maximum_controller_commands{ 0 };
	std::size_t first_bad_point{ 0 };
	double minimum_jacobian_singular_value{ std::numeric_limits<double>::infinity() };
	double maximum_jacobian_condition_number{ 0.0 };

	bool safe() const
	{
		return controller_safe && joint_bounds_safe && timing_safe && jacobian_safe;
	}
};

double maximumJointDelta(const std::vector<double>& a, const std::vector<double>& b)
{
	if (a.size() != 7 || b.size() != 7)
		return std::numeric_limits<double>::infinity();
	double maximum = 0.0;
	for (std::size_t joint = 0; joint < 7; ++joint)
	{
		if (!std::isfinite(a[joint]) || !std::isfinite(b[joint]))
			return std::numeric_limits<double>::infinity();
		maximum = std::max(maximum, std::abs(a[joint] - b[joint]));
	}
	return maximum;
}

/// Mirror RobotSession::_compress_connected_joint_path exactly.  MTC is not
/// allowed to call a candidate successful if either side of a gripper event
/// cannot fit into the controller's real connected-command queue.
bool controllerTrajectorySafe(
    const std::vector<grabber_mtc::TrajectoryPoint>& points,
    const std::size_t begin, const std::size_t end,
    std::vector<std::vector<double>>* commands)
{
	commands->clear();
	if (points.empty() || begin >= points.size() || end >= points.size() || begin > end)
		return false;
	std::vector<std::vector<double>> path{ points[begin].positions_deg };
	std::vector<double> last = path.front();
	for (std::size_t index = begin; index <= end; ++index)
	{
		const auto& candidate = points[index].positions_deg;
		if (!std::isfinite(maximumJointDelta(last, candidate)))
			return false;
		if (maximumJointDelta(last, candidate) <= 1e-9)
			continue;
		path.push_back(candidate);
		last = candidate;
	}
	if (path.size() == 1)
		return true;
	for (std::size_t index = 1; index < path.size(); ++index)
		if (maximumJointDelta(path[index - 1], path[index]) >
		    EXECUTION_CONTROLLER_MAX_STEP_DEG)
			return false;

	std::size_t anchor = 0;
	while (anchor + 1 < path.size())
	{
		std::size_t farthest = anchor + 1;
		for (std::size_t candidate_end = anchor + 2; candidate_end < path.size();
		     ++candidate_end)
		{
			std::array<double, 7> direction{};
			double length_squared = 0.0;
			for (std::size_t joint = 0; joint < 7; ++joint)
			{
				direction[joint] = path[candidate_end][joint] - path[anchor][joint];
				length_squared += direction[joint] * direction[joint];
			}
			if (length_squared <= 1e-12 ||
			    maximumJointDelta(path[anchor], path[candidate_end]) >
			        EXECUTION_CONTROLLER_MAX_STEP_DEG)
				break;
			double previous_fraction = -1.0;
			bool fits = true;
			for (std::size_t middle = anchor + 1; middle < candidate_end; ++middle)
			{
				double projection = 0.0;
				for (std::size_t joint = 0; joint < 7; ++joint)
					projection +=
					    (path[middle][joint] - path[anchor][joint]) * direction[joint];
				const double fraction = projection / length_squared;
				double maximum_error = 0.0;
				for (std::size_t joint = 0; joint < 7; ++joint)
					maximum_error = std::max(
					    maximum_error,
					    std::abs(path[middle][joint] -
					             (path[anchor][joint] + fraction * direction[joint])));
				if (fraction < 0.0 || fraction > 1.0 || fraction < previous_fraction ||
				    maximum_error > EXECUTION_CONTROLLER_MAX_ERROR_DEG)
				{
					fits = false;
					break;
				}
				previous_fraction = fraction;
			}
			if (!fits)
				break;
			farthest = candidate_end;
		}
		commands->push_back(path[farthest]);
		anchor = farthest;
	}
	return commands->size() <= EXECUTION_PLANNER_MAX_COMMANDS;
}

ExecutionTrajectoryAudit auditExecutionTrajectory(
    const ArmConfig& arm, const moveit::core::RobotState& start_state,
    const std::vector<std::string>& joint_names,
    const std::vector<grabber_mtc::TrajectoryPoint>& points,
    const std::vector<std::size_t>& gripper_event_indices)
{
	ExecutionTrajectoryAudit audit;
	if (joint_names.size() != 7 || points.empty())
	{
		audit.controller_safe = false;
		audit.joint_bounds_safe = false;
		audit.timing_safe = false;
		audit.jacobian_safe = false;
		return audit;
	}
	const auto model = start_state.getRobotModel();
	const auto* group = model->getJointModelGroup(arm.planning_group);
	const auto* link = model->getLinkModel(arm.ik_link);
	if (!group || !link)
	{
		audit.joint_bounds_safe = false;
		audit.jacobian_safe = false;
		return audit;
	}

	for (std::size_t index = 0; index < points.size(); ++index)
	{
		const auto& point = points[index];
		if (point.positions_deg.size() != 7 || point.velocities_deg_s.size() != 7 ||
		    point.accelerations_deg_s2.size() != 7 ||
		    !std::isfinite(point.time_from_start_s) ||
		    (index > 0 && point.time_from_start_s <= points[index - 1].time_from_start_s))
		{
			audit.timing_safe = false;
			audit.first_bad_point = index;
			break;
		}
		for (std::size_t joint = 0; joint < 7; ++joint)
		{
			const double velocity = point.velocities_deg_s[joint] * M_PI / 180.0;
			const double acceleration = point.accelerations_deg_s2[joint] * M_PI / 180.0;
			const auto& bounds = model->getVariableBounds(joint_names[joint]);
			if (!std::isfinite(velocity) || !std::isfinite(acceleration) ||
			    (bounds.velocity_bounded_ &&
			     std::abs(velocity) > bounds.max_velocity_ + 1e-6) ||
			    (bounds.acceleration_bounded_ &&
			     std::abs(acceleration) > bounds.max_acceleration_ + 1e-6))
			{
				audit.timing_safe = false;
				audit.first_bad_point = index;
				break;
			}
		}
		if (!audit.timing_safe)
			break;
	}

	std::vector<std::size_t> boundaries{ 0 };
	for (const std::size_t event : gripper_event_indices)
		if (event > boundaries.back() && event < points.size())
			boundaries.push_back(event);
	if (boundaries.back() != points.size() - 1)
		boundaries.push_back(points.size() - 1);

	moveit::core::RobotState state(start_state);
	for (std::size_t segment = 0; segment + 1 < boundaries.size(); ++segment)
	{
		std::vector<std::vector<double>> commands;
		if (!controllerTrajectorySafe(points, boundaries[segment], boundaries[segment + 1],
		                              &commands))
		{
			audit.controller_safe = false;
			continue;
		}
		audit.maximum_controller_commands =
		    std::max(audit.maximum_controller_commands, commands.size());
		std::vector<double> previous = points[boundaries[segment]].positions_deg;
		for (const auto& command : commands)
		{
			const std::size_t steps = std::max<std::size_t>(
			    1, static_cast<std::size_t>(std::ceil(
			           maximumJointDelta(previous, command) /
			           EXECUTION_DENSE_FK_JOINT_STEP_DEG)));
			for (std::size_t step = 0; step <= steps; ++step)
			{
				const double fraction = static_cast<double>(step) / steps;
				for (std::size_t joint = 0; joint < 7; ++joint)
				{
					const double position_deg =
					    previous[joint] + fraction * (command[joint] - previous[joint]);
					const double position = position_deg * M_PI / 180.0;
					const auto& bounds = model->getVariableBounds(joint_names[joint]);
					if (!std::isfinite(position) ||
					    (bounds.position_bounded_ &&
					     (position < bounds.min_position_ - 1e-9 ||
					      position > bounds.max_position_ + 1e-9)))
						audit.joint_bounds_safe = false;
					state.setVariablePosition(joint_names[joint], position);
				}
				state.update();
				Eigen::MatrixXd jacobian;
				state.getJacobian(group, link,
				                  arm.tcp_transform_from_ik_link.translation(), jacobian);
				if (jacobian.rows() != 6 || jacobian.cols() != 7 ||
				    !jacobian.allFinite())
				{
					audit.jacobian_safe = false;
					continue;
				}
				jacobian.topRows(3) /= EXECUTION_JACOBIAN_LENGTH_SCALE_M;
				const Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian);
				const auto singular = svd.singularValues();
				const double minimum = singular.minCoeff();
				const double maximum = singular.maxCoeff();
				const double condition =
				    minimum > 0.0 ? maximum / minimum :
				                    std::numeric_limits<double>::infinity();
				audit.minimum_jacobian_singular_value =
				    std::min(audit.minimum_jacobian_singular_value, minimum);
				audit.maximum_jacobian_condition_number =
				    std::max(audit.maximum_jacobian_condition_number, condition);
				if (!std::isfinite(minimum) || !std::isfinite(condition) ||
				    minimum < EXECUTION_MIN_JACOBIAN_SINGULAR_VALUE ||
				    condition > EXECUTION_MAX_JACOBIAN_CONDITION_NUMBER)
					audit.jacobian_safe = false;
			}
			previous = command;
		}
	}
	return audit;
}

double authoredFingerRollDeg(const geometry_msgs::msg::Pose& pose)
{
	const Eigen::Matrix3d rotation = poseTransform(pose).linear();
	return std::asin(std::clamp(std::abs(rotation(2, 1)), 0.0, 1.0)) * 180.0 / M_PI;
}

double plannedFingerRollDeg(
    const Scenario& scenario, const ArmConfig& arm,
    const moveit::core::RobotState& start_state,
    const std::vector<std::string>& joint_names,
    const grabber_mtc::TrajectoryPoint& point)
{
	if (joint_names.size() != 7 || point.positions_deg.size() != 7)
		return std::numeric_limits<double>::infinity();
	moveit::core::RobotState state(start_state);
	for (std::size_t joint = 0; joint < 7; ++joint)
		state.setVariablePosition(joint_names[joint],
		                          point.positions_deg[joint] * M_PI / 180.0);
	state.update();
	Eigen::Isometry3d model_from_frame = Eigen::Isometry3d::Identity();
	if (scenario.frame_id != state.getRobotModel()->getModelFrame())
		model_from_frame = state.getGlobalLinkTransform(scenario.frame_id);
	const Eigen::Isometry3d tcp_in_frame =
	    model_from_frame.inverse() * state.getGlobalLinkTransform(arm.ik_link) *
	    arm.tcp_transform_from_ik_link;
	return std::asin(std::clamp(std::abs(tcp_in_frame.linear()(2, 1)), 0.0, 1.0)) *
	       180.0 / M_PI;
}

bool tcpWorkspaceContainsTrajectory(
    const Scenario& s, const ArmConfig& arm,
    const moveit::core::RobotState& start_state,
    const grabber_mtc::PickTrajectoryExport& trajectory,
    std::size_t* first_bad_segment, Eigen::Vector3d* first_bad_tcp)
{
	moveit::core::RobotState state(start_state);
	Eigen::Isometry3d model_from_frame = Eigen::Isometry3d::Identity();
	if (s.frame_id != state.getRobotModel()->getModelFrame())
		model_from_frame = state.getGlobalLinkTransform(s.frame_id);
	const Eigen::Isometry3d workspace_from_model =
	    (model_from_frame * poseTransform(s.tcp_path_workspace.pose)).inverse();
	const Eigen::Vector3d half_size(
	    s.tcp_path_workspace.size[0] / 2.0,
	    s.tcp_path_workspace.size[1] / 2.0,
	    s.tcp_path_workspace.size[2] / 2.0);

	for (std::size_t segment = 0; segment + 1 < trajectory.points.size(); ++segment)
	{
		const auto& from = trajectory.points[segment].positions_deg;
		const auto& to = trajectory.points[segment + 1].positions_deg;
		double maximum_delta = 0.0;
		for (std::size_t joint = 0; joint < trajectory.joint_names.size(); ++joint)
			maximum_delta = std::max(maximum_delta, std::abs(to[joint] - from[joint]));
		const std::size_t steps = std::max<std::size_t>(
		    1, static_cast<std::size_t>(
		           std::ceil(maximum_delta / EXECUTION_DENSE_FK_JOINT_STEP_DEG)));
		for (std::size_t step = 0; step <= steps; ++step)
		{
			const double fraction = static_cast<double>(step) / static_cast<double>(steps);
			for (std::size_t joint = 0; joint < trajectory.joint_names.size(); ++joint)
				state.setVariablePosition(
				    trajectory.joint_names[joint],
				    (from[joint] + fraction * (to[joint] - from[joint])) * M_PI / 180.0);
			state.update();
			const Eigen::Vector3d tcp =
			    workspace_from_model *
			    (state.getGlobalLinkTransform(arm.ik_link) *
			     arm.tcp_transform_from_ik_link)
			        .translation();
			if ((tcp.array().abs() > half_size.array() + 1e-9).any())
			{
				*first_bad_segment = segment;
				*first_bad_tcp = tcp;
				return false;
			}
		}
	}
	return true;
}

geometry_msgs::msg::Vector3Stamped stamped(const geometry_msgs::msg::Vector3& v, const std::string& frame)
{
	geometry_msgs::msg::Vector3Stamped out;
	out.header.frame_id = frame;
	out.vector = v;
	return out;
}

geometry_msgs::msg::PoseStamped stamped(const geometry_msgs::msg::Pose& p, const std::string& frame)
{
	geometry_msgs::msg::PoseStamped out;
	out.header.frame_id = frame;
	out.pose = p;
	return out;
}

std::unique_ptr<mtc::SerialContainer> buildPlaceBranch(
    const Scenario& s, const ArmConfig& arm, const std::string& branch_id,
    const mtc::solvers::PlannerInterfacePtr& sampling,
    const mtc::solvers::PlannerInterfacePtr& cartesian,
    mtc::SerialContainer** raw)
{
	const std::string p = branch_id + "/";
	const Eigen::Isometry3d ik_frame = tcpFrame(arm);
	auto branch = std::make_unique<mtc::SerialContainer>(branch_id + "_branch");
	branch->setProperty("group", arm.planning_group);
	if (!arm.end_effector.empty())
		branch->setProperty("eef", arm.end_effector);

	mtc::Stage* attached_scene = nullptr;
	{
		auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "attach_held_bottle");
		stage->attachObject(s.bottle_id, arm.ik_link);
		stage->setCallback(
		    [object_id = s.bottle_id, touch_links = arm.touch_links](
		        const planning_scene::PlanningScenePtr& scene, const mtc::PropertyMap&) {
			    moveit_msgs::msg::AttachedCollisionObject attached;
			    if (!scene->getAttachedCollisionObjectMsg(attached, object_id))
				    return;
			    attached.touch_links = touch_links;
			    if (!scene->processAttachedCollisionObjectMsg(attached))
				    throw std::runtime_error("failed to set held bottle touch links");
		    });
		attached_scene = stage.get();
		branch->insert(std::move(stage));
	}
	{
		auto stage = std::make_unique<mtc::stages::Connect>(
		    p + "transport_to_target_preplace",
		    mtc::stages::Connect::GroupPlannerVector{ { arm.planning_group, sampling } });
		stage->setTimeout(s.planning_timeout_s);
		branch->insert(std::move(stage));
	}
	{
		auto place = std::make_unique<mtc::SerialContainer>(p + "target_place");
		place->setProperty("group", arm.planning_group);
		{
			auto preplace = s.target_place_pose;
			preplace.position.x -= s.target_insert_direction.x * s.target_preplace_offset_m;
			preplace.position.y -= s.target_insert_direction.y * s.target_preplace_offset_m;
			preplace.position.z -= s.target_insert_direction.z * s.target_preplace_offset_m;
			auto generator = std::make_unique<mtc::stages::GeneratePose>(p + "target_preplace_pose");
			generator->setPose(stamped(preplace, s.frame_id));
			generator->setMonitoredStage(attached_scene);
			auto stage =
			    std::make_unique<mtc::stages::ComputeIK>(p + "target_preplace_ik", std::move(generator));
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->setMaxIKSolutions(static_cast<uint32_t>(s.max_ik_solutions));
			stage->setMinSolutionDistance(1.0);
			stage->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "target_approach", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			const double distance = s.target_preplace_offset_m - s.target_contact_distance_m;
			stage->setMinMaxDistance(distance, distance);
			stage->setDirection(stamped(s.target_insert_direction, s.frame_id));
			place->insert(std::move(stage));
		}
		{
			auto stage =
			    std::make_unique<mtc::stages::ModifyPlanningScene>(p + "allow_final_support_contact");
			stage->allowCollisions(s.bottle_id, s.target_support_surface_id, true);
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "target_contact", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->setMinMaxDistance(s.target_contact_distance_m, s.target_contact_distance_m);
			stage->setDirection(stamped(s.target_insert_direction, s.frame_id));
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "open_gripper_semantic");
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "detach_bottle");
			stage->detachObject(s.bottle_id, arm.ik_link);
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "target_retreat", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->setMinMaxDistance(s.target_retreat_distance_m, s.target_retreat_distance_m);
			stage->setDirection(stamped(s.target_retreat_direction, s.frame_id));
			place->insert(std::move(stage));
		}
		{
			auto stage =
			    std::make_unique<mtc::stages::ModifyPlanningScene>(p + "restore_support_collision_check");
			stage->allowCollisions(s.bottle_id, s.target_support_surface_id, false);
			stage->allowCollisions(s.bottle_id, arm.touch_links, false);
			place->insert(std::move(stage));
		}
		branch->insert(std::move(place));
	}
	{
		std::map<std::string, double> home_goal;
		for (std::size_t i = 0; i < s.post_place_home_joints_deg.size(); ++i)
			home_goal.emplace("r_joint" + std::to_string(i + 1),
			                  s.post_place_home_joints_deg[i] * M_PI / 180.0);
		auto stage =
		    std::make_unique<mtc::stages::MoveTo>(p + "move_to_post_place_home", sampling);
		stage->setGroup(arm.planning_group);
		stage->setGoal(home_goal);
		branch->insert(std::move(stage));
	}
	*raw = branch.get();
	return branch;
}

/// One arm/candidate branch. Pick-only ends after source retreat with the
/// bottle still attached; full-transfer keeps the historical transport/place
/// tail and restores hand contact after detach.
std::unique_ptr<mtc::SerialContainer> buildArmBranch(const Scenario& s, const ArmConfig& arm,
                                                     const grabber_mtc::GraspCandidate& grasp,
                                                     const std::string& branch_id,
                                                     const mtc::solvers::PlannerInterfacePtr& sampling,
                                                     const mtc::solvers::PlannerInterfacePtr& cartesian,
                                                     const std::map<std::string, double>& joint_weights,
                                                     mtc::SerialContainer** raw)
{
	if (s.place_only)
		return buildPlaceBranch(s, arm, branch_id, sampling, cartesian, raw);

	const std::string p = branch_id + "/";
	const Eigen::Isometry3d ik_frame = tcpFrame(arm);

	auto branch = std::make_unique<mtc::SerialContainer>(branch_id + "_branch");
	branch->setProperty("group", arm.planning_group);
	if (!arm.end_effector.empty())
		branch->setProperty("eef", arm.end_effector);

	mtc::Stage* support_scene = nullptr;
	{
		// The target starts on its measured support surface.  This object↔surface
		// semantic is independent of hand contact and is the only ACM relaxation
		// present during current_state→pregrasp.
		auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "allow_support_contact");
		if (!s.source_support_surface_id.empty())
			stage->allowCollisions(s.bottle_id, s.source_support_surface_id, true);
		if (!s.target_support_surface_id.empty() && s.target_support_surface_id != s.source_support_surface_id)
			stage->allowCollisions(s.bottle_id, s.target_support_surface_id, true);
		support_scene = stage.get();
		branch->insert(std::move(stage));
	}

	// --- free space: current state -> start of the source approach ---------
	{
		auto stage = std::make_unique<mtc::stages::Connect>(
		    p + "connect_to_source_pregrasp",
		    mtc::stages::Connect::GroupPlannerVector{ { arm.planning_group, sampling } });
		stage->setTimeout(s.planning_timeout_s);
		// The long free-space leg is where the audit used to discard nearly
		// everything. Joint constraints are sampled natively by OMPL (unlike a
		// TCP pose box, which forces projection sampling), so this narrows the
		// search instead of filtering its output.
		// MTC expands solutions best-first by cost, so a weighted path length
		// steers which branches get explored — not just which one is picked at
		// the end. Keeping RRTConnect means this costs no extra planning time.
		if (!joint_weights.empty())
			stage->setCostTerm(std::make_unique<mtc::cost::PathLength>(joint_weights));
		branch->insert(std::move(stage));
	}

	mtc::Stage* attach_stage = nullptr;

	// --- source shelf: approach, grasp, close, attach, retreat -------------
	{
		auto pick = std::make_unique<mtc::SerialContainer>(p + "source_pick");
		pick->setProperty("group", arm.planning_group);

		// Generate an explicit collision-free pregrasp for this final TCP
		// candidate. Connect can then reach it without any hand↔bottle ACM
		// relaxation, and both Cartesian legs propagate forwards from it.
		{
			auto pregrasp = grasp.pose;
			pregrasp.position.x -= s.source_approach_direction.x * s.source_pregrasp_offset_m;
			pregrasp.position.y -= s.source_approach_direction.y * s.source_pregrasp_offset_m;
			pregrasp.position.z -= s.source_approach_direction.z * s.source_pregrasp_offset_m;
			auto generator = std::make_unique<mtc::stages::GeneratePose>(p + "source_pregrasp_pose");
			generator->setPose(stamped(pregrasp, s.frame_id));
			generator->setMonitoredStage(support_scene);

			auto stage =
			    std::make_unique<mtc::stages::ComputeIK>(p + "source_pregrasp_ik", std::move(generator));
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->setMaxIKSolutions(static_cast<uint32_t>(s.max_ik_solutions));
			stage->setMinSolutionDistance(1.0);
			stage->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
			// This IK choice fixes the arm configuration for the whole pick:
			// both Cartesian legs propagate forward from it.  Bias it toward
			// the demonstrated posture rather than letting the tie break at
			// random.
			if (!s.source_grasp_reference_joints_deg.empty() && !joint_weights.empty())
				stage->setCostTerm(referencePostureCost(
				    s.source_grasp_reference_joints_deg, arm.planning_group,
				    joint_weights));
			pick->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "source_approach", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->properties().set("marker_ns", p + "source_approach");
			const double distance = s.source_pregrasp_offset_m - s.source_contact_distance_m;
			stage->setMinMaxDistance(distance, distance);
			stage->setDirection(stamped(s.source_approach_direction, s.frame_id));
			pick->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "allow_final_grasp_contact");
			stage->allowCollisions(s.bottle_id, arm.touch_links, true);
			pick->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "source_contact", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->properties().set("marker_ns", p + "source_contact");
			stage->setMinMaxDistance(s.source_contact_distance_m, s.source_contact_distance_m);
			stage->setDirection(stamped(s.source_approach_direction, s.frame_id));
			pick->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "attach_bottle");
			stage->attachObject(s.bottle_id, arm.ik_link);
			// MTC's attachObject() does not populate AttachedBody touch links.
			// Copy the same narrow grasp set into the attached-body semantics so
			// the temporary ACM entry can be restored while the bottle is held.
			stage->setCallback(
			    [object_id = s.bottle_id, touch_links = arm.touch_links](
			        const planning_scene::PlanningScenePtr& scene, const mtc::PropertyMap&) {
				    moveit_msgs::msg::AttachedCollisionObject attached;
				    // Backward propagation has already detached the object.
				    if (!scene->getAttachedCollisionObjectMsg(attached, object_id))
					    return;
				    attached.touch_links = touch_links;
				    if (!scene->processAttachedCollisionObjectMsg(attached))
					    throw std::runtime_error("failed to set attached bottle touch links");
			    });
			attach_stage = stage.get();
			pick->insert(std::move(stage));
		}
		if (s.pick_only)
		{
			// Lift off the support before any horizontal shelf exit. Support
			// contact remains allowed only for this vertical separation.
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "source_lift", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->properties().set("marker_ns", p + "source_lift");
			stage->setMinMaxDistance(s.source_lift_distance_m, s.source_lift_distance_m);
			stage->setDirection(stamped(s.source_lift_direction, s.frame_id));
			pick->insert(std::move(stage));
		}
		if (s.pick_only)
		{
			auto stage =
			    std::make_unique<mtc::stages::ModifyPlanningScene>(p + "forbid_support_contact_after_lift");
			if (!s.source_support_surface_id.empty())
				stage->allowCollisions(s.bottle_id, s.source_support_surface_id, false);
			if (!s.target_support_surface_id.empty() && s.target_support_surface_id != s.source_support_surface_id)
				stage->allowCollisions(s.bottle_id, s.target_support_surface_id, false);
			pick->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "source_retreat", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->properties().set("marker_ns", p + "source_retreat");
			const double min_distance =
			    s.cartesian_transport ? s.source_retreat_distance_m :
			                            s.source_retreat_distance_m * s.cartesian_min_fraction;
			stage->setMinMaxDistance(min_distance, s.source_retreat_distance_m);
			stage->setDirection(stamped(s.source_retreat_direction, s.frame_id));
			pick->insert(std::move(stage));
		}
		if (s.pick_only)
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "restore_bottle_collision_check");
			stage->allowCollisions(s.bottle_id, arm.touch_links, false);
			pick->insert(std::move(stage));
		}
		branch->insert(std::move(pick));
	}

	if (s.pick_only)
	{
		*raw = branch.get();
		return branch;
	}

	// --- transport: carry the bottle to the target preplace ----------------
	if (s.cartesian_transport)
	{
		geometry_msgs::msg::Vector3 direction;
		direction.x = s.target_place_pose.position.x -
		              s.target_insert_direction.x * s.target_preplace_offset_m -
		              grasp.pose.position.x -
		              s.source_retreat_direction.x * s.source_retreat_distance_m;
		direction.y = s.target_place_pose.position.y -
		              s.target_insert_direction.y * s.target_preplace_offset_m -
		              grasp.pose.position.y -
		              s.source_retreat_direction.y * s.source_retreat_distance_m;
		direction.z = s.target_place_pose.position.z -
		              s.target_insert_direction.z * s.target_preplace_offset_m -
		              grasp.pose.position.z -
		              s.source_retreat_direction.z * s.source_retreat_distance_m;
		const double distance =
		    std::sqrt(direction.x * direction.x + direction.y * direction.y +
		              direction.z * direction.z);
		if (!std::isfinite(distance) || distance <= 1e-6)
			throw std::runtime_error("cartesian transport corridor must be non-zero");
		direction.x /= distance;
		direction.y /= distance;
		direction.z /= distance;
		auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "transport", cartesian);
		stage->setGroup(arm.planning_group);
		stage->setIKFrame(ik_frame, arm.ik_link);
		stage->properties().set("marker_ns", p + "transport");
		stage->setMinMaxDistance(distance, distance);
		stage->setDirection(stamped(direction, s.frame_id));
		branch->insert(std::move(stage));
	}
	else
	{
		auto stage = std::make_unique<mtc::stages::Connect>(
		    p + "transport", mtc::stages::Connect::GroupPlannerVector{ { arm.planning_group, sampling } });
		stage->setTimeout(s.planning_timeout_s);
		branch->insert(std::move(stage));
	}

	// --- target shelf: insert, place, open, detach, retreat ----------------
	{
		auto place = std::make_unique<mtc::SerialContainer>(p + "target_place");
		place->setProperty("group", arm.planning_group);

		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "target_insert", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->properties().set("marker_ns", p + "target_insert");
			const double min_distance =
			    s.cartesian_transport ? s.target_preplace_offset_m :
			                            s.target_preplace_offset_m * s.cartesian_min_fraction;
			stage->setMinMaxDistance(min_distance, s.target_preplace_offset_m);
			stage->setDirection(stamped(s.target_insert_direction, s.frame_id));
			place->insert(std::move(stage));
		}
		if (!s.cartesian_transport)
		{
			auto generator = std::make_unique<mtc::stages::GeneratePose>(p + "target_place_pose");
			generator->setPose(stamped(s.target_place_pose, s.frame_id));
			generator->setMonitoredStage(attach_stage);

			auto stage = std::make_unique<mtc::stages::ComputeIK>(p + "target_place_ik", std::move(generator));
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->setMaxIKSolutions(static_cast<uint32_t>(s.max_ik_solutions));
			stage->setMinSolutionDistance(1.0);
			stage->properties().configureInitFrom(mtc::Stage::INTERFACE, { "target_pose" });
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "open_gripper_semantic");
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "detach_bottle");
			stage->detachObject(s.bottle_id, arm.ik_link);
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::MoveRelative>(p + "target_retreat", cartesian);
			stage->setGroup(arm.planning_group);
			stage->setIKFrame(ik_frame, arm.ik_link);
			stage->properties().set("marker_ns", p + "target_retreat");
			stage->setMinMaxDistance(s.target_retreat_distance_m * s.cartesian_min_fraction,
			                         s.target_retreat_distance_m);
			stage->setDirection(stamped(s.target_retreat_direction, s.frame_id));
			place->insert(std::move(stage));
		}
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>(p + "restore_bottle_collision_check");
			stage->allowCollisions(s.bottle_id, arm.touch_links, false);
			if (!s.source_support_surface_id.empty())
				stage->allowCollisions(s.bottle_id, s.source_support_surface_id, false);
			if (!s.target_support_surface_id.empty() && s.target_support_surface_id != s.source_support_surface_id)
				stage->allowCollisions(s.bottle_id, s.target_support_surface_id, false);
			place->insert(std::move(stage));
		}
		branch->insert(std::move(place));
	}

	*raw = branch.get();
	return branch;
}

double bestCost(const mtc::Stage& stage)
{
	double best = -1.0;
	for (const auto& solution : stage.solutions())
	{
		const double cost = solution->cost();
		if (best < 0.0 || cost < best)
			best = cost;
	}
	return best;
}

ArmResult collectArmResult(const ArmConfig& arm, const std::string& branch_id,
                           const std::string& grasp_candidate_id, const mtc::SerialContainer& branch)
{
	ArmResult result;
	result.arm_id = arm.arm_id;
	result.branch_id = branch_id;
	result.grasp_candidate_id = grasp_candidate_id;
	result.execution_eligible = arm.execution_eligible;
	result.execution_block_reason = arm.execution_block_reason;
	result.complete_solution_count = branch.solutions().size();
	result.solved = result.complete_solution_count > 0;
	result.best_total_cost = bestCost(branch);
	if (result.solved)
		result.selected_solution_id = branch_id + "#best";

	// Empty stages can be downstream fallout and never have run.  numFailures()
	// distinguishes an attempted failing leaf (including a real connector
	// failure) from an untriggered one.
	std::string first_failed_leaf;
	branch.traverseRecursively([&](const mtc::Stage& stage, unsigned int /*depth*/) {
		const double cost = bestCost(stage);
		result.stage_costs.emplace_back(stage.name(), cost);
		const bool is_ik = dynamic_cast<const mtc::stages::ComputeIK*>(&stage) != nullptr;
		if ((!is_ik && dynamic_cast<const mtc::ContainerBase*>(&stage)) || cost >= 0.0 ||
		    stage.numFailures() == 0)
			return true;
		if (first_failed_leaf.empty())
			first_failed_leaf = stage.name();
		return true;
	});
	result.earliest_failure_stage = first_failed_leaf;
	return result;
}

double durationSeconds(const builtin_interfaces::msg::Duration& duration)
{
	return static_cast<double>(duration.sec) + static_cast<double>(duration.nanosec) * 1e-9;
}

void validateTrajectoryTiming(const std::vector<grabber_mtc::TrajectoryPoint>& points,
                              const std::string& mode)
{
	if (points.empty())
		throw std::runtime_error(mode + " export contains no trajectory points");
	for (std::size_t i = 0; i < points.size(); ++i)
		if (!std::isfinite(points[i].time_from_start_s) || points[i].time_from_start_s < 0.0 ||
		    (i > 0 && points[i].time_from_start_s <= points[i - 1].time_from_start_s))
			throw std::runtime_error(mode + " export time_from_start is not strictly increasing");
}

grabber_mtc::PickTrajectoryExport exportPickTrajectory(
    const Scenario& scenario, const ArmConfig& arm,
    const std::vector<std::string>& arm_joint_names, const std::string& grasp_candidate_id,
    const mtc::SolutionBase& solution)
{
	moveit_task_constructor_msgs::msg::Solution message;
	solution.toMsg(message);

	const auto& expected = arm_joint_names;
	std::vector<const trajectory_msgs::msg::JointTrajectory*> segments;
	for (const auto& sub : message.sub_trajectory)
		if (!sub.trajectory.joint_trajectory.points.empty())
			segments.push_back(&sub.trajectory.joint_trajectory);
	if (segments.size() != 5)
		throw std::runtime_error("pick-only export expected exactly five motion segments, got " +
		                         std::to_string(segments.size()));
	constexpr std::size_t pregrasp_segments = 1;

	grabber_mtc::PickTrajectoryExport out;
	out.scenario_id = scenario.scenario_id;
	out.arm_id = arm.arm_id;
	out.execution_block_reason = arm.execution_eligible ?
	                                 "EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED" :
	                                 arm.execution_block_reason;
	out.grasp_candidate_id = grasp_candidate_id;
	out.target_captured_at_utc = scenario.target_captured_at_utc;
	out.scene_captured_at_utc = scenario.scene_captured_at_utc;
	out.freshness_max_age_s = scenario.freshness_max_age_s;
	out.joint_names = expected;

	const auto append = [&out, &expected](const trajectory_msgs::msg::JointTrajectory& segment) {
		if (segment.joint_names.size() != expected.size())
			throw std::runtime_error("pick-only segment does not contain exactly seven selected-arm joints");
		std::map<std::string, std::size_t> columns;
		for (std::size_t i = 0; i < segment.joint_names.size(); ++i)
			if (!columns.emplace(segment.joint_names[i], i).second)
				throw std::runtime_error("pick-only segment contains duplicate joint names");
		for (const auto& name : expected)
			if (columns.count(name) == 0)
				throw std::runtime_error("pick-only segment is missing joint " + name);

		const double offset = out.points.empty() ? 0.0 : out.points.back().time_from_start_s;
		for (std::size_t point_index = 0; point_index < segment.points.size(); ++point_index)
		{
			const auto& point = segment.points[point_index];
			if (point.positions.size() != segment.joint_names.size() ||
			    point.velocities.size() != segment.joint_names.size() ||
			    (!point.accelerations.empty() &&
			     point.accelerations.size() != segment.joint_names.size()))
				throw std::runtime_error(
				    "pick-only segment point dimensions do not match joint_names");
			std::vector<double> positions;
			std::vector<double> velocities;
			std::vector<double> accelerations;
			positions.reserve(expected.size());
			velocities.reserve(expected.size());
			accelerations.reserve(point.accelerations.empty() ? 0 : expected.size());
			for (const auto& name : expected)
			{
				positions.push_back(point.positions[columns.at(name)] * 180.0 / M_PI);
				velocities.push_back(point.velocities[columns.at(name)] * 180.0 / M_PI);
				if (!point.accelerations.empty())
					accelerations.push_back(point.accelerations[columns.at(name)] * 180.0 / M_PI);
			}

			if (!out.points.empty() && point_index == 0)
			{
				double error = 0.0;
				for (std::size_t i = 0; i < positions.size(); ++i)
					error = std::max(error, std::abs(positions[i] - out.points.back().positions_deg[i]));
				if (error > 0.1)
					throw std::runtime_error("pick-only motion segments are discontinuous");
				continue;
			}
			out.points.push_back({ offset + durationSeconds(point.time_from_start),
			                       std::move(positions), std::move(velocities),
			                       std::move(accelerations) });
		}
	};

	for (std::size_t i = 0; i < pregrasp_segments; ++i)
		append(*segments[i]);
	if (out.points.empty())
		throw std::runtime_error("pick-only pregrasp segment is empty");
	const std::size_t pregrasp_end = out.points.size() - 1;
	out.phases.push_back({ "pregrasp", 0, pregrasp_end });

	const std::size_t before_approach = out.points.size();
	append(*segments[pregrasp_segments]);
	append(*segments[pregrasp_segments + 1]);
	if (out.points.size() == before_approach)
		throw std::runtime_error("pick-only approach/contact segments contain no motion");
	const std::size_t attach = out.points.size() - 1;
	out.phases.push_back({ "approach", pregrasp_end, attach });
	out.phases.push_back({ "attach", attach, attach });
	out.attach_point_index = attach;

	const std::size_t before_retreat = out.points.size();
	append(*segments[pregrasp_segments + 2]);
	append(*segments[pregrasp_segments + 3]);
	if (out.points.size() == before_retreat)
		throw std::runtime_error("pick-only lift/retreat segments contain no motion");
	out.phases.push_back({ "retreat", attach, out.points.size() - 1 });
	validateTrajectoryTiming(out.points, "pick-only");
	return out;
}

grabber_mtc::PlaceTrajectoryExport exportPlaceTrajectory(
    const Scenario& scenario, const ArmConfig& arm, const mtc::SolutionBase& solution)
{
	moveit_task_constructor_msgs::msg::Solution message;
	solution.toMsg(message);
	const std::vector<std::string> expected = { "r_joint1", "r_joint2", "r_joint3", "r_joint4",
		                                        "r_joint5", "r_joint6", "r_joint7" };
	std::vector<const trajectory_msgs::msg::JointTrajectory*> segments;
	for (const auto& sub : message.sub_trajectory)
		if (!sub.trajectory.joint_trajectory.points.empty())
			segments.push_back(&sub.trajectory.joint_trajectory);
	if (segments.size() != 5)
		throw std::runtime_error("place-only export expected five motion segments "
		                         "(transport, approach, contact, retreat, home), got " +
		                         std::to_string(segments.size()));

	grabber_mtc::PlaceTrajectoryExport out;
	out.scenario_id = scenario.scenario_id;
	out.arm_id = arm.arm_id;
	out.scene_captured_at_utc = scenario.scene_captured_at_utc;
	out.freshness_max_age_s = scenario.freshness_max_age_s;
	out.joint_names = expected;
	const auto append = [&out, &expected](const trajectory_msgs::msg::JointTrajectory& segment) {
		if (segment.joint_names.size() != expected.size())
			throw std::runtime_error("place-only segment does not contain exactly seven right-arm joints");
		std::map<std::string, std::size_t> columns;
		for (std::size_t i = 0; i < segment.joint_names.size(); ++i)
			if (!columns.emplace(segment.joint_names[i], i).second)
				throw std::runtime_error("place-only segment contains duplicate joint names");
		for (const auto& name : expected)
			if (columns.count(name) == 0)
				throw std::runtime_error("place-only segment is missing joint " + name);
		const double offset = out.points.empty() ? 0.0 : out.points.back().time_from_start_s;
		for (std::size_t point_index = 0; point_index < segment.points.size(); ++point_index)
		{
			const auto& point = segment.points[point_index];
			if (point.positions.size() != segment.joint_names.size() ||
			    point.velocities.size() != segment.joint_names.size() ||
			    (!point.accelerations.empty() &&
			     point.accelerations.size() != segment.joint_names.size()))
				throw std::runtime_error(
				    "place-only segment point dimensions do not match joint_names");
			std::vector<double> positions, velocities, accelerations;
			for (const auto& name : expected)
			{
				positions.push_back(point.positions[columns.at(name)] * 180.0 / M_PI);
				velocities.push_back(point.velocities[columns.at(name)] * 180.0 / M_PI);
				if (!point.accelerations.empty())
					accelerations.push_back(point.accelerations[columns.at(name)] * 180.0 / M_PI);
			}
			if (!out.points.empty() && point_index == 0)
			{
				double error = 0.0;
				for (std::size_t i = 0; i < positions.size(); ++i)
					error = std::max(error, std::abs(positions[i] - out.points.back().positions_deg[i]));
				if (error > 0.1)
					throw std::runtime_error("place-only motion segments are discontinuous");
				continue;
			}
			out.points.push_back({ offset + durationSeconds(point.time_from_start),
			                       std::move(positions), std::move(velocities),
			                       std::move(accelerations) });
		}
	};

	append(*segments[0]);
	const std::size_t transport_end = out.points.size() - 1;
	out.phases.push_back({ "transport", 0, transport_end });
	append(*segments[1]);
	append(*segments[2]);
	const std::size_t release = out.points.size() - 1;
	out.phases.push_back({ "approach", transport_end, release });
	out.phases.push_back({ "release", release, release });
	out.release_point_index = release;
	append(*segments[3]);
	append(*segments[4]);
	out.phases.push_back({ "retreat", release, out.points.size() - 1 });
	validateTrajectoryTiming(out.points, "place-only");
	return out;
}

std::pair<double, double> placeTransportTcpMetrics(
    const ArmConfig& arm, const moveit::core::RobotState& start_state,
    const grabber_mtc::PlaceTrajectoryExport& trajectory)
{
	const auto phase = std::find_if(
	    trajectory.phases.begin(), trajectory.phases.end(),
	    [](const auto& item) { return item.name == "transport"; });
	if (phase == trajectory.phases.end() || phase->start_index >= phase->end_index ||
	    phase->end_index >= trajectory.points.size())
		throw std::runtime_error("place-only transport phase is invalid");
	moveit::core::RobotState state(start_state);
	std::vector<Eigen::Vector3d> tcp;
	for (std::size_t index = phase->start_index; index <= phase->end_index; ++index)
	{
		const auto& point = trajectory.points[index];
		for (std::size_t joint = 0; joint < trajectory.joint_names.size(); ++joint)
			state.setVariablePosition(trajectory.joint_names[joint],
			                          point.positions_deg[joint] * M_PI / 180.0);
		state.update();
		tcp.push_back(
		    (state.getGlobalLinkTransform(arm.ik_link) * arm.tcp_transform_from_ik_link)
		        .translation());
	}
	double path_length_m = 0.0;
	for (std::size_t index = 1; index < tcp.size(); ++index)
		path_length_m += (tcp[index] - tcp[index - 1]).norm();
	return { path_length_m, (tcp.back() - tcp.front()).norm() };
}

grabber_mtc::FullTransferTrajectoryExport exportFullTransferTrajectory(
    const Scenario& scenario, const ArmConfig& arm,
    const std::string& grasp_candidate_id, const mtc::SolutionBase& solution)
{
	moveit_task_constructor_msgs::msg::Solution message;
	solution.toMsg(message);
	std::vector<const trajectory_msgs::msg::JointTrajectory*> segments;
	for (const auto& sub : message.sub_trajectory)
		if (!sub.trajectory.joint_trajectory.points.empty())
			segments.push_back(&sub.trajectory.joint_trajectory);
	if (segments.size() != 7)
		throw std::runtime_error(
		    "full-transfer export expected seven motion segments "
		    "(pregrasp, approach, contact, source retreat, transport, place, "
		    "target retreat), got " +
		    std::to_string(segments.size()));

	grabber_mtc::FullTransferTrajectoryExport out;
	out.scenario_id = scenario.scenario_id;
	out.arm_id = arm.arm_id;
	out.grasp_candidate_id = grasp_candidate_id;
	out.joint_names = segments.front()->joint_names;
	std::sort(out.joint_names.begin(), out.joint_names.end());
	if (out.joint_names.size() != 7 ||
	    std::adjacent_find(out.joint_names.begin(), out.joint_names.end()) !=
	        out.joint_names.end())
		throw std::runtime_error(
		    "full-transfer segment must contain seven unique arm joints");

	const auto append = [&out](const trajectory_msgs::msg::JointTrajectory& segment) {
		if (segment.joint_names.size() != out.joint_names.size())
			throw std::runtime_error(
			    "full-transfer segment does not contain exactly seven joints");
		std::map<std::string, std::size_t> columns;
		for (std::size_t i = 0; i < segment.joint_names.size(); ++i)
			if (!columns.emplace(segment.joint_names[i], i).second)
				throw std::runtime_error(
				    "full-transfer segment contains duplicate joint names");
		for (const auto& name : out.joint_names)
			if (columns.count(name) == 0)
				throw std::runtime_error(
				    "full-transfer segment is missing joint " + name);
		const double offset =
		    out.points.empty() ? 0.0 : out.points.back().time_from_start_s;
		for (std::size_t point_index = 0;
		     point_index < segment.points.size(); ++point_index)
		{
			const auto& point = segment.points[point_index];
			if (point.positions.size() != segment.joint_names.size() ||
			    point.velocities.size() != segment.joint_names.size() ||
			    (!point.accelerations.empty() &&
			     point.accelerations.size() != segment.joint_names.size()))
				throw std::runtime_error(
				    "full-transfer point dimensions do not match joint_names");
			std::vector<double> positions;
			std::vector<double> velocities;
			std::vector<double> accelerations;
			for (const auto& name : out.joint_names)
			{
				positions.push_back(
				    point.positions[columns.at(name)] * 180.0 / M_PI);
				velocities.push_back(
				    point.velocities[columns.at(name)] * 180.0 / M_PI);
				if (!point.accelerations.empty())
					accelerations.push_back(
					    point.accelerations[columns.at(name)] * 180.0 / M_PI);
			}
			if (!out.points.empty() && point_index == 0)
			{
				double error = 0.0;
				for (std::size_t i = 0; i < positions.size(); ++i)
					error = std::max(
					    error,
					    std::abs(positions[i] -
					             out.points.back().positions_deg[i]));
				if (error > 0.1)
					throw std::runtime_error(
					    "full-transfer motion segments are discontinuous");
				continue;
			}
			out.points.push_back(
			    { offset + durationSeconds(point.time_from_start),
			      std::move(positions), std::move(velocities),
			      std::move(accelerations) });
		}
	};

	append(*segments[0]);
	const std::size_t pregrasp = out.points.size() - 1;
	out.phases.push_back({ "pregrasp", 0, pregrasp });
	append(*segments[1]);
	append(*segments[2]);
	const std::size_t attach = out.points.size() - 1;
	out.phases.push_back({ "approach", pregrasp, attach });
	out.phases.push_back({ "attach", attach, attach });
	out.attach_point_index = attach;
	append(*segments[3]);
	const std::size_t source_retreat = out.points.size() - 1;
	out.phases.push_back({ "source_retreat", attach, source_retreat });
	append(*segments[4]);
	const std::size_t transport = out.points.size() - 1;
	out.phases.push_back({ "transport", source_retreat, transport });
	append(*segments[5]);
	const std::size_t release = out.points.size() - 1;
	out.phases.push_back({ "place", transport, release });
	out.phases.push_back({ "release", release, release });
	out.release_point_index = release;
	append(*segments[6]);
	out.phases.push_back(
	    { "target_retreat", release, out.points.size() - 1 });
	validateTrajectoryTiming(out.points, "full-transfer");
	return out;
}

}  // namespace

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::NodeOptions node_options;
	node_options.automatically_declare_parameters_from_overrides(true);
	auto node = rclcpp::Node::make_shared("grabber_mtc_planner", node_options);

	Options options;
	Scenario scenario;
	std::vector<ArmConfig> arms;
	try
	{
		options = parseArgs(rclcpp::remove_ros_arguments(argc, argv));
		scenario = grabber_mtc::loadScenario(options.scenario_path);
		arms = grabber_mtc::loadArmConfigs(options.arms_path);
	}
	catch (const std::exception& e)
	{
		RCLCPP_ERROR(LOGGER, "%s", e.what());
		rclcpp::shutdown();
		return 2;
	}

	// MTC's CurrentState fetches the live PlanningScene over a service, so the
	// node has to be spinning while we plan.
	rclcpp::executors::MultiThreadedExecutor executor;
	executor.add_node(node);
	std::thread spinner([&executor]() { executor.spin(); });
	sensor_msgs::msg::JointState::SharedPtr latest_joint_state;
	std::mutex latest_joint_state_mutex;
	auto joint_state_subscription = node->create_subscription<sensor_msgs::msg::JointState>(
	    "/joint_states", rclcpp::SensorDataQoS(),
	    [&latest_joint_state, &latest_joint_state_mutex](
	        sensor_msgs::msg::JointState::SharedPtr message) {
		    std::lock_guard<std::mutex> lock(latest_joint_state_mutex);
		    latest_joint_state = std::move(message);
	    });
	(void)joint_state_subscription;

	int exit_code = 0;
	std::string selected_arm_joint_state_id;
	std::map<std::string, double> selected_arm_joint_snapshot;
	std::int64_t selected_arm_joint_state_stamp_ns = 0;
	double selected_arm_joint_state_age_s = -1.0;
	try
	{
		mtc::Task task;
		task.stages()->setName("shelf_transfer/" + scenario.scenario_id);
		task.loadRobotModel(node);

		const auto& model = task.getRobotModel();
		// The scenario frame does not have to BE the model frame (the shelf is
		// measured against platform_base_link, which rides on the lift), but it
		// must be a frame the model knows, or every pose lands somewhere else.
		if (scenario.frame_id != model->getModelFrame() && !model->hasLinkModel(scenario.frame_id))
			throw std::runtime_error("scenario frame_id '" + scenario.frame_id +
			                         "' is neither the robot model frame '" + model->getModelFrame() +
			                         "' nor a link of it; poses would be silently misplaced");
		for (const auto& arm : arms)
		{
			if (!model->hasJointModelGroup(arm.planning_group))
				throw std::runtime_error("planning group '" + arm.planning_group + "' is not in robot model '" +
				                         model->getName() + "'");
			if (!model->hasLinkModel(arm.ik_link))
				throw std::runtime_error("ik_link '" + arm.ik_link + "' is not in robot model '" + model->getName() +
				                         "'");
		}
		if (scenario.pick_only || scenario.place_only)
		{
			const auto arm = std::find_if(
			    arms.begin(), arms.end(), [&scenario](const ArmConfig& item) {
				    return item.arm_id == scenario.planning_arm_id;
			    });
			if (arm == arms.end())
				throw std::runtime_error("selected planning arm is missing from arm config");
			selected_arm_joint_state_id = arm->arm_id;
			const auto required_names = activeJointNames(*model, *arm);
			// How long to wait for DDS discovery, NOT how stale a sample may be.
			// Freshness is judged below on the message's own stamp (age <= 0.5 s)
			// and is unaffected by this; the loop breaks the moment a good
			// sample arrives, so a generous value costs nothing when healthy.
			//
			// 2 s was too tight. /joint_states is published SensorDataQoS, i.e.
			// VOLATILE, so a subscriber gets nothing published before it finished
			// discovering the publisher — and every scenario spawns a brand new
			// process that must rediscover from scratch. On a domain churning a
			// node every few seconds that intermittently exceeded 2 s, and the
			// run died with "no complete fresh /joint_states sample" before
			// planning anything. Observed 2026-08-02 17:53-17:55: the same seed
			// failed once then succeeded on retry, and a later case failed three
			// times running, while the fixture publisher was up the whole time
			// at 20 Hz on the same ROS_DOMAIN_ID. The batch harness was retrying
			// around it, which hid a startup race as a planning failure.
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
			while (std::chrono::steady_clock::now() < deadline)
			{
				sensor_msgs::msg::JointState::SharedPtr message;
				{
					std::lock_guard<std::mutex> lock(latest_joint_state_mutex);
					message = latest_joint_state;
				}
				if (message && message->name.size() == message->position.size())
				{
					const rclcpp::Time stamp(message->header.stamp);
					const double age_s = (node->now() - stamp).seconds();
					std::map<std::string, double> values;
					for (std::size_t i = 0; i < message->name.size(); ++i)
						if (std::isfinite(message->position[i]))
							values[message->name[i]] = message->position[i];
					const bool complete = std::all_of(
					    required_names.begin(), required_names.end(),
					    [&values](const std::string& name) { return values.count(name) == 1; });
					if (stamp.nanoseconds() > 0 && age_s >= 0.0 && age_s <= 0.5 && complete)
					{
						for (const auto& name : required_names)
							selected_arm_joint_snapshot[name] = values.at(name);
						selected_arm_joint_state_stamp_ns = stamp.nanoseconds();
						selected_arm_joint_state_age_s = age_s;
						break;
					}
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(20));
			}
			if (selected_arm_joint_snapshot.size() != required_names.size())
				throw std::runtime_error(
				    "selected arm has no complete fresh /joint_states sample; refusing to plan");
		}

		auto sampling = std::make_shared<mtc::solvers::PipelinePlanner>(node);
		sampling->setPlannerId(scenario.planner_id);
		sampling->setTimeout(scenario.planning_timeout_s);

		// Deterministic straight lines for the shelf-mouth segments.
		auto cartesian = std::make_shared<mtc::solvers::CartesianPath>();
		cartesian->setMaxVelocityScalingFactor(0.2);
		cartesian->setMaxAccelerationScalingFactor(0.2);
		cartesian->setStepSize(0.005);
		mtc::solvers::PlannerInterfacePtr local_motion = cartesian;
		if (scenario.local_motion_planner == "pilz_lin")
		{
			auto pilz_lin = std::make_shared<mtc::solvers::PipelinePlanner>(
			    node, "pilz_industrial_motion_planner");
			pilz_lin->setPlannerId("LIN");
			local_motion = pilz_lin;
		}

		mtc::Stage* current_state = nullptr;
		// Kept separately because `current_state` is deliberately re-pointed at
		// later scene-modifying stages so generators monitor the scene they plan
		// against; this one stays the actual CurrentState stage so the start
		// state can be read back out of it after planning.
		mtc::Stage* current_state_stage = nullptr;
		{
			auto stage = std::make_unique<mtc::stages::CurrentState>("current_state");
			current_state = stage.get();
			current_state_stage = current_state;
			task.add(std::move(stage));
		}
		if (scenario.spawn_scene_objects)
		{
			auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("spawn_scenario_objects");
			for (const auto& object : grabber_mtc::buildSceneObjects(scenario))
				stage->addObject(object);
			if (scenario.place_only)
			{
				const auto arm = std::find_if(
				    arms.begin(), arms.end(), [&scenario](const ArmConfig& item) {
					    return item.arm_id == scenario.planning_arm_id;
				    });
				if (arm == arms.end())
					throw std::runtime_error("place-only planning arm is missing");
				stage->allowCollisions(scenario.bottle_id, arm->touch_links, true);
			}
			if (scenario.pick_only && scenario.has_tcp_path_workspace)
			{
				const auto arm = std::find_if(
				    arms.begin(), arms.end(), [&scenario](const ArmConfig& item) {
					    return item.arm_id == scenario.planning_arm_id;
				    });
				if (arm == arms.end())
					throw std::runtime_error("pick-only planning arm is missing");
				stage->addObject(buildTcpWorkspaceShell(scenario, *arm));
				// The audit this mirrors bounds the TCP point, not the whole
				// robot. Only the links that carry the TCP may be stopped by the
				// shell; the torso, base and left arm legitimately sit outside
				// the certified volume and must keep ignoring it.
				std::vector<std::string> tcp_links = arm->touch_links;
				if (std::find(tcp_links.begin(), tcp_links.end(), arm->ik_link) == tcp_links.end())
					tcp_links.push_back(arm->ik_link);
				stage->allowCollisions(TCP_WORKSPACE_SHELL_ID, model->getLinkModelNames(), true);
				stage->allowCollisions(TCP_WORKSPACE_SHELL_ID, tcp_links, false);
				// The bottle rides the hand after attach and is not the thing the
				// workspace bound is about.
				stage->allowCollisions(TCP_WORKSPACE_SHELL_ID, scenario.bottle_id, true);
			}
			current_state = stage.get();  // generators must monitor the scene they plan against
			task.add(std::move(stage));
		}

		struct PlannedBranch
		{
			const ArmConfig* arm;
			std::string branch_id;
			std::string grasp_candidate_id;
			mtc::SerialContainer* stage;
		};
		std::vector<PlannedBranch> planned_branches;
		auto alternatives = std::make_unique<mtc::Alternatives>(
		    scenario.pick_only ? "grasp_candidate_selection" :
		    (scenario.place_only ? "place_plan" : "arm_selection"));
		if (scenario.pick_only)
		{
			const auto arm = std::find_if(arms.begin(), arms.end(), [&scenario](const ArmConfig& item) {
				return item.arm_id == scenario.planning_arm_id;
			});
			if (arm == arms.end())
				throw std::runtime_error("pick-only planning_arm_id '" + scenario.planning_arm_id +
				                         "' is not present in the arm config");
			for (const auto& candidate : scenario.source_grasp_candidates)
			{
				mtc::SerialContainer* raw = nullptr;
				const std::string branch_id = arm->arm_id + "__" + candidate.id;
				alternatives->insert(
				    buildArmBranch(scenario, *arm, candidate, branch_id, sampling, local_motion,
				                   jointCostWeights(scenario, *model, *arm), &raw));
				planned_branches.push_back({ &*arm, branch_id, candidate.id, raw });
			}
		}
		else if (scenario.place_only)
		{
			const auto arm = std::find_if(arms.begin(), arms.end(), [&scenario](const ArmConfig& item) {
				return item.arm_id == scenario.planning_arm_id;
			});
			if (arm == arms.end())
				throw std::runtime_error("place-only planning_arm_id is not present in arm config");
			mtc::SerialContainer* raw = nullptr;
			const std::string branch_id = arm->arm_id + "__place";
			alternatives->insert(buildArmBranch(
			    scenario, *arm, scenario.source_grasp_candidates.front(), branch_id,
			    sampling, local_motion, {}, &raw));
			planned_branches.push_back(
			    { &*arm, branch_id, scenario.source_grasp_candidates.front().id, raw });
		}
		else
		{
			const auto& candidate = scenario.source_grasp_candidates.front();
			for (const auto& arm : arms)
			{
				if (!scenario.planning_arm_id.empty() &&
				    arm.arm_id != scenario.planning_arm_id)
					continue;
				mtc::SerialContainer* raw = nullptr;
				alternatives->insert(
				    buildArmBranch(scenario, arm, candidate, arm.arm_id, sampling, local_motion,
				                   {}, &raw));
				planned_branches.push_back({ &arm, arm.arm_id, candidate.id, raw });
			}
			if (planned_branches.empty())
				throw std::runtime_error(
				    "full-transfer planning_arm_id '" +
				    scenario.planning_arm_id +
				    "' is not present in the arm config");
		}
		task.add(std::move(alternatives));

		RunResult run;
		run.scenario_id = scenario.scenario_id;
		run.mode = scenario.pick_only ? "pick_only" :
		           (scenario.place_only ? "place_only" : "full_transfer");
		run.scenario = &scenario;
		run.scene_version = scenario.scene_version;
		run.fixture_source = scenario.fixture_source;
		run.robot_model_name = model->getName();
		run.start_state_selected_arm = selected_arm_joint_state_id;
		run.start_state_joint_state_stamp_ns = selected_arm_joint_state_stamp_ns;
		run.start_state_joint_state_age_s_at_planning = selected_arm_joint_state_age_s;
		for (const auto& branch : planned_branches)
			if (std::find(run.planning_group_names.begin(), run.planning_group_names.end(),
			              branch.arm->planning_group) == run.planning_group_names.end())
				run.planning_group_names.push_back(branch.arm->planning_group);

		const auto started = std::chrono::steady_clock::now();
		try
		{
			task.init();
			static_cast<void>(task.plan(scenario.max_solutions));
		}
		catch (const mtc::InitStageException& e)
		{
			std::ostringstream os;
			os << e;
			RCLCPP_ERROR(LOGGER, "task init failed: %s", os.str().c_str());
			exit_code = 3;
		}
		run.planning_wall_time_s =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

		// Record what CurrentState actually produced. The global all-zero bit is
		// legacy evidence only: platform/other-arm values cannot prove the selected
		// arm is live, so the selected-arm snapshot above is compared joint by joint.
		if (current_state_stage && !current_state_stage->solutions().empty())
		{
			const auto& state = (*current_state_stage->solutions().begin())->end()->scene()->getCurrentState();
			for (const std::string& variable : model->getVariableNames())
			{
				const double value = state.getVariablePosition(variable);
				run.start_state_joints.emplace_back(variable, value);
				if (std::abs(value) > 1e-9)
					run.start_state_all_zero = false;
			}
			run.start_state_selected_arm_complete =
			    !selected_arm_joint_snapshot.empty();
			for (const auto& [name, expected] : selected_arm_joint_snapshot)
				if (std::abs(state.getVariablePosition(name) - expected) > 0.02)
					run.start_state_selected_arm_complete = false;
		}
		if ((scenario.pick_only || scenario.place_only) &&
		    !run.start_state_selected_arm_complete)
			throw std::runtime_error(
			    "MTC CurrentState does not match fresh /joint_states for selected arm");

		std::size_t selected_index = planned_branches.size();
		const mtc::SolutionBase* selected_pick_solution = nullptr;
		grabber_mtc::PickTrajectoryExport selected_pick_trajectory;
		double selected_pick_joint_travel = std::numeric_limits<double>::infinity();
		double selected_pick_j4_margin = -std::numeric_limits<double>::infinity();
		const mtc::SolutionBase* selected_place_solution = nullptr;
		grabber_mtc::PlaceTrajectoryExport selected_place_trajectory;
		double selected_place_transport_length = std::numeric_limits<double>::infinity();
		const mtc::SolutionBase* selected_full_solution = nullptr;
		grabber_mtc::FullTransferTrajectoryExport selected_full_trajectory;
		for (const auto& branch : planned_branches)
		{
			if (branch.stage)
				run.arms.push_back(collectArmResult(*branch.arm, branch.branch_id,
				                                   branch.grasp_candidate_id, *branch.stage));
			else
			{
				ArmResult failed;
				failed.arm_id = branch.arm->arm_id;
				failed.branch_id = branch.branch_id;
				failed.grasp_candidate_id = branch.grasp_candidate_id;
				failed.earliest_failure_stage = "task_init";
				run.arms.push_back(std::move(failed));
			}
		}
		// Hard feasibility first. Prefer execution eligibility; cost chooses
		// between complete solutions with the same eligibility.
		for (std::size_t i = 0; i < run.arms.size(); ++i)
		{
			const auto& arm_result = run.arms[i];
			const ArmResult* selected =
			    selected_index < run.arms.size() ? &run.arms[selected_index] : nullptr;
			if (arm_result.solved &&
			    (!selected || (arm_result.execution_eligible && !selected->execution_eligible) ||
			     (arm_result.execution_eligible == selected->execution_eligible &&
			      arm_result.best_total_cost < selected->best_total_cost)))
					selected_index = i;
		}
		if (scenario.pick_only)
		{
			if (!current_state_stage || current_state_stage->solutions().empty())
				throw std::runtime_error(
				    "pick-only task has no CurrentState for post-plan FK audit");
			const auto& planned_start_state =
			    (*current_state_stage->solutions().begin())->end()->scene()->getCurrentState();
			// A hand parked outside the certified volume collides with the
			// shell on the very first state, which would otherwise surface as
			// an unexplained "no solution".
			if (scenario.has_tcp_path_workspace && !planned_branches.empty())
			{
				const auto& arm = *planned_branches.front().arm;
				const Eigen::Vector3d tcp =
				    tcpInWorkspaceFrame(scenario, arm, planned_start_state);
				const Eigen::Vector3d half_size(scenario.tcp_path_workspace.size[0] / 2.0,
				                                scenario.tcp_path_workspace.size[1] / 2.0,
				                                scenario.tcp_path_workspace.size[2] / 2.0);
				if ((tcp.array().abs() > half_size.array()).any())
					throw std::runtime_error(
					    "start state TCP is outside tcp_path_workspace at local [" +
					    std::to_string(tcp.x()) + ", " + std::to_string(tcp.y()) + ", " +
					    std::to_string(tcp.z()) + "], half size [" +
					    std::to_string(half_size.x()) + ", " + std::to_string(half_size.y()) +
					    ", " + std::to_string(half_size.z()) +
					    "]; move the arm to the taught home before planning");
			}
			selected_index = planned_branches.size();
			for (std::size_t i = 0; i < planned_branches.size(); ++i)
			{
				const auto& branch = planned_branches[i];
				if (!branch.stage)
					continue;
				for (const auto& candidate : branch.stage->solutions())
				{
					auto trajectory = exportPickTrajectory(
					    scenario, *branch.arm, activeJointNames(*model, *branch.arm),
					    branch.grasp_candidate_id, *candidate);
					const auto execution_audit = auditExecutionTrajectory(
					    *branch.arm, planned_start_state, trajectory.joint_names,
					    trajectory.points, { trajectory.attach_point_index });
					const auto grasp_candidate = std::find_if(
					    scenario.source_grasp_candidates.begin(),
					    scenario.source_grasp_candidates.end(),
					    [&branch](const auto& item) {
						    return item.id == branch.grasp_candidate_id;
					    });
					const double authored_roll =
					    grasp_candidate == scenario.source_grasp_candidates.end() ?
					        std::numeric_limits<double>::infinity() :
					        authoredFingerRollDeg(grasp_candidate->pose);
					const double planned_roll = plannedFingerRollDeg(
					    scenario, *branch.arm, planned_start_state, trajectory.joint_names,
					    trajectory.points.at(trajectory.attach_point_index));
					const bool roll_safe = authored_roll <= 1e-3 &&
					                       planned_roll <= EXECUTION_MAX_FINGER_ROLL_DEG;
					std::vector<double> minimum(7, std::numeric_limits<double>::infinity());
					std::vector<double> maximum(7, -std::numeric_limits<double>::infinity());
					for (const auto& point : trajectory.points)
						for (std::size_t joint = 0; joint < 7; ++joint)
						{
							minimum[joint] =
							    std::min(minimum[joint], point.positions_deg.at(joint));
							maximum[joint] =
							    std::max(maximum[joint], point.positions_deg.at(joint));
						}
					bool joint_range_safe = true;
					for (std::size_t joint = 0; joint < 7; ++joint)
						if (maximum[joint] - minimum[joint] >=
						    EXECUTION_MAX_JOINT_RANGE_DEG)
							joint_range_safe = false;
					// The J4 band is a CARTESIAN-control gate, and it is scoped to
					// where that control actually happens.
					//
					// Elbow-extended is a singularity of the Jacobian: reaching a
					// TCP velocity needs unbounded joint velocity, and IK along a
					// straight line becomes ill-conditioned and jumps. That is
					// real for the approach/contact/lift/retreat legs, which MTC
					// plans as Cartesian straight lines, and for the pregrasp IK
					// solution those legs start from.
					//
					// It is NOT real for the free-space leg. Every segment here is
					// executed by RobotSession::execute_planned_joints -> rm_movej
					// (rm_movel appears nowhere in the MTC execution path), so the
					// arm interpolates joint angles and no Jacobian is inverted.
					// Passing through elbow-extended in joint space is ordinary
					// motion, still fully collision-checked, with per-step joint
					// deltas bounded separately by planned_joint_step_deg.
					//
					// Checking it over the whole trajectory rejected paths that
					// execute perfectly well, and made success depend on whether
					// the arm happened to be parked on the same elbow branch as
					// the reachable pregrasp — if not, EVERY joint-space path has
					// to cross zero and all of them were refused.
					const std::size_t j4_audit_begin = trajectory.phases.front().end_index;
					bool j4_safe = true;
					double previous_j4 = 0.0;
					double j4_margin = std::numeric_limits<double>::infinity();
					for (std::size_t point = j4_audit_begin; point < trajectory.points.size();
					     ++point)
					{
						const double j4 = trajectory.points[point].positions_deg.at(3);
						j4_margin = std::min(j4_margin, std::abs(j4));
						// previous_j4 starts at 0, so seed it from the first audited
						// point rather than reading a sign change against nothing.
						if (std::abs(j4) < EXECUTION_J4_SINGULARITY_DEG ||
						    (point > j4_audit_begin && previous_j4 * j4 < 0.0))
						{
							j4_safe = false;
							break;
						}
						previous_j4 = j4;
					}
					// Weighted with the same table the planner's cost term uses.
					// Summing all seven joints equally priced a wrist roll like a
					// shoulder sweep, so the "shortest" solution was often the
					// one that looked least coordinated.
					double joint_travel = 0.0;
					const std::size_t pregrasp_end = trajectory.phases.front().end_index;
					for (std::size_t point = 1; point <= pregrasp_end; ++point)
						for (std::size_t joint = 0; joint < 7; ++joint)
							joint_travel += scenario.planning_joint_weights[joint] *
							                std::abs(
							                    trajectory.points[point].positions_deg[joint] -
							                    trajectory.points[point - 1].positions_deg[joint]);
					std::size_t first_bad_segment = 0;
					Eigen::Vector3d first_bad_tcp = Eigen::Vector3d::Zero();
					const bool workspace_safe = tcpWorkspaceContainsTrajectory(
					    scenario, *branch.arm, planned_start_state, trajectory,
					    &first_bad_segment, &first_bad_tcp);
					if (!execution_audit.safe() || !roll_safe || !j4_safe || !joint_range_safe ||
					    !workspace_safe)
					{
						RCLCPP_WARN(
						    LOGGER,
						    "rejecting %s solution: execution_safe=%s controller_safe=%s "
						    "joint_bounds_safe=%s timing_safe=%s jacobian_safe=%s "
						    "roll_safe=%s j4_safe=%s wrap_safe=%s workspace_safe=%s commands=%zu "
						    "min_jacobian_sigma=%.5f max_jacobian_condition=%.2f "
						    "authored_roll=%.4fdeg planned_roll=%.4fdeg "
						    "pregrasp_joint_travel=%.3fdeg j4_margin=%.3fdeg "
						    "first_bad_segment=%zu first_bad_tcp_local=[%.3f,%.3f,%.3f]",
						    branch.branch_id.c_str(), execution_audit.safe() ? "true" : "false",
						    execution_audit.controller_safe ? "true" : "false",
						    execution_audit.joint_bounds_safe ? "true" : "false",
						    execution_audit.timing_safe ? "true" : "false",
						    execution_audit.jacobian_safe ? "true" : "false",
						    roll_safe ? "true" : "false",
						    j4_safe ? "true" : "false",
						    joint_range_safe ? "true" : "false",
						    workspace_safe ? "true" : "false",
						    execution_audit.maximum_controller_commands,
						    execution_audit.minimum_jacobian_singular_value,
						    execution_audit.maximum_jacobian_condition_number,
						    authored_roll, planned_roll,
						    joint_travel, j4_margin,
						    first_bad_segment, first_bad_tcp.x(), first_bad_tcp.y(),
						    first_bad_tcp.z());
						continue;
					}
					const bool score_better =
					    joint_travel < selected_pick_joint_travel - 1e-6 ||
					    (std::abs(joint_travel - selected_pick_joint_travel) <= 1e-6 &&
					     (j4_margin > selected_pick_j4_margin + 1e-6 ||
					      (std::abs(j4_margin - selected_pick_j4_margin) <= 1e-6 &&
					       candidate->cost() < selected_pick_solution->cost())));
					const bool prefer =
					    !selected_pick_solution ||
					    (branch.arm->execution_eligible &&
					     !planned_branches[selected_index].arm->execution_eligible) ||
					    (branch.arm->execution_eligible ==
					         planned_branches[selected_index].arm->execution_eligible &&
					     score_better);
					if (prefer)
					{
						selected_index = i;
						selected_pick_solution = candidate.get();
						selected_pick_trajectory = std::move(trajectory);
						selected_pick_joint_travel = joint_travel;
						selected_pick_j4_margin = j4_margin;
					}
				}
			}
		}
		else if (scenario.place_only)
		{
			if (!current_state_stage || current_state_stage->solutions().empty())
				throw std::runtime_error("place-only task has no CurrentState for TCP path audit");
			const auto& planned_start_state =
			    (*current_state_stage->solutions().begin())->end()->scene()->getCurrentState();
			selected_index = planned_branches.size();
			for (std::size_t i = 0; i < planned_branches.size(); ++i)
			{
				const auto& branch = planned_branches[i];
				if (!branch.stage)
					continue;
				for (const auto& candidate : branch.stage->solutions())
				{
					auto trajectory = exportPlaceTrajectory(scenario, *branch.arm, *candidate);
					const auto execution_audit = auditExecutionTrajectory(
					    *branch.arm, planned_start_state, trajectory.joint_names,
					    trajectory.points, { trajectory.release_point_index });
					const double authored_roll =
					    authoredFingerRollDeg(scenario.target_place_pose);
					const double planned_roll = plannedFingerRollDeg(
					    scenario, *branch.arm, planned_start_state, trajectory.joint_names,
					    trajectory.points.at(trajectory.release_point_index));
					const bool roll_safe = authored_roll <= 1e-3 &&
					                       planned_roll <= EXECUTION_MAX_FINGER_ROLL_DEG;
					const auto [transport_length, direct_distance] =
					    placeTransportTcpMetrics(*branch.arm, planned_start_state, trajectory);
					const bool transport_safe =
					    transport_length <= std::max(0.25, 3.0 * direct_distance) + 1e-9;
					bool j4_safe = true;
					double previous_j4 = 0.0;
					for (const auto& point : trajectory.points)
					{
						const double j4 = point.positions_deg.at(3);
						if (std::abs(j4) < EXECUTION_J4_SINGULARITY_DEG ||
						    previous_j4 * j4 < 0.0)
						{
							j4_safe = false;
							break;
						}
						previous_j4 = j4;
					}
					if (!execution_audit.safe() || !roll_safe || !j4_safe || !transport_safe)
					{
						RCLCPP_WARN(
						    LOGGER,
						    "rejecting %s place solution: execution_safe=%s "
						    "controller_safe=%s joint_bounds_safe=%s timing_safe=%s "
						    "jacobian_safe=%s roll_safe=%s j4_safe=%s transport_safe=%s commands=%zu "
						    "min_jacobian_sigma=%.5f max_jacobian_condition=%.2f "
						    "authored_roll=%.4fdeg planned_roll=%.4fdeg "
						    "tcp_path=%.3fm direct=%.3fm detour=%.3fx",
						    branch.branch_id.c_str(),
						    execution_audit.safe() ? "true" : "false",
						    execution_audit.controller_safe ? "true" : "false",
						    execution_audit.joint_bounds_safe ? "true" : "false",
						    execution_audit.timing_safe ? "true" : "false",
						    execution_audit.jacobian_safe ? "true" : "false",
						    roll_safe ? "true" : "false",
						    j4_safe ? "true" : "false",
						    transport_safe ? "true" : "false",
						    execution_audit.maximum_controller_commands,
						    execution_audit.minimum_jacobian_singular_value,
						    execution_audit.maximum_jacobian_condition_number,
						    authored_roll, planned_roll,
						    transport_length,
						    direct_distance,
						    transport_length / std::max(direct_distance, 1e-9));
						continue;
					}
					if (!selected_place_solution ||
					    transport_length < selected_place_transport_length - 1e-9 ||
					    (std::abs(transport_length - selected_place_transport_length) <= 1e-9 &&
					     candidate->cost() < selected_place_solution->cost()))
					{
						selected_index = i;
						selected_place_solution = candidate.get();
						selected_place_trajectory = std::move(trajectory);
						selected_place_transport_length = transport_length;
					}
				}
			}
		}
		else
		{
			if (!current_state_stage || current_state_stage->solutions().empty())
				throw std::runtime_error(
				    "full-transfer task has no CurrentState for execution audit");
			const auto& planned_start_state =
			    (*current_state_stage->solutions().begin())->end()->scene()->getCurrentState();
			selected_index = planned_branches.size();
			for (std::size_t i = 0; i < planned_branches.size(); ++i)
			{
				const auto& branch = planned_branches[i];
				if (!branch.stage)
					continue;
				for (const auto& candidate : branch.stage->solutions())
				{
					auto trajectory = exportFullTransferTrajectory(
					    scenario, *branch.arm, branch.grasp_candidate_id,
					    *candidate);
					const auto execution_audit = auditExecutionTrajectory(
					    *branch.arm, planned_start_state, trajectory.joint_names,
					    trajectory.points,
					    { trajectory.attach_point_index, trajectory.release_point_index });
					const auto grasp_candidate = std::find_if(
					    scenario.source_grasp_candidates.begin(),
					    scenario.source_grasp_candidates.end(),
					    [&branch](const auto& item) {
						    return item.id == branch.grasp_candidate_id;
					    });
					const double authored_grasp_roll =
					    grasp_candidate == scenario.source_grasp_candidates.end() ?
					        std::numeric_limits<double>::infinity() :
					        authoredFingerRollDeg(grasp_candidate->pose);
					const double planned_grasp_roll = plannedFingerRollDeg(
					    scenario, *branch.arm, planned_start_state, trajectory.joint_names,
					    trajectory.points.at(trajectory.attach_point_index));
					const double authored_place_roll =
					    authoredFingerRollDeg(scenario.target_place_pose);
					const double planned_place_roll = plannedFingerRollDeg(
					    scenario, *branch.arm, planned_start_state, trajectory.joint_names,
					    trajectory.points.at(trajectory.release_point_index));
					const bool roll_safe =
					    authored_grasp_roll <= 1e-3 && authored_place_roll <= 1e-3 &&
					    planned_grasp_roll <= EXECUTION_MAX_FINGER_ROLL_DEG &&
					    planned_place_roll <= EXECUTION_MAX_FINGER_ROLL_DEG;
					if (!execution_audit.safe() || !roll_safe)
					{
						RCLCPP_WARN(
						    LOGGER,
						    "rejecting %s full-transfer solution: controller_safe=%s "
						    "joint_bounds_safe=%s timing_safe=%s jacobian_safe=%s roll_safe=%s "
						    "commands=%zu min_jacobian_sigma=%.5f "
						    "max_jacobian_condition=%.2f grasp_roll=%.4f/%.4fdeg "
						    "place_roll=%.4f/%.4fdeg",
						    branch.branch_id.c_str(),
						    execution_audit.controller_safe ? "true" : "false",
						    execution_audit.joint_bounds_safe ? "true" : "false",
						    execution_audit.timing_safe ? "true" : "false",
						    execution_audit.jacobian_safe ? "true" : "false",
						    roll_safe ? "true" : "false",
						    execution_audit.maximum_controller_commands,
						    execution_audit.minimum_jacobian_singular_value,
						    execution_audit.maximum_jacobian_condition_number,
						    authored_grasp_roll, planned_grasp_roll,
						    authored_place_roll, planned_place_roll);
						continue;
					}
					const bool prefer =
					    !selected_full_solution ||
					    (branch.arm->execution_eligible &&
					     !planned_branches[selected_index].arm->execution_eligible) ||
					    (branch.arm->execution_eligible ==
					         planned_branches[selected_index].arm->execution_eligible &&
					     candidate->cost() < selected_full_solution->cost());
					if (prefer)
					{
						selected_index = i;
						selected_full_solution = candidate.get();
						selected_full_trajectory = std::move(trajectory);
					}
				}
			}
		}

		if (selected_index < run.arms.size())
		{
			const auto& selected = run.arms[selected_index];
			run.solved = true;
			run.selected_arm = selected.arm_id;
			run.selected_grasp_candidate = selected.grasp_candidate_id;
			run.selected_solution_id = selected.selected_solution_id;
			run.execution_eligible = selected.execution_eligible;
			run.execution_block_reason = selected.execution_block_reason;
			for (const auto& arm : arms)
				if (arm.arm_id == selected.arm_id)
				{
					run.tool_version = arm.tool_version;
					run.calibration_version = arm.calibration_version;
				}
			if (scenario.pick_only)
			{
				run.execution_eligible = false;
				run.execution_block_reason = selected.execution_eligible ?
				                                 "EXTERNAL_GATED_PICK_EXECUTOR_REQUIRED" :
				                                 selected.execution_block_reason;
				const auto& branch = planned_branches[selected_index];
				if (!selected_pick_solution)
					throw std::runtime_error(
					    "pick-only task has no execution-safe exportable solution");
				run.selected_solution_id = branch.branch_id + "#execution_safe";
				run.trajectory_export_path = options.result_path + ".trajectory.json";
				grabber_mtc::writePickTrajectoryJson(run.trajectory_export_path,
				                                     selected_pick_trajectory);
			}
			else if (scenario.place_only)
			{
				run.execution_eligible = false;
				run.execution_block_reason = "EXTERNAL_GATED_PLACE_EXECUTOR_REQUIRED";
				const auto& branch = planned_branches[selected_index];
				if (!selected_place_solution)
					throw std::runtime_error(
					    "place-only task has no execution-safe exportable solution");
				run.selected_solution_id = branch.branch_id + "#execution_safe";
				run.trajectory_export_path = options.result_path + ".trajectory.json";
				grabber_mtc::writePlaceTrajectoryJson(
				    run.trajectory_export_path, selected_place_trajectory);
			}
			else
			{
				if (!selected_full_solution)
					throw std::runtime_error(
					    "full-transfer task has no execution-safe exportable solution");
				run.execution_eligible = false;
				run.execution_block_reason = "PLAN_ONLY_FULL_TRANSFER";
				run.selected_solution_id =
				    planned_branches[selected_index].branch_id +
				    "#execution_safe";
				run.trajectory_export_path =
				    options.result_path + ".trajectory.json";
				grabber_mtc::writeFullTransferTrajectoryJson(
				    run.trajectory_export_path, selected_full_trajectory);
			}
		}
		else
		{
			run.execution_block_reason = "NO_COMPLETE_SOLUTION";
		}

		grabber_mtc::writeResultJson(options.result_path, run);
		RCLCPP_INFO(LOGGER, "%s", grabber_mtc::formatResultSummary(run).c_str());
		{
			// The stage tree with per-stage solution and failure counts is the
			// audit trail behind earliest_failure_stage; log it verbatim.
			std::ostringstream tree;
			tree << task;
			RCLCPP_INFO(LOGGER, "MTC stage tree:\n%s", tree.str().c_str());
		}
		RCLCPP_INFO(LOGGER, "plan-only run, no motion was commanded; result written to %s",
		            options.result_path.c_str());
		if (!run.solved && exit_code == 0)
			exit_code = 1;

		if (options.hold_seconds > 0.0)
		{
			RCLCPP_INFO(LOGGER, "holding %.0fs so the RViz Motion Planning Tasks panel can inspect the solutions",
			            options.hold_seconds);
			std::this_thread::sleep_for(std::chrono::duration<double>(options.hold_seconds));
		}
	}
	catch (const std::exception& e)
	{
		RCLCPP_ERROR(LOGGER, "%s", e.what());
		exit_code = 2;
	}

	executor.cancel();
	spinner.join();
	rclcpp::shutdown();
	return exit_code;
}
