#include "scenario.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace grabber_mtc
{
namespace
{

[[noreturn]] void fail(const std::string& what)
{
	throw std::runtime_error(what);
}

const YAML::Node require(const YAML::Node& parent, const std::string& key, const std::string& where)
{
	if (!parent[key])
		fail("scenario field missing: " + where + "." + key);
	return parent[key];
}

std::array<double, 3> readXyz(const YAML::Node& node, const std::string& where)
{
	if (!node.IsSequence() || node.size() != 3)
		fail(where + " must be a 3-element sequence");
	return { { node[0].as<double>(), node[1].as<double>(), node[2].as<double>() } };
}

geometry_msgs::msg::Vector3 readVector3(const YAML::Node& node, const std::string& where)
{
	const auto v = readXyz(node, where);
	geometry_msgs::msg::Vector3 out;
	out.x = v[0];
	out.y = v[1];
	out.z = v[2];
	const double norm = std::sqrt(out.x * out.x + out.y * out.y + out.z * out.z);
	if (norm < 1e-9)
		fail(where + " must not be a zero vector");
	out.x /= norm;
	out.y /= norm;
	out.z /= norm;
	return out;
}

/// Accepts {xyz: [...], quat_xyzw: [...]} or {xyz: [...], rpy_deg: [...]}.
geometry_msgs::msg::Pose readPose(const YAML::Node& node, const std::string& where)
{
	geometry_msgs::msg::Pose pose;
	const auto xyz = readXyz(require(node, "xyz", where), where + ".xyz");
	pose.position.x = xyz[0];
	pose.position.y = xyz[1];
	pose.position.z = xyz[2];

	if (node["quat_xyzw"])
	{
		const auto& q = node["quat_xyzw"];
		if (!q.IsSequence() || q.size() != 4)
			fail(where + ".quat_xyzw must be a 4-element sequence");
		pose.orientation.x = q[0].as<double>();
		pose.orientation.y = q[1].as<double>();
		pose.orientation.z = q[2].as<double>();
		pose.orientation.w = q[3].as<double>();
		const double norm = std::sqrt(pose.orientation.x * pose.orientation.x +
		                              pose.orientation.y * pose.orientation.y +
		                              pose.orientation.z * pose.orientation.z +
		                              pose.orientation.w * pose.orientation.w);
		if (norm < 1e-6)
			fail(where + ".quat_xyzw is degenerate");
		pose.orientation.x /= norm;
		pose.orientation.y /= norm;
		pose.orientation.z /= norm;
		pose.orientation.w /= norm;
		return pose;
	}

	std::array<double, 3> rpy{ { 0.0, 0.0, 0.0 } };
	if (node["rpy_deg"])
		rpy = readXyz(node["rpy_deg"], where + ".rpy_deg");
	else if (!node["quat_xyzw"])
		fail(where + " needs either quat_xyzw or rpy_deg");
	const double deg = M_PI / 180.0;
	const double cr = std::cos(rpy[0] * deg / 2), sr = std::sin(rpy[0] * deg / 2);
	const double cp = std::cos(rpy[1] * deg / 2), sp = std::sin(rpy[1] * deg / 2);
	const double cy = std::cos(rpy[2] * deg / 2), sy = std::sin(rpy[2] * deg / 2);
	pose.orientation.w = cr * cp * cy + sr * sp * sy;
	pose.orientation.x = sr * cp * cy - cr * sp * sy;
	pose.orientation.y = cr * sp * cy + sr * cp * sy;
	pose.orientation.z = cr * cp * sy - sr * sp * cy;
	return pose;
}

Eigen::Isometry3d readRigidTransform(const YAML::Node& node, const std::string& where)
{
	if (!node.IsSequence() || node.size() != 4)
		fail(where + " must be a 4x4 matrix");
	Eigen::Matrix4d matrix;
	for (std::size_t row = 0; row < 4; ++row)
	{
		if (!node[row].IsSequence() || node[row].size() != 4)
			fail(where + " must be a 4x4 matrix");
		for (std::size_t col = 0; col < 4; ++col)
			matrix(row, col) = node[row][col].as<double>();
	}
	if (!matrix.allFinite())
		fail(where + " must contain only finite values");
	if (!matrix.row(3).isApprox(Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-9))
		fail(where + " last row must be [0, 0, 0, 1]");
	const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
	if (!(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
	    std::abs(rotation.determinant() - 1.0) > 1e-6)
		fail(where + " rotation must be orthonormal with determinant +1");
	Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
	transform.matrix() = matrix;
	return transform;
}

std::string jsonEscape(const std::string& in)
{
	std::string out;
	out.reserve(in.size() + 8);
	for (char c : in)
	{
		switch (c)
		{
			case '"':
				out += "\\\"";
				break;
			case '\\':
				out += "\\\\";
				break;
			case '\n':
				out += "\\n";
				break;
			default:
				out += c;
		}
	}
	return out;
}

std::string jsonString(const std::string& in)
{
	return "\"" + jsonEscape(in) + "\"";
}

std::string jsonPose(const geometry_msgs::msg::Pose& p)
{
	std::ostringstream os;
	os << std::setprecision(9) << "{\"xyz\": [" << p.position.x << ", " << p.position.y << ", " << p.position.z
	   << "], \"quat_xyzw\": [" << p.orientation.x << ", " << p.orientation.y << ", " << p.orientation.z << ", "
	   << p.orientation.w << "]}";
	return os.str();
}

}  // namespace

Scenario loadScenario(const std::string& path)
{
	YAML::Node root;
	try
	{
		root = YAML::LoadFile(path);
	}
	catch (const std::exception& e)
	{
		fail("cannot read scenario " + path + ": " + e.what());
	}

	Scenario s;
	s.scenario_id = require(root, "scenario_id", "scenario").as<std::string>();
	s.frame_id = require(root, "frame_id", "scenario").as<std::string>();
	const std::string mode = root["mode"].as<std::string>("full_transfer");
	if (mode != "full_transfer" && mode != "pick_only" && mode != "place_only")
		fail("scenario.mode must be 'full_transfer', 'pick_only', or 'place_only'");
	s.pick_only = mode == "pick_only";
	s.place_only = mode == "place_only";
	s.planning_arm_id = root["planning_arm_id"].as<std::string>("");
	s.source_layer_id = root["source_layer_id"].as<std::string>("");
	s.target_layer_id = root["target_layer_id"].as<std::string>("");
	s.lift_state_id = root["lift_state_id"].as<std::string>("");
	s.source_support_surface_id = root["source_support_surface_id"].as<std::string>("");
	s.target_support_surface_id = root["target_support_surface_id"].as<std::string>("");

	s.source_grasp_pose = readPose(require(root, "source_grasp_pose", "scenario"), "source_grasp_pose");
	if (const auto& candidates = root["source_grasp_candidates"])
	{
		for (std::size_t i = 0; i < candidates.size(); ++i)
		{
			const std::string where = "source_grasp_candidates[" + std::to_string(i) + "]";
			GraspCandidate candidate;
			candidate.id = require(candidates[i], "id", where).as<std::string>();
			candidate.pose = readPose(require(candidates[i], "pose", where), where + ".pose");
			if (candidate.id.empty())
				fail(where + ".id must not be empty");
			for (const auto& existing : s.source_grasp_candidates)
				if (existing.id == candidate.id)
					fail("duplicate source_grasp_candidates id: " + candidate.id);
			s.source_grasp_candidates.push_back(std::move(candidate));
		}
	}
	if (s.source_grasp_candidates.empty())
		s.source_grasp_candidates.push_back({ "primary", s.source_grasp_pose });
	s.target_place_pose = readPose(require(root, "target_place_pose", "scenario"), "target_place_pose");

	s.source_approach_direction =
	    readVector3(require(root, "source_approach_direction", "scenario"), "source_approach_direction");
	if (root["source_lift_direction"])
		s.source_lift_direction = readVector3(root["source_lift_direction"], "source_lift_direction");
	else
		s.source_lift_direction.z = 1.0;
	s.source_retreat_direction =
	    readVector3(require(root, "source_retreat_direction", "scenario"), "source_retreat_direction");
	s.target_insert_direction =
	    readVector3(require(root, "target_insert_direction", "scenario"), "target_insert_direction");
	s.target_retreat_direction =
	    readVector3(require(root, "target_retreat_direction", "scenario"), "target_retreat_direction");

	s.source_pregrasp_offset_m = root["source_pregrasp_offset_m"].as<double>(s.source_pregrasp_offset_m);
	s.source_contact_distance_m = root["source_contact_distance_m"].as<double>(s.source_contact_distance_m);
	s.source_lift_distance_m = root["source_lift_distance_m"].as<double>(s.source_lift_distance_m);
	s.source_retreat_distance_m = root["source_retreat_distance_m"].as<double>(s.source_retreat_distance_m);
	s.target_preplace_offset_m = root["target_preplace_offset_m"].as<double>(s.target_preplace_offset_m);
	s.target_contact_distance_m =
	    root["target_contact_distance_m"].as<double>(s.target_contact_distance_m);
	s.target_retreat_distance_m = root["target_retreat_distance_m"].as<double>(s.target_retreat_distance_m);
	s.cartesian_min_fraction = root["cartesian_min_fraction"].as<double>(s.cartesian_min_fraction);
	s.cartesian_transport = root["cartesian_transport"].as<bool>(s.cartesian_transport);
	if (const auto& carry = root["post_pick_carry_joints_deg"])
	{
		s.post_pick_carry_joints_deg = carry.as<std::vector<double>>();
		if (s.post_pick_carry_joints_deg.size() != 7 ||
		    !std::all_of(s.post_pick_carry_joints_deg.begin(),
		                 s.post_pick_carry_joints_deg.end(),
		                 [](double value) { return std::isfinite(value); }))
			fail("post_pick_carry_joints_deg must contain seven finite degrees");
	}
	if (const auto& weights = root["planning_joint_weights"])
	{
		s.planning_joint_weights = weights.as<std::vector<double>>();
		if (s.planning_joint_weights.size() != 7 ||
		    !std::all_of(s.planning_joint_weights.begin(), s.planning_joint_weights.end(),
		                 [](double value) { return std::isfinite(value) && value > 0.0; }))
			fail("planning_joint_weights must contain seven positive finite weights");
	}
	if (const auto& reference = root["source_grasp_reference_joints_deg"])
	{
		s.source_grasp_reference_joints_deg = reference.as<std::vector<double>>();
		if (s.source_grasp_reference_joints_deg.size() != 7 ||
		    !std::all_of(s.source_grasp_reference_joints_deg.begin(),
		                 s.source_grasp_reference_joints_deg.end(),
		                 [](double value) { return std::isfinite(value); }))
			fail("source_grasp_reference_joints_deg must contain seven finite degrees");
	}
	if (const auto& home = root["post_place_home_joints_deg"])
	{
		s.post_place_home_joints_deg = home.as<std::vector<double>>();
		if (s.post_place_home_joints_deg.size() != 7 ||
		    !std::all_of(s.post_place_home_joints_deg.begin(),
		                 s.post_place_home_joints_deg.end(),
		                 [](double value) { return std::isfinite(value); }))
			fail("post_place_home_joints_deg must contain seven finite degrees");
	}
	if (!std::isfinite(s.source_contact_distance_m) || s.source_contact_distance_m <= 0.0 ||
	    s.source_contact_distance_m >= s.source_pregrasp_offset_m)
		fail("source_contact_distance_m must be positive and shorter than source_pregrasp_offset_m");

	if (const auto& bottle = root["bottle"])
	{
		s.bottle_id = bottle["id"].as<std::string>(s.bottle_id);
		s.bottle_radius_m = bottle["radius_m"].as<double>(s.bottle_radius_m);
		s.bottle_height_m = bottle["height_m"].as<double>(s.bottle_height_m);
		s.bottle_pose = bottle["pose"] ? readPose(bottle["pose"], "bottle.pose") : s.source_grasp_pose;
	}
	else
	{
		s.bottle_pose = s.source_grasp_pose;
	}

	if (const auto& boxes = root["shelf_boxes"])
	{
		for (std::size_t i = 0; i < boxes.size(); ++i)
		{
			const std::string where = "shelf_boxes[" + std::to_string(i) + "]";
			BoxObject box;
			box.id = require(boxes[i], "id", where).as<std::string>();
			box.size = readXyz(require(boxes[i], "size", where), where + ".size");
			box.pose = readPose(require(boxes[i], "pose", where), where + ".pose");
			s.shelf_boxes.push_back(std::move(box));
		}
	}
	if (const auto& workspace = root["tcp_path_workspace"])
	{
		s.tcp_path_workspace.id =
		    require(workspace, "id", "tcp_path_workspace").as<std::string>();
		s.tcp_path_workspace.size = readXyz(
		    require(workspace, "size", "tcp_path_workspace"),
		    "tcp_path_workspace.size");
		s.tcp_path_workspace.pose = readPose(
		    require(workspace, "pose", "tcp_path_workspace"),
		    "tcp_path_workspace.pose");
		if (s.tcp_path_workspace.id.empty() ||
		    !std::all_of(
		        s.tcp_path_workspace.size.begin(),
		        s.tcp_path_workspace.size.end(),
		        [](double value) { return std::isfinite(value) && value > 0.0; }))
			fail("tcp_path_workspace must have a non-empty id and positive finite size");
		s.has_tcp_path_workspace = true;
	}
	s.dynamic_obstacle_id = root["dynamic_obstacle_id"].as<std::string>(s.dynamic_obstacle_id);
	s.obstacle_voxel_size_m = root["obstacle_voxel_size_m"].as<double>(s.obstacle_voxel_size_m);
	if (const auto& voxels = root["obstacle_voxels"])
		for (std::size_t i = 0; i < voxels.size(); ++i)
			s.obstacle_voxels.push_back(readXyz(voxels[i], "obstacle_voxels[" + std::to_string(i) + "]"));
	if (!std::isfinite(s.obstacle_voxel_size_m) || s.obstacle_voxel_size_m <= 0.0)
		fail("obstacle_voxel_size_m must be a positive finite value");
	if (s.dynamic_obstacle_id.empty() || s.dynamic_obstacle_id == s.bottle_id)
		fail("dynamic_obstacle_id must be non-empty and distinct from bottle.id");
	for (const auto& box : s.shelf_boxes)
		if (box.id == s.dynamic_obstacle_id || box.id == s.bottle_id)
			fail("collision object ids for bottle, shelf boxes and dynamic obstacles must be distinct");
	s.spawn_scene_objects = root["spawn_scene_objects"].as<bool>(s.spawn_scene_objects);
	if (s.spawn_scene_objects)
	{
		const auto validate_support = [&s](const std::string& id, const char* field) {
			if (id.empty())
				return;
			for (const auto& box : s.shelf_boxes)
				if (box.id == id)
					return;
			fail(std::string(field) + " '" + id + "' is not present in shelf_boxes");
		};
		validate_support(s.source_support_surface_id, "source_support_surface_id");
		validate_support(s.target_support_surface_id, "target_support_surface_id");
	}
	s.start_state_source = root["start_state_source"].as<std::string>(s.start_state_source);
	if (s.start_state_source != "current_state")
		fail("start_state_source only supports 'current_state'; refusing to fake a start state");
	s.scene_version = root["scene_version"].as<std::string>("");
	s.target_captured_at_utc = root["target_captured_at_utc"].as<std::string>("");
	s.scene_captured_at_utc = root["scene_captured_at_utc"].as<std::string>("");
	s.freshness_max_age_s = root["freshness_max_age_s"].as<double>(s.freshness_max_age_s);
	s.fixture_source = root["fixture_source"].as<bool>(s.fixture_source);
	const bool simulation_source = root["simulation_source"].as<bool>(false);
	if (simulation_source && !s.fixture_source)
		fail("simulation_source scenarios must remain fixture_source");
	if (s.pick_only)
	{
		if (s.planning_arm_id.empty())
			fail("pick_only scenario requires planning_arm_id");
		if (s.source_grasp_candidates.size() < 2)
			fail("pick_only scenario requires at least two source_grasp_candidates");
		if (s.fixture_source && !simulation_source)
			fail("pick_only scenario must come from fresh perception, not a fixture");
		if (!root["obstacle_voxels"] || s.shelf_boxes.empty() || s.source_support_surface_id.empty())
			fail("pick_only scenario requires the live obstacle field and measured support geometry");
		if (s.target_captured_at_utc.empty() || s.scene_captured_at_utc.empty() ||
		    !std::isfinite(s.freshness_max_age_s) || s.freshness_max_age_s <= 0.0)
			fail("pick_only scenario requires explicit target/scene freshness");
		if (!root["source_lift_direction"] || !root["source_lift_distance_m"] ||
		    !std::isfinite(s.source_lift_distance_m) || s.source_lift_distance_m <= 0.0)
			fail("pick_only scenario requires a positive source lift");
		if (!s.has_tcp_path_workspace)
			fail("pick_only scenario requires tcp_path_workspace");
		const double retreat_dot =
		    s.source_approach_direction.x * s.source_retreat_direction.x +
		    s.source_approach_direction.y * s.source_retreat_direction.y +
		    s.source_approach_direction.z * s.source_retreat_direction.z;
		if (retreat_dot > -0.99)
			fail("pick_only source_retreat_direction must reverse the source approach");
	}
	if (s.place_only)
	{
		if (s.planning_arm_id.empty())
			fail("place_only scenario requires planning_arm_id");
		if (s.planning_arm_id != "right_arm")
			fail("place_only first version supports only right_arm");
		if (s.fixture_source && !simulation_source)
			fail("place_only scenario must come from fresh perception, not a fixture");
		if (!root["obstacle_voxels"] || s.shelf_boxes.empty() ||
		    s.target_support_surface_id.empty())
			fail("place_only scenario requires live obstacles and target support geometry");
		if (s.scene_captured_at_utc.empty() || !std::isfinite(s.freshness_max_age_s) ||
		    s.freshness_max_age_s <= 0.0)
			fail("place_only scenario requires explicit scene freshness");
		if (!std::isfinite(s.target_contact_distance_m) ||
		    s.target_contact_distance_m <= 0.0 ||
		    s.target_contact_distance_m >= s.target_preplace_offset_m)
			fail("target_contact_distance_m must be positive and shorter than target_preplace_offset_m");
		if (s.post_place_home_joints_deg.empty())
			fail("place_only scenario requires post_place_home_joints_deg");
	}

	s.planner_id = root["planner_id"].as<std::string>(s.planner_id);
	s.local_motion_planner =
	    root["local_motion_planner"].as<std::string>(s.local_motion_planner);
	if (s.local_motion_planner != "cartesian" &&
	    s.local_motion_planner != "pilz_lin")
		fail("local_motion_planner must be 'cartesian' or 'pilz_lin'");
	s.planning_timeout_s = root["planning_timeout_s"].as<double>(s.planning_timeout_s);
	s.max_ik_solutions = root["max_ik_solutions"].as<std::size_t>(s.max_ik_solutions);
	s.max_solutions = root["max_solutions"].as<std::size_t>(s.max_solutions);
	return s;
}

std::vector<ArmConfig> loadArmConfigs(const std::string& path)
{
	YAML::Node root;
	try
	{
		root = YAML::LoadFile(path);
	}
	catch (const std::exception& e)
	{
		fail("cannot read arm config " + path + ": " + e.what());
	}

	const YAML::Node arms = require(root, "arms", "arm_config");
	std::vector<ArmConfig> out;
	for (std::size_t i = 0; i < arms.size(); ++i)
	{
		const YAML::Node& node = arms[i];
		const std::string where = "arms[" + std::to_string(i) + "]";
		ArmConfig cfg;
		cfg.arm_id = require(node, "arm_id", where).as<std::string>();
		cfg.planning_group = require(node, "planning_group", where).as<std::string>();
		cfg.ik_link = require(node, "ik_link", where).as<std::string>();
		cfg.end_effector = node["end_effector"].as<std::string>("");
		cfg.hand_group = node["hand_group"].as<std::string>("");
		if (const auto& links = node["touch_links"])
			for (std::size_t j = 0; j < links.size(); ++j)
				cfg.touch_links.push_back(links[j].as<std::string>());
		if (cfg.touch_links.empty())
			fail(where + ".touch_links must name the links allowed to contact the bottle");
		cfg.home_state = node["home_state"].as<std::string>("");
		cfg.tool_id = node["tool_id"].as<std::string>("");
		cfg.tool_version = node["tool_version"].as<std::string>("");
		cfg.calibration_version = node["calibration_version"].as<std::string>("");
		cfg.tcp_transform_from_ik_link =
		    readRigidTransform(require(node, "tcp_transform_from_ik_link", where),
		                       where + ".tcp_transform_from_ik_link");
		cfg.execution_eligible = node["execution_eligible"].as<bool>(false);
		cfg.execution_block_reason = node["execution_block_reason"].as<std::string>("");
		if (!cfg.execution_eligible && cfg.execution_block_reason.empty())
			fail(where + " is not execution_eligible but gives no execution_block_reason");
		out.push_back(std::move(cfg));
	}
	if (out.size() < 2)
		fail("arm config must describe both arms; a single-arm config cannot answer arm selection");
	return out;
}

std::vector<moveit_msgs::msg::CollisionObject> buildSceneObjects(const Scenario& scenario)
{
	std::vector<moveit_msgs::msg::CollisionObject> objects;

	moveit_msgs::msg::CollisionObject bottle;
	bottle.id = scenario.bottle_id;
	bottle.header.frame_id = scenario.frame_id;
	bottle.operation = moveit_msgs::msg::CollisionObject::ADD;
	shape_msgs::msg::SolidPrimitive cylinder;
	cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
	cylinder.dimensions = { scenario.bottle_height_m, scenario.bottle_radius_m };
	bottle.primitives.push_back(cylinder);
	bottle.primitive_poses.push_back(scenario.bottle_pose);
	objects.push_back(std::move(bottle));

	for (const auto& box : scenario.shelf_boxes)
	{
		moveit_msgs::msg::CollisionObject obj;
		obj.id = box.id;
		obj.header.frame_id = scenario.frame_id;
		obj.operation = moveit_msgs::msg::CollisionObject::ADD;
		shape_msgs::msg::SolidPrimitive prim;
		prim.type = shape_msgs::msg::SolidPrimitive::BOX;
		prim.dimensions = { box.size[0], box.size[1], box.size[2] };
		obj.primitives.push_back(prim);
		obj.primitive_poses.push_back(box.pose);
		objects.push_back(std::move(obj));
	}
	if (!scenario.obstacle_voxels.empty())
	{
		moveit_msgs::msg::CollisionObject obj;
		obj.id = scenario.dynamic_obstacle_id;
		obj.header.frame_id = scenario.frame_id;
		obj.operation = moveit_msgs::msg::CollisionObject::ADD;
		for (const auto& center : scenario.obstacle_voxels)
		{
			shape_msgs::msg::SolidPrimitive prim;
			prim.type = shape_msgs::msg::SolidPrimitive::BOX;
			prim.dimensions = { scenario.obstacle_voxel_size_m, scenario.obstacle_voxel_size_m,
				                scenario.obstacle_voxel_size_m };
			geometry_msgs::msg::Pose pose;
			pose.position.x = center[0];
			pose.position.y = center[1];
			pose.position.z = center[2];
			pose.orientation.w = 1.0;
			obj.primitives.push_back(std::move(prim));
			obj.primitive_poses.push_back(std::move(pose));
		}
		objects.push_back(std::move(obj));
	}
	return objects;
}

void writeResultJson(const std::string& path, const RunResult& r)
{
	std::ofstream os(path);
	if (!os)
		fail("cannot write result json: " + path);
	os << std::setprecision(9);
	os << "{\n";
	os << "  \"scenario_id\": " << jsonString(r.scenario_id) << ",\n";
	os << "  \"mode\": " << jsonString(r.mode) << ",\n";
	os << "  \"plan_only\": true,\n";
	os << "  \"solved\": " << (r.solved ? "true" : "false") << ",\n";
	os << "  \"selected_arm\": " << jsonString(r.selected_arm) << ",\n";
	os << "  \"selected_grasp_candidate\": " << jsonString(r.selected_grasp_candidate) << ",\n";
	os << "  \"selected_solution_id\": " << jsonString(r.selected_solution_id) << ",\n";
	os << "  \"trajectory_export_path\": " << jsonString(r.trajectory_export_path) << ",\n";
	os << "  \"execution_eligible\": " << (r.execution_eligible ? "true" : "false") << ",\n";
	os << "  \"execution_block_reason\": " << jsonString(r.execution_block_reason) << ",\n";
	os << "  \"fixture_source\": " << (r.fixture_source ? "true" : "false") << ",\n";
	os << "  \"planning_wall_time\": " << r.planning_wall_time_s << ",\n";
	os << "  \"robot_model_name\": " << jsonString(r.robot_model_name) << ",\n";

	os << "  \"planning_group_names\": [";
	for (std::size_t i = 0; i < r.planning_group_names.size(); ++i)
		os << (i ? ", " : "") << jsonString(r.planning_group_names[i]);
	os << "],\n";

	os << "  \"start_state\": {\n";
	os << "    \"source\": \"mtc stages::CurrentState\",\n";
	os << "    \"all_zero\": " << (r.start_state_all_zero ? "true" : "false") << ",\n";
	os << "    \"selected_arm\": " << jsonString(r.start_state_selected_arm) << ",\n";
	os << "    \"selected_arm_complete\": "
	   << (r.start_state_selected_arm_complete ? "true" : "false") << ",\n";
	os << "    \"joint_state_stamp_ns\": " << r.start_state_joint_state_stamp_ns << ",\n";
	os << "    \"joint_state_age_s_at_planning\": "
	   << r.start_state_joint_state_age_s_at_planning << ",\n";
	os << "    \"joints\": {";
	for (std::size_t i = 0; i < r.start_state_joints.size(); ++i)
		os << (i ? ", " : "") << jsonString(r.start_state_joints[i].first) << ": " << r.start_state_joints[i].second;
	os << "}\n  },\n";

	os << "  \"scene_version\": " << jsonString(r.scene_version) << ",\n";
	os << "  \"tool_version\": " << jsonString(r.tool_version) << ",\n";
	os << "  \"calibration_version\": " << jsonString(r.calibration_version) << ",\n";

	const auto armMap = [&os, &r](const char* name, auto&& value) {
		os << "  \"" << name << "\": {";
		for (std::size_t i = 0; i < r.arms.size(); ++i)
			os << (i ? ", " : "") << jsonString(r.arms[i].branch_id) << ": " << value(r.arms[i]);
		os << "},\n";
	};
	armMap("complete_solution_count_by_arm", [](const ArmResult& a) { return std::to_string(a.complete_solution_count); });
	armMap("earliest_failure_stage_by_arm",
	       [](const ArmResult& a) { return a.earliest_failure_stage.empty() ? std::string("null") : jsonString(a.earliest_failure_stage); });
	armMap("total_cost_by_arm", [](const ArmResult& a) {
		if (a.best_total_cost < 0.0)
			return std::string("null");
		std::ostringstream tmp;
		tmp << std::setprecision(9) << a.best_total_cost;
		return tmp.str();
	});
	armMap("solved_by_arm", [](const ArmResult& a) { return std::string(a.solved ? "true" : "false"); });
	armMap("execution_eligible_by_arm", [](const ArmResult& a) { return std::string(a.execution_eligible ? "true" : "false"); });
	armMap("execution_block_reason_by_arm", [](const ArmResult& a) { return jsonString(a.execution_block_reason); });

	os << "  \"stage_costs\": {\n";
	for (std::size_t i = 0; i < r.arms.size(); ++i)
	{
		os << "    " << jsonString(r.arms[i].branch_id) << ": {";
		const auto& stages = r.arms[i].stage_costs;
		for (std::size_t j = 0; j < stages.size(); ++j)
		{
			os << (j ? ", " : "") << jsonString(stages[j].first) << ": ";
			if (stages[j].second < 0.0)
				os << "null";
			else
				os << stages[j].second;
		}
		os << "}" << (i + 1 < r.arms.size() ? "," : "") << "\n";
	}
	os << "  },\n";

	if (r.scenario)
	{
		os << "  \"source_pose\": " << jsonPose(r.scenario->source_grasp_pose) << ",\n";
		os << "  \"target_pose\": " << jsonPose(r.scenario->target_place_pose) << ",\n";
		os << "  \"frame_id\": " << jsonString(r.scenario->frame_id) << "\n";
	}
	else
	{
		os << "  \"source_pose\": null,\n  \"target_pose\": null,\n  \"frame_id\": null\n";
	}
	os << "}\n";
}

void writePickTrajectoryJson(const std::string& path, const PickTrajectoryExport& t)
{
	std::ofstream os(path);
	if (!os)
		fail("cannot write pick trajectory json: " + path);
	os << std::setprecision(9);
	os << "{\n";
	os << "  \"schema_version\": \"grabber.mtc_pick.v2\",\n";
	os << "  \"plan_only\": true,\n";
	os << "  \"execution_supported\": false,\n";
	os << "  \"execution_block_reason\": " << jsonString(t.execution_block_reason) << ",\n";
	os << "  \"mode\": \"pick_only\",\n";
	os << "  \"scenario_id\": " << jsonString(t.scenario_id) << ",\n";
	os << "  \"arm_id\": " << jsonString(t.arm_id) << ",\n";
	os << "  \"grasp_candidate_id\": " << jsonString(t.grasp_candidate_id) << ",\n";
	os << "  \"target_captured_at_utc\": " << jsonString(t.target_captured_at_utc) << ",\n";
	os << "  \"scene_captured_at_utc\": " << jsonString(t.scene_captured_at_utc) << ",\n";
	os << "  \"freshness_max_age_s\": " << t.freshness_max_age_s << ",\n";
	os << "  \"joint_units\": \"degrees\",\n";
	os << "  \"joint_names\": [";
	for (std::size_t i = 0; i < t.joint_names.size(); ++i)
		os << (i ? ", " : "") << jsonString(t.joint_names[i]);
	os << "],\n";
	os << "  \"points\": [\n";
	for (std::size_t i = 0; i < t.points.size(); ++i)
	{
		os << "    {\"time_from_start_s\": " << t.points[i].time_from_start_s << ", \"positions_deg\": [";
		for (std::size_t j = 0; j < t.points[i].positions_deg.size(); ++j)
			os << (j ? ", " : "") << t.points[i].positions_deg[j];
		os << "], \"velocities_deg_s\": [";
		for (std::size_t j = 0; j < t.points[i].velocities_deg_s.size(); ++j)
			os << (j ? ", " : "") << t.points[i].velocities_deg_s[j];
		os << "], \"accelerations_deg_s2\": [";
		for (std::size_t j = 0; j < t.points[i].accelerations_deg_s2.size(); ++j)
			os << (j ? ", " : "") << t.points[i].accelerations_deg_s2[j];
		os << "]}" << (i + 1 < t.points.size() ? "," : "") << "\n";
	}
	os << "  ],\n";
	os << "  \"phase_boundaries\": [\n";
	for (std::size_t i = 0; i < t.phases.size(); ++i)
	{
		os << "    {\"name\": " << jsonString(t.phases[i].name) << ", \"start_index\": "
		   << t.phases[i].start_index << ", \"end_index\": " << t.phases[i].end_index << "}"
		   << (i + 1 < t.phases.size() ? "," : "") << "\n";
	}
	os << "  ],\n";
	os << "  \"gripper_events\": [\n";
	os << "    {\"name\": \"open_before_motion\", \"point_index\": 0, "
	      "\"operation\": \"RobotSession.open_gripper\", \"feedback_required\": true},\n";
	os << "    {\"name\": \"close_at_attach\", \"point_index\": " << t.attach_point_index
	   << ", \"operation\": \"RobotSession.close_gripper\", \"feedback_required\": true}\n";
	os << "  ]\n";
	os << "}\n";
}

void writePlaceTrajectoryJson(const std::string& path, const PlaceTrajectoryExport& t)
{
	std::ofstream os(path);
	if (!os)
		fail("cannot write place trajectory json: " + path);
	os << std::setprecision(9);
	os << "{\n";
	os << "  \"schema_version\": \"grabber.mtc_place.v1\",\n";
	os << "  \"plan_only\": true,\n";
	os << "  \"execution_supported\": false,\n";
	os << "  \"execution_block_reason\": \"EXTERNAL_GATED_PLACE_EXECUTOR_REQUIRED\",\n";
	os << "  \"mode\": \"place_only\",\n";
	os << "  \"scenario_id\": " << jsonString(t.scenario_id) << ",\n";
	os << "  \"arm_id\": " << jsonString(t.arm_id) << ",\n";
	os << "  \"scene_captured_at_utc\": " << jsonString(t.scene_captured_at_utc) << ",\n";
	os << "  \"freshness_max_age_s\": " << t.freshness_max_age_s << ",\n";
	os << "  \"joint_units\": \"degrees\",\n";
	os << "  \"joint_names\": [";
	for (std::size_t i = 0; i < t.joint_names.size(); ++i)
		os << (i ? ", " : "") << jsonString(t.joint_names[i]);
	os << "],\n  \"points\": [\n";
	for (std::size_t i = 0; i < t.points.size(); ++i)
	{
		os << "    {\"time_from_start_s\": " << t.points[i].time_from_start_s
		   << ", \"positions_deg\": [";
		for (std::size_t j = 0; j < t.points[i].positions_deg.size(); ++j)
			os << (j ? ", " : "") << t.points[i].positions_deg[j];
		os << "], \"velocities_deg_s\": [";
		for (std::size_t j = 0; j < t.points[i].velocities_deg_s.size(); ++j)
			os << (j ? ", " : "") << t.points[i].velocities_deg_s[j];
		os << "], \"accelerations_deg_s2\": [";
		for (std::size_t j = 0; j < t.points[i].accelerations_deg_s2.size(); ++j)
			os << (j ? ", " : "") << t.points[i].accelerations_deg_s2[j];
		os << "]}" << (i + 1 < t.points.size() ? "," : "") << "\n";
	}
	os << "  ],\n  \"phase_boundaries\": [\n";
	for (std::size_t i = 0; i < t.phases.size(); ++i)
	{
		os << "    {\"name\": " << jsonString(t.phases[i].name)
		   << ", \"start_index\": " << t.phases[i].start_index
		   << ", \"end_index\": " << t.phases[i].end_index << "}"
		   << (i + 1 < t.phases.size() ? "," : "") << "\n";
	}
	os << "  ],\n  \"gripper_events\": [\n";
	os << "    {\"name\": \"hold_before_motion\", \"point_index\": 0, "
	      "\"operation\": \"validate_holding_gripper_feedback\", \"feedback_required\": true},\n";
	os << "    {\"name\": \"open_at_release\", \"point_index\": "
	   << t.release_point_index
	   << ", \"operation\": \"RobotSession.open_gripper\", \"feedback_required\": true}\n";
	os << "  ]\n}\n";
}

void writeFullTransferTrajectoryJson(const std::string& path,
                                     const FullTransferTrajectoryExport& t)
{
	std::ofstream os(path);
	if (!os)
		fail("cannot write full-transfer trajectory json: " + path);
	os << std::setprecision(9);
	os << "{\n";
	os << "  \"schema_version\": \"grabber.mtc_full_transfer.v1\",\n";
	os << "  \"plan_only\": true,\n";
	os << "  \"execution_supported\": false,\n";
	os << "  \"execution_block_reason\": \"PLAN_ONLY_FULL_TRANSFER\",\n";
	os << "  \"mode\": \"full_transfer\",\n";
	os << "  \"scenario_id\": " << jsonString(t.scenario_id) << ",\n";
	os << "  \"arm_id\": " << jsonString(t.arm_id) << ",\n";
	os << "  \"grasp_candidate_id\": " << jsonString(t.grasp_candidate_id) << ",\n";
	os << "  \"joint_units\": \"degrees\",\n";
	os << "  \"joint_names\": [";
	for (std::size_t i = 0; i < t.joint_names.size(); ++i)
		os << (i ? ", " : "") << jsonString(t.joint_names[i]);
	os << "],\n  \"points\": [\n";
	for (std::size_t i = 0; i < t.points.size(); ++i)
	{
		os << "    {\"time_from_start_s\": " << t.points[i].time_from_start_s
		   << ", \"positions_deg\": [";
		for (std::size_t j = 0; j < t.points[i].positions_deg.size(); ++j)
			os << (j ? ", " : "") << t.points[i].positions_deg[j];
		os << "], \"velocities_deg_s\": [";
		for (std::size_t j = 0; j < t.points[i].velocities_deg_s.size(); ++j)
			os << (j ? ", " : "") << t.points[i].velocities_deg_s[j];
		os << "], \"accelerations_deg_s2\": [";
		for (std::size_t j = 0; j < t.points[i].accelerations_deg_s2.size(); ++j)
			os << (j ? ", " : "") << t.points[i].accelerations_deg_s2[j];
		os << "]}" << (i + 1 < t.points.size() ? "," : "") << "\n";
	}
	os << "  ],\n  \"phase_boundaries\": [\n";
	for (std::size_t i = 0; i < t.phases.size(); ++i)
	{
		os << "    {\"name\": " << jsonString(t.phases[i].name)
		   << ", \"start_index\": " << t.phases[i].start_index
		   << ", \"end_index\": " << t.phases[i].end_index << "}"
		   << (i + 1 < t.phases.size() ? "," : "") << "\n";
	}
	os << "  ],\n  \"gripper_events\": [\n";
	os << "    {\"name\": \"open_before_motion\", \"point_index\": 0},\n";
	os << "    {\"name\": \"close_at_attach\", \"point_index\": "
	   << t.attach_point_index << "},\n";
	os << "    {\"name\": \"open_at_release\", \"point_index\": "
	   << t.release_point_index << "}\n";
	os << "  ]\n}\n";
}

std::string formatResultSummary(const RunResult& r)
{
	std::ostringstream os;
	os << "scenario=" << r.scenario_id << " mode=" << r.mode << " solved=" << (r.solved ? "true" : "false")
	   << " selected_arm=" << (r.selected_arm.empty() ? "<none>" : r.selected_arm)
	   << " execution_eligible=" << (r.execution_eligible ? "true" : "false");
	if (!r.execution_block_reason.empty())
		os << " (" << r.execution_block_reason << ")";
	for (const auto& arm : r.arms)
	{
		os << "\n  " << arm.branch_id << ": complete_solutions=" << arm.complete_solution_count;
		if (arm.best_total_cost >= 0.0)
			os << " best_cost=" << arm.best_total_cost;
		if (!arm.earliest_failure_stage.empty())
			os << " earliest_failure_stage=" << arm.earliest_failure_stage;
	}
	return os.str();
}

}  // namespace grabber_mtc
