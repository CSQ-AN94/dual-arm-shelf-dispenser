// Closed-loop, rotation-only Woosh helper for the bottle side-table workflow.
//
// This is the source for the binary that mobile_body.py's WooshChassisAdapter
// shells out to (`rotate_helper_path`, default
// /home/rm/agv_debug_tools/grabber_rotate_relative).  It lives here so
// robot_code_drift.py can hash it: the source drifting silently is the whole
// reason it was pulled into the repo.
//
// Drift tracking covers this file, NOT the binary next to it.  Editing here
// and pushing does not change what the chassis runs until you rebuild on the
// robot -- do that in the same session, never separately.
//
// Build on the robot (do not run it during build/inspection):
// g++ -std=c++17 -O2 /home/rm/dual-arm-shelf-dispenser/scripts/woosh_rotate_relative.cpp \
//   -I /home/rm/rmc_aida_l_atom/include \
//   /home/rm/rmc_aida_l_atom/lib/linux/libwoosh_robot.so.1.1 \
//   /home/rm/rmc_aida_l_atom/lib/linux/libprotobuf.so.32 -lpthread \
//   -Wl,-rpath,/home/rm/rmc_aida_l_atom/lib/linux \
//   -o /home/rm/agv_debug_tools/grabber_rotate_relative

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "woosh_robot.h"

namespace {

std::atomic<bool> g_stop{false};
constexpr int kControlPeriodMs = 50;   // Match the proven 20 Hz Woosh tools.
constexpr int kPoseCheckEveryCommands = 4;  // Re-verify odometry at 5 Hz.

void SignalHandler(int) { g_stop.store(true); }

double Wrap(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

struct Pose {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double linear = 0.0;
  double angular = 0.0;
};

template <typename Result, typename Start>
bool WaitResult(Start start, Result* output, std::string* message,
                int timeout_ms = 3000) {
  struct SharedResult {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    bool ok = false;
    Result result;
    std::string message;
  };
  auto shared = std::make_shared<SharedResult>();
  start([shared](const Result& result, bool result_ok,
                 const std::string& result_msg) {
    {
      std::lock_guard<std::mutex> lock(shared->mutex);
      shared->result = result;
      shared->message = result_msg;
      shared->ok = result_ok;
      shared->done = true;
    }
    shared->cv.notify_one();
  });
  std::unique_lock<std::mutex> lock(shared->mutex);
  if (!shared->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           [&] { return shared->done; })) {
    *message = "request timed out";
    return false;
  }
  *output = shared->result;
  *message = shared->message;
  return shared->ok;
}

bool QueryPose(const std::shared_ptr<woosh::RobotInterface>& robot, Pose* pose,
               std::string* error) {
  woosh::robot::PoseSpeed request;
  woosh::robot::PoseSpeed result;
  const bool ok = WaitResult<woosh::robot::PoseSpeed>(
      [&](auto finish) {
        robot->robotPoseSpeedReq(request, finish, woosh::PPLB, woosh::PPLB);
      },
      &result, error, 4000);
  if (!ok) return false;
  pose->x = result.pose().x();
  pose->y = result.pose().y();
  pose->yaw = result.pose().theta();
  pose->linear = result.twist().linear();
  pose->angular = result.twist().angular();
  return true;
}

template <typename Request, typename Result, typename Start>
bool QueryDebug(Request request, Start start, std::string* debug,
                std::string* error) {
  Result result;
  const bool ok = WaitResult<Result>(
      [&](auto finish) { start(request, finish); }, &result, error, 4000);
  if (ok) *debug = result.DebugString();
  return ok;
}

bool SendTwist(const std::shared_ptr<woosh::RobotInterface>& robot,
               double angular, std::string* error) {
  woosh::robot::Twist request;
  request.set_linear(0.0);  // Invariant: this helper can never command translation.
  request.set_angular(angular);
  google::protobuf::Empty result;
  return WaitResult<google::protobuf::Empty>(
      [&](auto finish) {
        robot->twistReq(request, finish, woosh::PPLB, woosh::PPLB);
      },
      &result, error, 2500);
}

void StopRepeated(const std::shared_ptr<woosh::RobotInterface>& robot) {
  for (int i = 0; i < 3; ++i) {
    std::string ignored;
    SendTwist(robot, 0.0, &ignored);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
  }
}

void PrintPose(const Pose& pose) {
  std::printf("twist linear=%.9f angular=%.9f\n", pose.linear, pose.angular);
  std::printf("pose x=%.9f y=%.9f theta=%.9f\n", pose.x, pose.y, pose.yaw);
}

bool ParseDouble(int argc, char** argv, const std::string& name, double* value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      char* end = nullptr;
      const double parsed = std::strtod(argv[i + 1], &end);
      if (end == argv[i + 1] || *end != '\0' || !std::isfinite(parsed)) {
        return false;
      }
      *value = parsed;
      return true;
    }
  }
  return false;
}

bool HasFlag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  woosh::CommuSetting setting;
  setting.addr = "169.254.128.2";
  setting.port = 5410;
  setting.identity = "grabber-closed-loop-rotate";
  setting.log_call_fun = [](const std::string&) {};
  setting.print_pack_call_fun = [](const std::string&) {};
  setting.connect_status_call_fun = [](const bool&) {};
  auto robot = woosh::Factory::newRobotInterface(setting);
  if (!robot->run()) {
    std::fprintf(stderr, "cannot start Woosh interface\n");
    return 2;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(700));

  if (HasFlag(argc, argv, "--stop")) {
    StopRepeated(robot);
    Pose stopped;
    std::string error;
    if (!QueryPose(robot, &stopped, &error)) {
      std::fprintf(stderr, "stop pose query failed: %s\n", error.c_str());
      return 3;
    }
    PrintPose(stopped);
    return 0;
  }

  double requested = 0.0;
  double max_angular = 0.12;
  double tolerance = 2.0 * M_PI / 180.0;
  double max_translation = 0.035;
  double timeout_s = 25.0;
  if (!ParseDouble(argc, argv, "--rotate-relative-rad", &requested) ||
      !ParseDouble(argc, argv, "--max-angular-radps", &max_angular) ||
      !ParseDouble(argc, argv, "--yaw-tolerance-rad", &tolerance) ||
      !ParseDouble(argc, argv, "--max-translation-m", &max_translation) ||
      !ParseDouble(argc, argv, "--timeout-s", &timeout_s)) {
    std::fprintf(stderr, "missing/invalid arguments\n");
    return 4;
  }
  if (std::abs(requested) < 80.0 * M_PI / 180.0 ||
      std::abs(requested) > 100.0 * M_PI / 180.0 || max_angular <= 0.0 ||
      max_angular > 0.20 || tolerance <= 0.0 || tolerance > 5.0 * M_PI / 180.0 ||
      max_translation <= 0.0 || max_translation > 0.08 || timeout_s < 5.0 ||
      timeout_s > 60.0) {
    std::fprintf(stderr, "arguments outside audited limits\n");
    return 5;
  }

  std::string mode_debug, state_debug, error;
  woosh::robot::Mode mode_request;
  if (!QueryDebug<woosh::robot::Mode, woosh::robot::Mode>(
          mode_request,
          [&](auto request, auto finish) {
            robot->robotModeReq(request, finish, woosh::PPLB, woosh::PPLB);
          },
          &mode_debug, &error) ||
      mode_debug.find("ctrl: kAuto") == std::string::npos) {
    std::fprintf(stderr, "physical control mode is not kAuto: %s %s\n",
                 mode_debug.c_str(), error.c_str());
    return 6;
  }
  woosh::robot::RobotState state_request;
  if (!QueryDebug<woosh::robot::RobotState, woosh::robot::RobotState>(
          state_request,
          [&](auto request, auto finish) {
            robot->robotStateReq(request, finish, woosh::PPLB, woosh::PPLB);
          },
          &state_debug, &error) ||
      state_debug.find("state: kIdle") == std::string::npos) {
    std::fprintf(stderr, "robot state is not kIdle: %s %s\n",
                 state_debug.c_str(), error.c_str());
    return 7;
  }

  Pose start;
  if (!QueryPose(robot, &start, &error)) {
    std::fprintf(stderr, "start pose query failed: %s\n", error.c_str());
    return 8;
  }
  if (std::abs(start.linear) > 0.005 || std::abs(start.angular) > 0.01) {
    std::fprintf(stderr, "chassis is not stationary at start\n");
    return 9;
  }
  const double target = Wrap(start.yaw + requested);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<int>(timeout_s * 1000));
  int stable = 0;
  Pose current = start;
  int result_code = 0;
  while (!g_stop.load() && std::chrono::steady_clock::now() < deadline) {
    if (!QueryPose(robot, &current, &error)) {
      std::fprintf(stderr, "live pose query failed: %s\n", error.c_str());
      result_code = 10;
      break;
    }
    const double translation = std::hypot(current.x - start.x, current.y - start.y);
    if (translation > max_translation) {
      std::fprintf(stderr, "translation guard tripped: %.6f m\n", translation);
      result_code = 11;
      break;
    }
    const double yaw_error = Wrap(target - current.yaw);
    if (std::abs(yaw_error) <= tolerance) {
      StopRepeated(robot);
      ++stable;
      if (stable >= 3) break;
    } else {
      stable = 0;
      const double magnitude = std::clamp(std::abs(yaw_error) * 0.7, 0.03, max_angular);
      const double command = std::copysign(magnitude, yaw_error);
      // Woosh continuous velocity control is watchdog-driven.  Keep the same
      // 20 Hz refresh cadence as the robot-side tool proven on this chassis,
      // while bounding the open-loop window to 200 ms before every fresh pose
      // and translation-guard check.
      for (int i = 0; i < kPoseCheckEveryCommands; ++i) {
        if (g_stop.load() || std::chrono::steady_clock::now() >= deadline) break;
        if (!SendTwist(robot, command, &error)) {
          std::fprintf(stderr, "twist rejected: %s\n", error.c_str());
          result_code = 12;
          break;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kControlPeriodMs));
      }
      if (result_code != 0) break;
    }
  }
  StopRepeated(robot);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!QueryPose(robot, &current, &error)) {
    std::fprintf(stderr, "final pose query failed: %s\n", error.c_str());
    return 13;
  }
  PrintPose(current);
  if (g_stop.load()) return 130;
  if (result_code != 0) return result_code;
  if (stable < 3) {
    std::fprintf(stderr, "rotation timed out before stable convergence\n");
    return 14;
  }
  const double translation = std::hypot(current.x - start.x, current.y - start.y);
  const double final_error = std::abs(Wrap(target - current.yaw));
  if (translation > max_translation || final_error > tolerance ||
      std::abs(current.linear) > 0.005 || std::abs(current.angular) > 0.01) {
    std::fprintf(stderr, "final verification failed\n");
    return 15;
  }
  return 0;
}
