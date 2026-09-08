// Closed-loop short-range chassis pose servo for the two taught body stops.
// Build on the robot; --self-test never connects to hardware.

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
constexpr int kPeriodMs = 50;
constexpr int kCommandsPerPose = 4;

double Wrap(double value) {
  return std::atan2(std::sin(value), std::cos(value));
}

struct Pose {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double linear = 0.0;
  double angular = 0.0;
};

struct Command {
  double linear = 0.0;
  double angular = 0.0;
  double position_error = 0.0;
  double yaw_error = 0.0;
};

void SignalHandler(int) { g_stop.store(true); }

Pose RelativeTarget(const Pose& start, double dx, double dy, double dyaw) {
  const double c = std::cos(start.yaw);
  const double s = std::sin(start.yaw);
  return Pose{start.x + c * dx - s * dy,
              start.y + s * dx + c * dy,
              Wrap(start.yaw + dyaw)};
}

Command PoseCommand(const Pose& current, const Pose& target,
                    double max_linear, double max_angular,
                    double position_tolerance, double yaw_tolerance) {
  const double dx = target.x - current.x;
  const double dy = target.y - current.y;
  const double rho = std::hypot(dx, dy);
  const double yaw_error = Wrap(target.yaw - current.yaw);
  if (rho <= position_tolerance) {
    double angular = std::clamp(1.1 * yaw_error, -max_angular, max_angular);
    if (std::abs(yaw_error) > yaw_tolerance && std::abs(angular) < 0.025) {
      angular = std::copysign(0.025, angular);
    }
    return Command{0.0, angular, rho, std::abs(yaw_error)};
  }

  double alpha = Wrap(std::atan2(dy, dx) - current.yaw);
  double signed_rho = rho;
  if (std::abs(alpha) > M_PI / 2.0) {
    alpha = Wrap(alpha + (alpha > 0.0 ? -M_PI : M_PI));
    signed_rho = -rho;
  }
  const double beta = Wrap(target.yaw - current.yaw - alpha);
  double linear = std::clamp(0.35 * signed_rho, -max_linear, max_linear);
  if (std::abs(linear) < 0.012) linear = std::copysign(0.012, linear);
  double angular = std::clamp(1.15 * alpha - 0.28 * beta,
                              -max_angular, max_angular);
  if (std::abs(alpha) > yaw_tolerance && std::abs(angular) < 0.025) {
    angular = std::copysign(0.025, angular);
  }
  return Command{linear, angular, rho, std::abs(yaw_error)};
}

bool ParseDouble(int argc, char** argv, const std::string& name,
                 double* value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] != name) continue;
    char* end = nullptr;
    const double parsed = std::strtod(argv[i + 1], &end);
    if (end == argv[i + 1] || *end != '\0' || !std::isfinite(parsed)) {
      return false;
    }
    *value = parsed;
    return true;
  }
  return false;
}

bool HasFlag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) return true;
  }
  return false;
}

bool SelfTest() {
  Pose start{1.0, 2.0, M_PI / 2.0};
  const Pose target = RelativeTarget(start, 0.2, -0.1, M_PI / 3.0);
  if (std::abs(target.x - 1.1) > 1e-9 ||
      std::abs(target.y - 2.2) > 1e-9 ||
      std::abs(Wrap(target.yaw - 5.0 * M_PI / 6.0)) > 1e-9) {
    return false;
  }
  const Command forward = PoseCommand(start, target, 0.03, 0.08, 0.01,
                                      M_PI / 180.0);
  if (!(forward.position_error > 0.2 && std::abs(forward.linear) <= 0.03 &&
        std::abs(forward.angular) <= 0.08)) {
    return false;
  }
  const Command done = PoseCommand(target, target, 0.03, 0.08, 0.01,
                                   M_PI / 180.0);
  return done.linear == 0.0 && done.angular == 0.0;
}

template <typename Result, typename Start>
bool WaitResult(Start start, Result* output, std::string* message,
                int timeout_ms = 3000) {
  struct Shared {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    bool ok = false;
    Result result;
    std::string message;
  };
  auto shared = std::make_shared<Shared>();
  start([shared](const Result& result, bool ok, const std::string& message) {
    {
      std::lock_guard<std::mutex> lock(shared->mutex);
      shared->result = result;
      shared->ok = ok;
      shared->message = message;
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

template <typename Request, typename Result, typename Start>
bool QueryDebug(Request request, Start start, std::string* debug,
                std::string* error) {
  Result result;
  const bool ok = WaitResult<Result>(
      [&](auto finish) { start(request, finish); }, &result, error, 4000);
  if (ok) *debug = result.DebugString();
  return ok;
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

bool SendTwist(const std::shared_ptr<woosh::RobotInterface>& robot,
               double linear, double angular, std::string* error) {
  woosh::robot::Twist request;
  request.set_linear(linear);
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
    SendTwist(robot, 0.0, 0.0, &ignored);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
  }
}

void PrintPose(const char* label, const Pose& pose) {
  std::printf("%s x=%.9f y=%.9f yaw=%.9f linear=%.9f angular=%.9f\n",
              label, pose.x, pose.y, pose.yaw, pose.linear, pose.angular);
}

}  // namespace

int main(int argc, char** argv) {
  if (HasFlag(argc, argv, "--self-test")) {
    const bool ok = SelfTest();
    std::printf("self_test=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }

  double dx = 0.0, dy = 0.0, dyaw = 0.0;
  double target_x = 0.0, target_y = 0.0, target_yaw = 0.0;
  double max_linear = 0.025, max_angular = 0.06;
  double position_tolerance = 0.012;
  double yaw_tolerance = M_PI / 180.0;
  double timeout_s = 75.0;
  const bool has_absolute =
      ParseDouble(argc, argv, "--target-x-m", &target_x) &&
      ParseDouble(argc, argv, "--target-y-m", &target_y) &&
      ParseDouble(argc, argv, "--target-yaw-rad", &target_yaw);
  const bool has_relative =
      ParseDouble(argc, argv, "--dx-m", &dx) &&
      ParseDouble(argc, argv, "--dy-m", &dy) &&
      ParseDouble(argc, argv, "--dyaw-rad", &dyaw);
  if (has_absolute == has_relative) {
    std::fprintf(stderr, "missing/invalid relative target\n");
    return 2;
  }
  ParseDouble(argc, argv, "--max-linear-mps", &max_linear);
  ParseDouble(argc, argv, "--max-angular-radps", &max_angular);
  ParseDouble(argc, argv, "--position-tolerance-m", &position_tolerance);
  ParseDouble(argc, argv, "--yaw-tolerance-rad", &yaw_tolerance);
  ParseDouble(argc, argv, "--timeout-s", &timeout_s);
  if ((!has_absolute &&
       (std::hypot(dx, dy) > 0.30 || std::abs(dyaw) > 2.2)) ||
      max_linear <= 0.0 || max_linear > 0.04 || max_angular <= 0.0 ||
      max_angular > 0.10 || position_tolerance < 0.005 ||
      position_tolerance > 0.03 || yaw_tolerance < 0.005 ||
      yaw_tolerance > 0.05 || timeout_s < 10.0 || timeout_s > 120.0) {
    std::fprintf(stderr, "arguments outside audited limits\n");
    return 3;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);
  // Reuse the audited session preflight on every invocation. kIdle can be
  // left behind by an older client and is not proof of clean ownership.
  if (std::system("/home/rm/agv_debug_tools/agv_mode_init") != 0) {
    std::fprintf(stderr, "chassis session initialization failed\n");
    return 4;
  }
  woosh::CommuSetting setting;
  setting.addr = "169.254.128.2";
  setting.port = 5410;
  setting.identity = "grabber-short-pose-servo";
  setting.log_call_fun = [](const std::string&) {};
  setting.print_pack_call_fun = [](const std::string&) {};
  setting.connect_status_call_fun = [](bool) {};
  auto robot = woosh::Factory::newRobotInterface(setting);
  if (!robot->run()) {
    std::fprintf(stderr, "cannot start Woosh interface\n");
    return 5;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(700));

  std::string mode_debug, state_debug, error;
  woosh::robot::Mode mode_request;
  if (!QueryDebug<woosh::robot::Mode, woosh::robot::Mode>(
          mode_request,
          [&](auto request, auto finish) {
            robot->robotModeReq(request, finish, woosh::PPLB, woosh::PPLB);
          },
          &mode_debug, &error) || mode_debug.find("ctrl: kAuto") == std::string::npos) {
    std::fprintf(stderr, "physical control mode is not kAuto\n");
    return 6;
  }
  woosh::robot::RobotState state_request;
  if (!QueryDebug<woosh::robot::RobotState, woosh::robot::RobotState>(
          state_request,
          [&](auto request, auto finish) {
            robot->robotStateReq(request, finish, woosh::PPLB, woosh::PPLB);
          },
          &state_debug, &error) || state_debug.find("state: kIdle") == std::string::npos) {
    std::fprintf(stderr, "robot state is not kIdle\n");
    return 7;
  }

  Pose start;
  if (!QueryPose(robot, &start, &error) || std::abs(start.linear) > 0.005 ||
      std::abs(start.angular) > 0.01) {
    std::fprintf(stderr, "chassis start pose is unavailable or moving: %s\n",
                 error.c_str());
    return 8;
  }
  const Pose target = has_absolute
      ? Pose{target_x, target_y, Wrap(target_yaw)}
      : RelativeTarget(start, dx, dy, dyaw);
  if (std::hypot(target.x - start.x, target.y - start.y) > 0.30 ||
      std::abs(Wrap(target.yaw - start.yaw)) > 2.2) {
    std::fprintf(stderr, "post-initialization target is outside local limits\n");
    return 8;
  }
  PrintPose("start", start);
  PrintPose("target", target);

  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(static_cast<int>(timeout_s * 1000));
  Pose current = start;
  int stable = 0;
  int result = 0;
  int cycle = 0;
  enum class Phase { kTurnToPath, kDrive, kFinalTurn };
  Phase phase = Phase::kTurnToPath;
  double direction = 1.0;
  int corrections = 0;
  auto choose_direction = [&](const Pose& pose) {
    const double bearing = std::atan2(target.y - pose.y, target.x - pose.x);
    const double forward = std::abs(Wrap(bearing - pose.yaw)) +
                           std::abs(Wrap(target.yaw - bearing));
    const double reverse_heading = Wrap(bearing + M_PI);
    const double reverse = std::abs(Wrap(reverse_heading - pose.yaw)) +
                           std::abs(Wrap(target.yaw - reverse_heading));
    direction = reverse < forward ? -1.0 : 1.0;
  };
  choose_direction(current);
  while (!g_stop.load() && std::chrono::steady_clock::now() < deadline) {
    if (!QueryPose(robot, &current, &error)) {
      std::fprintf(stderr, "live pose query failed: %s\n", error.c_str());
      result = 9;
      break;
    }
    const double from_start = std::hypot(current.x - start.x,
                                         current.y - start.y);
    if (from_start > 0.40) {
      std::fprintf(stderr, "travel envelope tripped: %.6f m\n", from_start);
      result = 10;
      break;
    }
    const double position_error = std::hypot(target.x - current.x,
                                             target.y - current.y);
    const double yaw_error = std::abs(Wrap(target.yaw - current.yaw));
    const double bearing = std::atan2(target.y - current.y,
                                      target.x - current.x);
    const double path_heading = Wrap(bearing + (direction < 0.0 ? M_PI : 0.0));
    const double heading_error = Wrap(path_heading - current.yaw);
    Command command{0.0, 0.0, position_error, yaw_error};
    if (phase == Phase::kTurnToPath) {
      command.angular = std::clamp(1.1 * heading_error,
                                   -max_angular, max_angular);
      if (std::abs(heading_error) <= yaw_tolerance) {
        phase = Phase::kDrive;
        command.angular = 0.0;
      } else if (std::abs(command.angular) < 0.025) {
        command.angular = std::copysign(0.025, command.angular);
      }
    } else if (phase == Phase::kDrive) {
      if (position_error <= position_tolerance) {
        phase = Phase::kFinalTurn;
      } else if (std::abs(heading_error) > 0.35) {
        phase = Phase::kTurnToPath;
      } else {
        command.linear = direction * std::clamp(0.35 * position_error,
                                                0.012, max_linear);
        command.angular = std::clamp(1.2 * heading_error,
                                     -max_angular, max_angular);
      }
    }
    if (phase == Phase::kFinalTurn) {
      const double signed_yaw_error = Wrap(target.yaw - current.yaw);
      if (std::abs(signed_yaw_error) <= yaw_tolerance) {
        if (position_error <= position_tolerance) {
          command = Command{0.0, 0.0, position_error, yaw_error};
        } else if (++corrections <= 3) {
          choose_direction(current);
          phase = Phase::kTurnToPath;
        } else {
          std::fprintf(stderr, "position drifted during final turns\n");
          result = 11;
          break;
        }
      } else {
        command.angular = std::clamp(1.1 * signed_yaw_error,
                                     -max_angular, max_angular);
        if (std::abs(command.angular) < 0.025) {
          command.angular = std::copysign(0.025, command.angular);
        }
      }
    }
    if (++cycle % 5 == 0) {
      std::printf("feedback position_error=%.6f yaw_error_deg=%.3f "
                  "cmd_linear=%.4f cmd_angular=%.4f\n",
                  command.position_error, command.yaw_error * 180.0 / M_PI,
                  command.linear, command.angular);
    }
    if (command.position_error <= position_tolerance &&
        command.yaw_error <= yaw_tolerance) {
      StopRepeated(robot);
      if (++stable >= 3) break;
      continue;
    }
    stable = 0;
    for (int i = 0; i < kCommandsPerPose; ++i) {
      if (g_stop.load()) break;
      if (!SendTwist(robot, command.linear, command.angular, &error)) {
        std::fprintf(stderr, "twist rejected: %s\n", error.c_str());
        result = 12;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kPeriodMs));
    }
    if (result != 0) break;
  }
  StopRepeated(robot);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!QueryPose(robot, &current, &error)) {
    std::fprintf(stderr, "final pose query failed: %s\n", error.c_str());
    return 13;
  }
  PrintPose("final", current);
  const double final_position = std::hypot(target.x - current.x,
                                           target.y - current.y);
  const double final_yaw = std::abs(Wrap(target.yaw - current.yaw));
  std::printf("result position_error_m=%.9f yaw_error_deg=%.6f stable=%d\n",
              final_position, final_yaw * 180.0 / M_PI, stable);
  if (g_stop.load()) return 130;
  if (result != 0) return result;
  if (stable < 3 || final_position > position_tolerance ||
      final_yaw > yaw_tolerance || std::abs(current.linear) > 0.005 ||
      std::abs(current.angular) > 0.01) {
    std::fprintf(stderr, "pose servo did not converge\n");
    return 14;
  }
  return 0;
}
