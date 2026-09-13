#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <linux/input.h>
#include <arpa/inet.h>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <poll.h>
#include <sstream>
#include <csignal>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <sys/socket.h>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/video/video_client.hpp>
#include "collector_input_backend.h"
#include "collector_input_utils.h"
#include "raw_terminal_guard.h"
#include "web_ui_server.h"

namespace
{

std::atomic<bool> gSignalStopRequested{false};

void HandleStopSignal(int)
{
    gSignalStopRequested.store(true);
}

namespace fs = std::filesystem;
constexpr char kDefaultDataDirName[] = "data";

constexpr char kSportStateTopic[] = "rt/sportmodestate";
constexpr char kWirelessControllerTopic[] = "rt/wirelesscontroller";
constexpr char kSchemaVersion[] = "go2_local_dataset_v1";
constexpr char kRealSourceType[] = "real_go2";
constexpr char kTrajectoryCaptureMode[] = "trajectory";
constexpr char kPreviewSceneId[] = "preview_scene";
constexpr char kPreviewOperatorId[] = "preview_operator";
constexpr char kPreviewInstruction[] = "go to the door";
constexpr char kPreviewTaskFamily[] = "goal_navigation";
constexpr char kPreviewTargetType[] = "door";
constexpr char kPreviewTargetDescription[] = "preview camera sample";
constexpr double kStateTimeoutSeconds = 1.0;
constexpr double kWirelessControllerTimeoutSeconds = 1.0;
constexpr double kMaxSafeAbsRollRad = 0.75;
constexpr double kMaxSafeAbsPitchRad = 0.75;
constexpr double kDefaultCmdVxMax = 1.0;
constexpr double kDefaultCmdVyMax = 0.3;
constexpr double kDefaultCmdWzMax = 0.8;
constexpr float kStandingBodyHeightThreshold = 0.18f;
constexpr float kKeyboardLinearAccelPerSecond = 0.9f;
constexpr float kKeyboardLinearDecelPerSecond = 1.1f;
constexpr float kKeyboardYawAccelPerSecond = 0.8f;
constexpr float kKeyboardYawDecelPerSecond = 1.0f;
constexpr float kLinearAccelPerSecond = 1.2f;
constexpr float kLinearDecelPerSecond = 1.8f;
constexpr float kYawAccelPerSecond = 1.5f;
constexpr float kYawDecelPerSecond = 2.5f;
constexpr float kWirelessControllerDiscreteThreshold = 0.85f;
constexpr float kWirelessControllerStickDeadZone = 0.08f;
constexpr float kWirelessControllerStickSmoothing = 0.18f;
constexpr float kWirelessCollectorDeadZone = 0.22f;
constexpr float kWirelessCollectorReleaseZone = 0.16f;
constexpr float kWirelessCollectorAxisExponent = 1.8f;
constexpr auto kTrajectoryFinalizeGracePeriod = std::chrono::milliseconds(400);
constexpr auto kTrajectoryStopReleaseTimeout = std::chrono::milliseconds(1200);
constexpr char kEscapeKey = 27;
constexpr char kBackspace = 127;
constexpr char kCtrlH = 8;

struct TrajectoryMotionGateConfig
{
    float vxStartThreshold = 0.04f;
    float vyStartThreshold = 0.04f;
    float wzStartThreshold = 0.10f;
    float vxStopThreshold = 0.02f;
    float vyStopThreshold = 0.02f;
    float wzStopThreshold = 0.06f;
    size_t startConsecutiveFrames = 4;
    size_t stopConsecutiveFrames = 6;
    size_t preRollFrames = 4;
};

const std::vector<std::string> kAllowedInstructions = {
    "go forward",
    "move backward",
    "strafe left",
    "strafe right",
    "stand up",
    "lie down",
    "turn left",
    "turn right",
    "stay still",
};

struct Config
{
    enum class InputBackend
    {
        WirelessController,
        Evdev,
    };

    enum class WirelessMotionMode
    {
        NativePassthrough,
        CollectorScaledMove,
    };

    std::string networkInterface;
    fs::path collectorRoot;
    fs::path outputDir;
    double loopHz = 50.0;
    double videoPollHz = 20.0;
    InputBackend inputBackend = InputBackend::WirelessController;
    WirelessMotionMode wirelessMotionMode = WirelessMotionMode::NativePassthrough;
    fs::path inputDevice;
    std::string sceneId;
    std::string operatorId;
    std::string instruction;
    std::string instructionSource = "semantic_text";
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string collectorNotes;
    double cmdVxMax = kDefaultCmdVxMax;
    double cmdVyMax = kDefaultCmdVyMax;
    double cmdWzMax = kDefaultCmdWzMax;
    bool d435iEnabled = false;
    bool d435iAutostartNode = true;
    std::string d435iRosSetupPath = "/opt/ros/noetic/setup.bash";
    std::string d435iBagBaseName = "d435i_rgbd";
    std::string d435iSerial;
    std::string d435iColorProfile = "640x480x30";
    std::string d435iDepthProfile = "640x480x30";
    std::string d435iLaunchFile = "realsense2_camera rs_camera.launch";
    bool d435iAlignDepth = false;
    double d435iDepthPreviewMinMeters = 0.2;
    double d435iDepthPreviewMaxMeters = 3.0;
    double d435iNodeWarmupSeconds = 3.0;
    double d435iTopicReadyTimeoutSeconds = 12.0;
    bool webUiEnabled = false;
    bool previewUi = false;
    int webPort = 8080;
};

struct EditableCollectorConfig
{
    std::string sceneId;
    std::string operatorId;
    std::string instruction;
    std::string instructionSource = "semantic_text";
    std::string taskFamily;
    std::string targetType;
    std::string targetDescription;
    std::string collectorNotes;
    double cmdVxMax = kDefaultCmdVxMax;
    double cmdVyMax = kDefaultCmdVyMax;
    double cmdWzMax = kDefaultCmdWzMax;
};

struct TaskMetadata
{
    std::string instruction;
    std::string captureMode = kTrajectoryCaptureMode;
    std::string taskFamily;
    std::string targetType;
    std::string targetLabel;
    std::string targetDescription;
    std::string collectorNotes;
    std::string instructionSource;
    std::string segmentStatus;
    std::string success;
    std::string terminationReason;
};

struct LatestState
{
    double timestamp = 0.0;
    uint32_t errorCode = 0;
    uint8_t mode = 0;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float velocityX = 0.0f;
    float velocityY = 0.0f;
    float velocityZ = 0.0f;
    float yawSpeed = 0.0f;
    float bodyHeight = 0.0f;
    uint8_t gaitType = 0;
    bool valid = false;
};

struct LatestImage
{
    double timestamp = 0.0;
    uint64_t sequence = 0;
    std::vector<uint8_t> jpegBytes;
    bool valid = false;
};

struct VelocityCommand
{
    double timestamp = 0.0;
    float vx = 0.0f;
    float vy = 0.0f;
    float wz = 0.0f;
    bool valid = false;
};

struct EffectiveControlAction
{
    VelocityCommand command;
    double timestamp = 0.0;
};

struct EpisodeFrame
{
    double timestamp = 0.0;
    double stateTimestamp = 0.0;
    double actionTimestamp = 0.0;
    double rawActionTimestamp = 0.0;
    double imageTimestamp = 0.0;
    float stateVx = 0.0f;
    float stateVy = 0.0f;
    float stateVz = 0.0f;
    float stateWz = 0.0f;
    float stateRoll = 0.0f;
    float statePitch = 0.0f;
    float stateYaw = 0.0f;
    float statePositionX = 0.0f;
    float statePositionY = 0.0f;
    float statePositionZ = 0.0f;
    float stateBodyHeight = 0.0f;
    uint32_t stateErrorCode = 0;
    uint8_t stateMode = 0;
    uint8_t stateGaitType = 0;
    float rawActionVx = 0.0f;
    float rawActionVy = 0.0f;
    float rawActionWz = 0.0f;
    float controlActionVx = 0.0f;
    float controlActionVy = 0.0f;
    float controlActionWz = 0.0f;
    bool motionInputActive = false;
    std::vector<uint8_t> jpegBytes;
};

struct EpisodeSummary
{
    std::string episodeId;
    std::string instruction;
    std::string captureMode = kTrajectoryCaptureMode;
    std::string taskFamily;
    std::string targetType;
    std::string targetLabel;
    std::string targetDescription;
    std::string collectorNotes;
    std::string instructionSource;
    std::string segmentStatus;
    std::string success;
    std::string terminationReason;
    std::string sceneId;
    std::string operatorId;
    size_t numFrames = 0;
    double startTimestamp = 0.0;
    double endTimestamp = 0.0;
};

enum class SegmentStatus
{
    Clean,
    Usable,
    Discard,
};

enum class SuccessLabel
{
    Success,
    Partial,
    Fail,
};

enum class TerminationReason
{
    GoalReached,
    NearGoalStop,
    Occluded,
    TargetLost,
    OperatorStop,
    BadDemo,
    Unsafe,
    Timeout,
};

enum class SafetyState
{
    SafeReady,
    FaultLatched,
    EstopLatched,
};

enum class CaptureState
{
    Idle,
    Capturing,
    Fault,
};

double NowSeconds()
{
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string Trim(const std::string& value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])))
    {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::string ToLowerAscii(std::string value)
{
    for (char& ch : value)
    {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool StartsWith(const std::string& value, const std::string& prefix)
{
    return value.rfind(prefix, 0) == 0;
}

std::optional<std::string> CanonicalTaskFamily(const std::string& value)
{
    const std::string lowered = ToLowerAscii(Trim(value));
    if (lowered.empty())
    {
        return std::nullopt;
    }
    if (lowered == "goal_navigation" || lowered == "goal-nav" || lowered == "navigate" || lowered == "navigation" ||
        lowered == "nav")
    {
        return std::string("goal_navigation");
    }
    if (lowered == "visual_following" || lowered == "follow" || lowered == "following" ||
        lowered == "person_following" || lowered == "pedestrian_following" || lowered == "human_following" ||
        lowered == "person_tracking" || lowered == "tracking")
    {
        return std::string("visual_following");
    }
    if (lowered == "obstacle_aware_navigation" || lowered == "obstacle_avoidance" || lowered == "avoid" ||
        lowered == "avoidance" || lowered == "go_around")
    {
        return std::string("obstacle_aware_navigation");
    }
    if (lowered == "legacy_motion" || lowered == "legacy" || lowered == "motion")
    {
        return std::string("legacy_motion");
    }
    return std::nullopt;
}

std::string NormalizeTaskFamily(const std::string& value)
{
    if (const auto canonical = CanonicalTaskFamily(value); canonical.has_value())
    {
        return canonical.value();
    }
    return Trim(value);
}

std::string ResolveTargetText(const std::string& targetType, const std::string& targetDescription)
{
    const std::string trimmedDescription = Trim(targetDescription);
    if (!trimmedDescription.empty())
    {
        return trimmedDescription;
    }
    return Trim(targetType);
}

bool HasLeadingArticle(const std::string& value)
{
    static const std::vector<std::string> prefixes = {
        "the ",
        "a ",
        "an ",
        "this ",
        "that ",
        "these ",
        "those ",
        "my ",
        "your ",
        "our ",
        "their ",
    };

    const std::string lowered = ToLowerAscii(Trim(value));
    for (const std::string& prefix : prefixes)
    {
        if (StartsWith(lowered, prefix))
        {
            return true;
        }
    }
    return false;
}

std::string WithDefiniteArticle(const std::string& targetText)
{
    const std::string trimmed = Trim(targetText);
    if (trimmed.empty() || HasLeadingArticle(trimmed))
    {
        return trimmed;
    }
    return "the " + trimmed;
}

std::vector<std::string> BuildInstructionCandidates(
    const std::string& taskFamily,
    const std::string& targetType,
    const std::string& targetDescription)
{
    const auto canonicalTask = CanonicalTaskFamily(taskFamily);
    if (!canonicalTask.has_value())
    {
        return {};
    }

    const std::string targetText = ResolveTargetText(targetType, targetDescription);
    if (targetText.empty())
    {
        return {};
    }

    const std::string articleTarget = WithDefiniteArticle(targetText);
    if (canonicalTask.value() == "goal_navigation")
    {
        return {
            "go to " + articleTarget,
            "move to " + articleTarget,
            "navigate to " + articleTarget,
            "approach " + articleTarget,
        };
    }
    if (canonicalTask.value() == "visual_following")
    {
        return {
            "follow " + articleTarget,
            "track " + articleTarget,
            "stay with " + articleTarget,
            "walk with " + articleTarget,
        };
    }
    if (canonicalTask.value() == "obstacle_aware_navigation")
    {
        return {
            "go around " + articleTarget,
            "move around " + articleTarget,
            "navigate around " + articleTarget,
            "avoid " + articleTarget,
        };
    }
    return {};
}

std::optional<std::string> AutoComposeInstruction(
    const std::string& taskFamily,
    const std::string& targetType,
    const std::string& targetDescription,
    const std::string& sceneId,
    const std::string& operatorId)
{
    const auto candidates = BuildInstructionCandidates(taskFamily, targetType, targetDescription);
    if (candidates.empty())
    {
        return std::nullopt;
    }

    const std::string seed = NormalizeTaskFamily(taskFamily) + "|" + ResolveTargetText(targetType, targetDescription) +
                             "|" + Trim(sceneId) + "|" + Trim(operatorId) + "|" + std::to_string(std::time(nullptr));
    const size_t candidateIndex = std::hash<std::string>{}(seed) % candidates.size();
    return candidates[candidateIndex];
}

std::string JsonEscape(const std::string& input)
{
    std::ostringstream oss;
    for (const char ch : input)
    {
        switch (ch)
        {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << ch;
            break;
        }
    }
    return oss.str();
}

std::string JsonString(const std::string& input)
{
    return "\"" + JsonEscape(input) + "\"";
}

std::string JsonStringArray(const std::vector<std::string>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t index = 0; index < values.size(); ++index)
    {
        oss << JsonString(values[index]);
        if (index + 1 != values.size())
        {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

std::string ShellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (const char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted.push_back(ch);
        }
    }
    quoted += "'";
    return quoted;
}

std::vector<std::string> SplitWhitespaceTokens(const std::string& value)
{
    std::vector<std::string> tokens;
    std::istringstream input(value);
    std::string token;
    while (input >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> SplitCsvList(const std::string& input)
{
    std::vector<std::string> values;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        const std::string trimmed = Trim(item);
        if (!trimmed.empty())
        {
            values.push_back(trimmed);
        }
    }
    return values;
}

std::string JoinCsvList(const std::vector<std::string>& values)
{
    std::ostringstream oss;
    for (size_t index = 0; index < values.size(); ++index)
    {
        if (index > 0)
        {
            oss << ",";
        }
        oss << values[index];
    }
    return oss.str();
}

std::string InferTaskFamily(const std::string& instruction)
{
    const std::string trimmed = Trim(instruction);
    if (std::find(kAllowedInstructions.begin(), kAllowedInstructions.end(), trimmed) != kAllowedInstructions.end())
    {
        return "legacy_motion";
    }

    const std::string lowered = ToLowerAscii(trimmed);
    if (StartsWith(lowered, "go to ") || StartsWith(lowered, "move to ") || StartsWith(lowered, "navigate to ") ||
        StartsWith(lowered, "approach "))
    {
        return "goal_navigation";
    }
    if (StartsWith(lowered, "follow ") || StartsWith(lowered, "track ") || StartsWith(lowered, "stay with ") ||
        StartsWith(lowered, "walk with "))
    {
        return "visual_following";
    }
    if (StartsWith(lowered, "go around ") || StartsWith(lowered, "move around ") ||
        StartsWith(lowered, "navigate around ") || StartsWith(lowered, "avoid "))
    {
        return "obstacle_aware_navigation";
    }
    return std::string{};
}

TaskMetadata ResolveTaskMetadata(const TaskMetadata& configured, const std::string& fallbackInstruction)
{
    TaskMetadata resolved = configured;
    resolved.instruction = Trim(resolved.instruction);
    resolved.taskFamily = NormalizeTaskFamily(resolved.taskFamily);
    resolved.targetType = Trim(resolved.targetType);
    resolved.targetLabel = ResolveTargetText(resolved.targetType, resolved.targetDescription);
    resolved.targetDescription = Trim(resolved.targetDescription);
    if (resolved.instruction.empty())
    {
        resolved.instruction = Trim(fallbackInstruction);
        resolved.instructionSource = "motion_label";
    }
    else if (resolved.instructionSource.empty())
    {
        resolved.instructionSource = "semantic_text";
    }
    if (resolved.taskFamily.empty())
    {
        resolved.taskFamily = InferTaskFamily(resolved.instruction);
    }
    return resolved;
}

double AgeSeconds(double timestampSeconds, double nowSeconds)
{
    if (timestampSeconds <= 0.0)
    {
        return -1.0;
    }
    return std::max(0.0, nowSeconds - timestampSeconds);
}

std::tm LocalTimeFromEpochSeconds(std::time_t seconds)
{
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &seconds);
#else
    localtime_r(&seconds, &localTime);
#endif
    return localTime;
}

std::string FormatTimestampForPath(double timestampSeconds, const char* format)
{
    const std::time_t seconds = static_cast<std::time_t>(timestampSeconds);
    const std::tm localTime = LocalTimeFromEpochSeconds(seconds);
    char buffer[64];
    if (std::strftime(buffer, sizeof(buffer), format, &localTime) == 0)
    {
        throw std::runtime_error("格式化时间戳失败");
    }
    return buffer;
}

fs::path CollectorRootFromArgv0(const char* argv0)
{
    std::error_code error;
    fs::path executablePath = fs::read_symlink("/proc/self/exe", error);
    if (error || executablePath.empty())
    {
        error.clear();
        executablePath = fs::weakly_canonical(fs::absolute(argv0), error);
        if (error || executablePath.empty())
        {
            executablePath = fs::absolute(argv0);
        }
    }
    return executablePath.parent_path().parent_path().parent_path();
}

fs::path CollectorDefaultsPath(const fs::path& collectorRoot)
{
    return collectorRoot / "collector_defaults.json";
}

fs::path LegacyCollectorDefaultsPath(const fs::path& collectorRoot)
{
    return collectorRoot / "collector_webui_defaults.json";
}

bool UsesWirelessNativePassthrough(const Config& config)
{
    (void)config;
    return false;
}

std::string StartupUnlockHint(const Config& config)
{
    (void)config;
    return "当前输入后端无需额外解锁，可直接开始采集";
}

std::string StartupPromptText(const Config& config)
{
    if (config.inputBackend == Config::InputBackend::WirelessController)
    {
        return "collector 已完成初始化；原生手柄直通已关闭，手柄按键映射由 collector 接管，可立即移动和录制";
    }
    return "collector 已完成初始化；配置在启动时固定，可直接移动和录制";
}

std::string SafetyStateName(SafetyState state)
{
    switch (state)
    {
    case SafetyState::SafeReady:
        return "safe_ready";
    case SafetyState::FaultLatched:
        return "fault_latched";
    case SafetyState::EstopLatched:
        return "estop_latched";
    }
    return "unknown";
}

std::string CaptureStateName(CaptureState state)
{
    switch (state)
    {
    case CaptureState::Idle:
        return "idle";
    case CaptureState::Capturing:
        return "capturing";
    case CaptureState::Fault:
        return "fault";
    }
    return "unknown";
}

std::string TrajectoryStopPhaseName(bool pendingLabel, bool stopRequested, bool waitingForRelease)
{
    if (pendingLabel)
    {
        return "label_ready";
    }
    if (!stopRequested)
    {
        return "idle";
    }
    if (waitingForRelease)
    {
        return "waiting_release";
    }
    return "finalizing";
}

std::string InputBackendName(Config::InputBackend backend)
{
    switch (backend)
    {
    case Config::InputBackend::WirelessController:
        return "wireless_controller";
    case Config::InputBackend::Evdev:
        return "evdev";
    }
    return "unknown";
}

bool IsSupportedCaptureModeValue(const std::string& value)
{
    return Trim(value) == kTrajectoryCaptureMode;
}

std::string SegmentStatusName(SegmentStatus status)
{
    switch (status)
    {
    case SegmentStatus::Clean:
        return "clean";
    case SegmentStatus::Usable:
        return "usable";
    case SegmentStatus::Discard:
        return "discard";
    }
    return "discard";
}

std::string SuccessLabelName(SuccessLabel label)
{
    switch (label)
    {
    case SuccessLabel::Success:
        return "success";
    case SuccessLabel::Partial:
        return "partial";
    case SuccessLabel::Fail:
        return "fail";
    }
    return "fail";
}

std::string TerminationReasonName(TerminationReason reason)
{
    switch (reason)
    {
    case TerminationReason::GoalReached:
        return "goal_reached";
    case TerminationReason::NearGoalStop:
        return "near_goal_stop";
    case TerminationReason::Occluded:
        return "occluded";
    case TerminationReason::TargetLost:
        return "target_lost";
    case TerminationReason::OperatorStop:
        return "operator_stop";
    case TerminationReason::BadDemo:
        return "bad_demo";
    case TerminationReason::Unsafe:
        return "unsafe";
    case TerminationReason::Timeout:
        return "timeout";
    }
    return "operator_stop";
}

bool IsValidSegmentStatusValue(const std::string& value)
{
    return value == SegmentStatusName(SegmentStatus::Clean) ||
           value == SegmentStatusName(SegmentStatus::Usable) ||
           value == SegmentStatusName(SegmentStatus::Discard);
}

bool IsValidSuccessValue(const std::string& value)
{
    return value == SuccessLabelName(SuccessLabel::Success) ||
           value == SuccessLabelName(SuccessLabel::Partial) ||
           value == SuccessLabelName(SuccessLabel::Fail);
}

bool IsValidTerminationReasonValue(const std::string& value)
{
    return value == TerminationReasonName(TerminationReason::GoalReached) ||
           value == TerminationReasonName(TerminationReason::NearGoalStop) ||
           value == TerminationReasonName(TerminationReason::Occluded) ||
           value == TerminationReasonName(TerminationReason::TargetLost) ||
           value == TerminationReasonName(TerminationReason::OperatorStop) ||
           value == TerminationReasonName(TerminationReason::BadDemo) ||
           value == TerminationReasonName(TerminationReason::Unsafe) ||
           value == TerminationReasonName(TerminationReason::Timeout);
}

UiActionResult ActionOk(const std::string& message, const std::string& episodeId = "")
{
    UiActionResult result;
    result.ok = true;
    result.code = "ok";
    result.message = message;
    result.episodeId = episodeId;
    return result;
}

UiActionResult ActionError(const std::string& code, const std::string& message)
{
    UiActionResult result;
    result.ok = false;
    result.code = code;
    result.message = message;
    return result;
}

std::optional<std::string> ExtractJsonStringFieldLocal(const std::string& body, const std::string& key)
{
    const std::string marker = "\"" + key + "\"";
    const size_t keyPos = body.find(marker);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t colonPos = body.find(':', keyPos + marker.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t quotePos = body.find('"', colonPos + 1);
    if (quotePos == std::string::npos)
    {
        return std::nullopt;
    }
    std::string value;
    bool escaping = false;
    for (size_t index = quotePos + 1; index < body.size(); ++index)
    {
        const char ch = body[index];
        if (escaping)
        {
            value.push_back(ch);
            escaping = false;
            continue;
        }
        if (ch == '\\')
        {
            escaping = true;
            continue;
        }
        if (ch == '"')
        {
            return value;
        }
        value.push_back(ch);
    }
    return std::nullopt;
}

std::optional<std::vector<std::string>> ExtractJsonStringArrayFieldLocal(const std::string& body, const std::string& key)
{
    const std::string marker = "\"" + key + "\"";
    const size_t keyPos = body.find(marker);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t colonPos = body.find(':', keyPos + marker.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t arrayBegin = body.find('[', colonPos + 1);
    if (arrayBegin == std::string::npos)
    {
        return std::nullopt;
    }

    std::vector<std::string> values;
    size_t index = arrayBegin + 1;
    while (index < body.size())
    {
        index = body.find_first_not_of(" \t\r\n,", index);
        if (index == std::string::npos)
        {
            return std::nullopt;
        }
        if (body[index] == ']')
        {
            return values;
        }
        if (body[index] != '"')
        {
            return std::nullopt;
        }

        std::string value;
        bool escaping = false;
        ++index;
        while (index < body.size())
        {
            const char ch = body[index++];
            if (escaping)
            {
                value.push_back(ch);
                escaping = false;
                continue;
            }
            if (ch == '\\')
            {
                escaping = true;
                continue;
            }
            if (ch == '"')
            {
                values.push_back(value);
                break;
            }
            value.push_back(ch);
        }
        if (escaping)
        {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<double> ExtractJsonNumberFieldLocal(const std::string& body, const std::string& key)
{
    const std::string marker = "\"" + key + "\"";
    const size_t keyPos = body.find(marker);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t colonPos = body.find(':', keyPos + marker.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }
    size_t begin = body.find_first_of("-0123456789", colonPos + 1);
    if (begin == std::string::npos)
    {
        return std::nullopt;
    }
    size_t end = begin;
    while (end < body.size() &&
           (std::isdigit(static_cast<unsigned char>(body[end])) || body[end] == '.' || body[end] == '-'))
    {
        ++end;
    }
    try
    {
        return std::stod(body.substr(begin, end - begin));
    }
    catch (...)
    {
        return std::nullopt;
    }
}

std::optional<bool> ExtractJsonBoolFieldLocal(const std::string& body, const std::string& key)
{
    const std::string marker = "\"" + key + "\"";
    const size_t keyPos = body.find(marker);
    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t colonPos = body.find(':', keyPos + marker.size());
    if (colonPos == std::string::npos)
    {
        return std::nullopt;
    }
    const size_t truePos = body.find("true", colonPos + 1);
    const size_t falsePos = body.find("false", colonPos + 1);
    const size_t nextPos = body.find_first_not_of(" \t\r\n", colonPos + 1);
    if (nextPos == std::string::npos)
    {
        return std::nullopt;
    }
    if (truePos == nextPos)
    {
        return true;
    }
    if (falsePos == nextPos)
    {
        return false;
    }
    return std::nullopt;
}

bool RunShellCommandCapture(const std::string& command, std::string& output, int& exitCode)
{
    output.clear();
    exitCode = -1;
    const std::string wrapped = "/bin/bash -lc " + ShellQuote(command + " 2>&1");
    FILE* pipe = ::popen(wrapped.c_str(), "r");
    if (!pipe)
    {
        return false;
    }
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe))
    {
        output += buffer;
    }
    const int status = ::pclose(pipe);
    if (WIFEXITED(status))
    {
        exitCode = WEXITSTATUS(status);
    }
    else
    {
        exitCode = -1;
    }
    return true;
}

std::optional<std::vector<uint8_t>> HttpGetLoopbackBinary(int port, const std::string& path, int timeoutMs)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        return std::nullopt;
    }

    timeval timeout{};
    timeout.tv_sec = timeoutMs / 1000;
    timeout.tv_usec = (timeoutMs % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    {
        ::close(fd);
        return std::nullopt;
    }

    const std::string request =
        "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if (::send(fd, request.data(), request.size(), 0) < 0)
    {
        ::close(fd);
        return std::nullopt;
    }

    std::vector<uint8_t> response;
    uint8_t chunk[8192];
    while (true)
    {
        const ssize_t bytes = ::recv(fd, chunk, sizeof(chunk), 0);
        if (bytes <= 0)
        {
            break;
        }
        response.insert(response.end(), chunk, chunk + bytes);
    }
    ::close(fd);

    const std::string headerDelimiter = "\r\n\r\n";
    const auto it = std::search(response.begin(), response.end(), headerDelimiter.begin(), headerDelimiter.end());
    if (it == response.end())
    {
        return std::nullopt;
    }
    const size_t headerSize = static_cast<size_t>(std::distance(response.begin(), it)) + headerDelimiter.size();
    const std::string header(response.begin(), response.begin() + static_cast<std::ptrdiff_t>(headerSize));
    if (header.find("200 OK") == std::string::npos)
    {
        return std::nullopt;
    }

    std::vector<uint8_t> body(response.begin() + static_cast<std::ptrdiff_t>(headerSize), response.end());
    return body;
}

std::optional<std::string> HttpGetLoopbackText(int port, const std::string& path, int timeoutMs)
{
    const auto binary = HttpGetLoopbackBinary(port, path, timeoutMs);
    if (!binary.has_value())
    {
        return std::nullopt;
    }
    return std::string(binary->begin(), binary->end());
}

int FindFreeLoopbackPort()
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        return -1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(0);
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    {
        ::close(fd);
        return -1;
    }

    socklen_t addrLen = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addrLen) != 0)
    {
        ::close(fd);
        return -1;
    }

    const int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

struct D435iCaptureReference
{
    std::string captureMetadataPath;
    std::string bagPath;
    std::string depthTopicSemantics;
};

std::optional<D435iCaptureReference> LoadD435iCaptureReference(const fs::path& sessionDir)
{
    const fs::path metadataPath = sessionDir / "d435i" / "d435i_capture.json";
    std::error_code ec;
    if (!fs::exists(metadataPath, ec) || ec)
    {
        return std::nullopt;
    }

    std::ifstream input(metadataPath);
    if (!input.is_open())
    {
        return std::nullopt;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string body = buffer.str();

    D435iCaptureReference reference;
    reference.captureMetadataPath = "d435i/d435i_capture.json";
    reference.bagPath = ExtractJsonStringFieldLocal(body, "bag_path").value_or("");
    reference.depthTopicSemantics = ExtractJsonStringFieldLocal(body, "depth_topic_semantics").value_or("");

    if (reference.bagPath.empty())
    {
        const fs::path bagDir = sessionDir / "d435i";
        for (const auto& entry : fs::directory_iterator(bagDir, ec))
        {
            if (ec)
            {
                break;
            }
            if (!entry.is_regular_file())
            {
                continue;
            }
            if (entry.path().extension() != ".bag")
            {
                continue;
            }
            reference.bagPath = fs::relative(entry.path(), sessionDir, ec).string();
            if (ec)
            {
                ec.clear();
                reference.bagPath = entry.path().filename().string();
            }
            break;
        }
    }

    if (reference.depthTopicSemantics.empty())
    {
        const bool alignedDepth = ExtractJsonBoolFieldLocal(body, "align_depth_applied").value_or(false);
        reference.depthTopicSemantics = alignedDepth ? "aligned_depth" : "raw_unaligned_depth";
    }

    if (reference.bagPath.empty())
    {
        return std::nullopt;
    }
    return reference;
}

void ApplyDefaultsFile(const fs::path& defaultsPath, Config& config)
{
    std::ifstream input(defaultsPath);
    if (!input.is_open())
    {
        return;
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string body = buffer.str();

    if (const auto value = ExtractJsonStringFieldLocal(body, "scene_id"); value.has_value())
    {
        config.sceneId = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "operator_id"); value.has_value())
    {
        config.operatorId = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "instruction"); value.has_value())
    {
        config.instruction = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "task_family"); value.has_value())
    {
        config.taskFamily = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "target_type"); value.has_value())
    {
        config.targetType = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "target_description"); value.has_value())
    {
        config.targetDescription = value.value();
    }
    if (const auto value = ExtractJsonStringFieldLocal(body, "collector_notes"); value.has_value())
    {
        config.collectorNotes = value.value();
    }
    if (const auto value = ExtractJsonNumberFieldLocal(body, "cmd_vx_max"); value.has_value())
    {
        config.cmdVxMax = value.value();
    }
    if (const auto value = ExtractJsonNumberFieldLocal(body, "cmd_vy_max"); value.has_value())
    {
        config.cmdVyMax = value.value();
    }
    if (const auto value = ExtractJsonNumberFieldLocal(body, "cmd_wz_max"); value.has_value())
    {
        config.cmdWzMax = value.value();
    }
}

void ApplyCollectorDefaults(const fs::path& collectorRoot, Config& config)
{
    const fs::path defaultsPath = CollectorDefaultsPath(collectorRoot);
    if (fs::exists(defaultsPath))
    {
        ApplyDefaultsFile(defaultsPath, config);
        return;
    }
    const fs::path legacyDefaultsPath = LegacyCollectorDefaultsPath(collectorRoot);
    if (fs::exists(legacyDefaultsPath))
    {
        ApplyDefaultsFile(legacyDefaultsPath, config);
    }
}

std::optional<Config::InputBackend> ParseInputBackend(const std::string& value)
{
    if (value == "wireless" || value == "wireless_controller" || value == "controller" || value == "gamepad")
    {
        return Config::InputBackend::WirelessController;
    }
    if (value == "evdev")
    {
        return Config::InputBackend::Evdev;
    }
    return std::nullopt;
}

EffectiveControlAction ResolveControlAction(const VelocityCommand& rawAction, double sampleTimestamp)
{
    EffectiveControlAction resolved;
    resolved.timestamp = sampleTimestamp;
    resolved.command = rawAction;
    resolved.command.timestamp = sampleTimestamp;
    resolved.command.valid = true;
    return resolved;
}

TrajectoryMotionGateConfig BuildTrajectoryMotionGateConfig(double sampleHz)
{
    const double safeHz = std::max(sampleHz, 1.0);
    TrajectoryMotionGateConfig config;
    config.startConsecutiveFrames = std::max<size_t>(3, static_cast<size_t>(std::lround(0.35 * safeHz)));
    config.stopConsecutiveFrames = std::max<size_t>(5, static_cast<size_t>(std::lround(0.60 * safeHz)));
    config.preRollFrames = std::max<size_t>(3, static_cast<size_t>(std::lround(0.40 * safeHz)));
    return config;
}

void PrintUsage(const char* program)
{
    std::cout
        << "用法: " << program << " [--network-interface IFACE] --scene-id SCENE --operator-id OPERATOR [options]\n\n"
        << "Options:\n"
        << "  --output-dir PATH        数据集根目录（默认：<collector>/" << kDefaultDataDirName << ")\n"
        << "  --loop-hz FLOAT          Control 和 logging loop frequency (default: 50.0)\n"
        << "  --video-poll-hz FLOAT    Camera polling frequency (default: 20.0)\n"
        << "  --input-backend MODE     Input backend: wireless_controller or evdev (default: wireless_controller)\n"
        << "  --input-device PATH      evdev device path (default: auto-detect keyboard)\n"
        << "  --scene-id TEXT          Required scene identifier\n"
        << "  --operator-id TEXT       Required operator identifier\n"
        << "  --instruction TEXT       Optional manual instruction override; otherwise auto-generated from task + target\n"
        << "  --task-family TEXT       Task family or alias, e.g. goal_navigation / navigate / visual_following / follow\n"
        << "  --task TEXT              Alias of --task-family for quick launch\n"
        << "  --target-type TEXT       Optional coarse target type, e.g. door / person / obstacle\n"
        << "  --target-description TEXT  Optional free-text target description; left/right/near/far can be derived offline\n"
        << "  --target TEXT            Alias of --target-description for quick launch; enough for auto-composed instruction\n"
        << "  --collector-notes TEXT   Optional free-text notes stored with each episode\n"
        << "  --cmd-vx-max FLOAT       Max forward/backward speed in m/s\n"
        << "  --cmd-vy-max FLOAT       Max strafe speed in m/s\n"
        << "  --cmd-wz-max FLOAT       Max yaw speed in rad/s\n"
        << "  --d435i                  Enable D435i ROS RGB-D rosbag capture for the whole session\n"
        << "  --d435i-no-launch        Record D435i topics only; assume rs_camera.launch is already running\n"
        << "  --d435i-ros-setup PATH   ROS setup script (default: /opt/ros/noetic/setup.bash)\n"
        << "  --d435i-bag-name NAME    D435i bag basename under <session>/d435i/ (default: d435i_rgbd)\n"
        << "  --d435i-serial TEXT      Optional D435i serial number for launch selection\n"
        << "  --d435i-color-profile P  D435i color profile, e.g. 640x480x30\n"
        << "  --d435i-depth-profile P  D435i depth profile, e.g. 640x480x30\n"
        << "  --d435i-align-depth      Launch D435i with depth alignment enabled\n"
        << "  --d435i-launch-file S    Launch target, e.g. 'realsense2_camera rs_camera.launch'\n"
        << "  --d435i-depth-preview-min-m FLOAT  Depth preview min range in meters (default: 0.2)\n"
        << "  --d435i-depth-preview-max-m FLOAT  Depth preview max range in meters (default: 3.0)\n"
        << "  --d435i-topic-timeout FLOAT  Topic ready timeout in seconds (default: 12.0)\n"
        << "  --web-ui                 Enable optional local Web UI\n"
        << "  --preview-ui             Launch Web UI without Go2 using local preview state/image\n"
        << "  --web-port INT           Local Web UI port (default: 8080)\n"
        << "  --help                   Show this help\n\n"
        << "Wireless controller:\n"
        << "  原生手柄直通已关闭：无线手柄按键由 collector 统一接管\n"
        << "  离散控制：D-pad Up/Down/Left/Right = W/S/A/D，X/B = Q/E\n"
        << "  when label_ready, D-pad Up/Right/Down/Left switches to label 1/2/3/4\n"
        << "  Start start capture  A stop capture  Y discard current segment\n"
        << "  R2 emergency stop  F1 clear fault  F2 toggle stand up/down\n"
        << "Evdev keyboard:\n"
        << "  W/S forward-backward  A/D strafe  Q/E turn\n"
        << "  R start  T stop  ESC discard  Space estop  C clear fault  V toggle stand  P status  H help\n"
        << "Input:\n"
        << "  wireless_controller subscribes Go2 native controller topic rt/wirelesscontroller\n"
        << "  evdev supports true multi-key press/release 和 smoother diagonal motion\n"
        << "Recording flow:\n"
        << "  trajectory flow is fixed: R starts buffering, effective motion gates logging, and T waits for motion settle before labeling\n";
}

std::optional<Config> ParseArgs(int argc, char** argv, std::string& error)
{
    Config config;
    const fs::path collectorRoot = CollectorRootFromArgv0(argv[0]);
    config.collectorRoot = collectorRoot;
    config.outputDir = collectorRoot / kDefaultDataDirName;
    ApplyCollectorDefaults(collectorRoot, config);
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--network-interface")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--network-interface";
                return std::nullopt;
            }
            config.networkInterface = argv[++index];
        }
        else if (arg == "--output-dir")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--output-dir";
                return std::nullopt;
            }
            fs::path outputDir = argv[++index];
            if (!outputDir.is_absolute())
            {
                outputDir = collectorRoot / outputDir;
            }
            config.outputDir = outputDir;
        }
        else if (arg == "--loop-hz")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--loop-hz";
                return std::nullopt;
            }
            config.loopHz = std::stod(argv[++index]);
        }
        else if (arg == "--video-poll-hz")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--video-poll-hz";
                return std::nullopt;
            }
            config.videoPollHz = std::stod(argv[++index]);
        }
        else if (arg == "--input-backend")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--input-backend";
                return std::nullopt;
            }
            const auto backend = ParseInputBackend(argv[++index]);
            if (!backend.has_value())
            {
                error = "输入后端必须是 wireless_controller 或 evdev";
                return std::nullopt;
            }
            config.inputBackend = backend.value();
        }
        else if (arg == "--wireless-motion-mode")
        {
            error = "--wireless-motion-mode 已移除；无线手柄现在始终使用原生直通，如需更平缓控制请改用 --input-backend evdev";
            return std::nullopt;
        }
        else if (arg == "--capture-mode")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--capture-mode";
                return std::nullopt;
            }
            if (!IsSupportedCaptureModeValue(argv[++index]))
            {
                error = "采集模式当前仅支持 trajectory";
                return std::nullopt;
            }
        }
        else if (arg == "--input-device")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--input-device";
                return std::nullopt;
            }
            config.inputDevice = argv[++index];
        }
        else if (arg == "--scene-id")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--scene-id";
                return std::nullopt;
            }
            config.sceneId = argv[++index];
        }
        else if (arg == "--operator-id")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--operator-id";
                return std::nullopt;
            }
            config.operatorId = argv[++index];
        }
        else if (arg == "--instruction")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--instruction";
                return std::nullopt;
            }
            config.instruction = argv[++index];
        }
        else if (arg == "--task-family")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--task-family";
                return std::nullopt;
            }
            config.taskFamily = argv[++index];
        }
        else if (arg == "--task" || arg == "--task-type")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：" + arg;
                return std::nullopt;
            }
            config.taskFamily = argv[++index];
        }
        else if (arg == "--target-type")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--target-type";
                return std::nullopt;
            }
            config.targetType = argv[++index];
        }
        else if (arg == "--target-description")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--target-description";
                return std::nullopt;
            }
            config.targetDescription = argv[++index];
        }
        else if (arg == "--target")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--target";
                return std::nullopt;
            }
            config.targetDescription = argv[++index];
        }
        else if (arg == "--target-instance-id" || arg == "--task-tags")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：" + arg;
                return std::nullopt;
            }
            ++index;
        }
        else if (arg == "--collector-notes")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--collector-notes";
                return std::nullopt;
            }
            config.collectorNotes = argv[++index];
        }
        else if (arg == "--cmd-vx-max")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--cmd-vx-max";
                return std::nullopt;
            }
            config.cmdVxMax = std::stod(argv[++index]);
        }
        else if (arg == "--cmd-vy-max")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--cmd-vy-max";
                return std::nullopt;
            }
            config.cmdVyMax = std::stod(argv[++index]);
        }
        else if (arg == "--cmd-wz-max")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--cmd-wz-max";
                return std::nullopt;
            }
            config.cmdWzMax = std::stod(argv[++index]);
        }
        else if (arg == "--d435i")
        {
            config.d435iEnabled = true;
        }
        else if (arg == "--d435i-no-launch")
        {
            config.d435iEnabled = true;
            config.d435iAutostartNode = false;
        }
        else if (arg == "--d435i-ros-setup")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-ros-setup";
                return std::nullopt;
            }
            config.d435iRosSetupPath = argv[++index];
        }
        else if (arg == "--d435i-bag-name")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-bag-name";
                return std::nullopt;
            }
            config.d435iBagBaseName = argv[++index];
        }
        else if (arg == "--d435i-serial")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-serial";
                return std::nullopt;
            }
            config.d435iSerial = argv[++index];
        }
        else if (arg == "--d435i-color-profile")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-color-profile";
                return std::nullopt;
            }
            config.d435iColorProfile = argv[++index];
        }
        else if (arg == "--d435i-depth-profile")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-depth-profile";
                return std::nullopt;
            }
            config.d435iDepthProfile = argv[++index];
        }
        else if (arg == "--d435i-align-depth")
        {
            config.d435iAlignDepth = true;
        }
        else if (arg == "--d435i-launch-file")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-launch-file";
                return std::nullopt;
            }
            config.d435iLaunchFile = argv[++index];
        }
        else if (arg == "--d435i-depth-preview-min-m")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-depth-preview-min-m";
                return std::nullopt;
            }
            config.d435iDepthPreviewMinMeters = std::stod(argv[++index]);
        }
        else if (arg == "--d435i-depth-preview-max-m")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-depth-preview-max-m";
                return std::nullopt;
            }
            config.d435iDepthPreviewMaxMeters = std::stod(argv[++index]);
        }
        else if (arg == "--d435i-topic-timeout")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--d435i-topic-timeout";
                return std::nullopt;
            }
            config.d435iTopicReadyTimeoutSeconds = std::stod(argv[++index]);
        }
        else if (arg == "--web-ui")
        {
            config.webUiEnabled = true;
        }
        else if (arg == "--preview-ui")
        {
            config.previewUi = true;
            config.webUiEnabled = true;
        }
        else if (arg == "--web-port")
        {
            if (index + 1 >= argc)
            {
                error = "参数缺少取值：--web-port";
                return std::nullopt;
            }
            config.webPort = std::stoi(argv[++index]);
        }
        else if (arg == "--help")
        {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        else
        {
            error = "未知参数：" + arg;
            return std::nullopt;
        }
    }

    config.sceneId = Trim(config.sceneId);
    config.operatorId = Trim(config.operatorId);
    config.instruction = Trim(config.instruction);
    config.taskFamily = NormalizeTaskFamily(config.taskFamily);
    config.targetType = Trim(config.targetType);
    config.targetDescription = Trim(config.targetDescription);
    config.collectorNotes = Trim(config.collectorNotes);
    config.d435iRosSetupPath = Trim(config.d435iRosSetupPath);
    config.d435iBagBaseName = Trim(config.d435iBagBaseName);
    config.d435iSerial = Trim(config.d435iSerial);
    config.d435iColorProfile = Trim(config.d435iColorProfile);
    config.d435iDepthProfile = Trim(config.d435iDepthProfile);
    config.d435iLaunchFile = Trim(config.d435iLaunchFile);
    if (config.targetType.empty() && !config.targetDescription.empty() &&
        config.targetDescription.find_first_of(" \t") == std::string::npos)
    {
        config.targetType = config.targetDescription;
    }

    if (config.previewUi)
    {
        if (config.sceneId.empty())
        {
            config.sceneId = kPreviewSceneId;
        }
        if (config.operatorId.empty())
        {
            config.operatorId = kPreviewOperatorId;
        }
        if (config.instruction.empty())
        {
            config.instruction = kPreviewInstruction;
        }
        if (config.taskFamily.empty())
        {
            config.taskFamily = kPreviewTaskFamily;
        }
        if (config.targetType.empty())
        {
            config.targetType = kPreviewTargetType;
        }
        if (config.targetDescription.empty())
        {
            config.targetDescription = kPreviewTargetDescription;
        }
    }

    if (config.instruction.empty())
    {
        if (const auto generatedInstruction = AutoComposeInstruction(
                config.taskFamily, config.targetType, config.targetDescription, config.sceneId, config.operatorId);
            generatedInstruction.has_value())
        {
            config.instruction = generatedInstruction.value();
            config.instructionSource = "auto_template";
        }
    }

    if (!config.previewUi && config.networkInterface.empty())
    {
        error = "--network-interface 为必填参数";
        return std::nullopt;
    }
    if (config.sceneId.empty())
    {
        error = "--scene-id 为必填参数";
        return std::nullopt;
    }
    if (config.operatorId.empty())
    {
        error = "--operator-id 为必填参数";
        return std::nullopt;
    }
    if (config.instruction.empty())
    {
        if (config.taskFamily.empty())
        {
            error = "必须提供 --instruction，或提供 --task-family/--task + --target";
        }
        else if (!CanonicalTaskFamily(config.taskFamily).has_value())
        {
            error = "当前 --task-family 不支持自动生成 instruction；请使用 navigate/follow 等已支持任务，或显式提供 --instruction";
        }
        else if (ResolveTargetText(config.targetType, config.targetDescription).empty())
        {
            error = "未提供 --instruction 时，必须提供 --target、--target-description 或 --target-type";
        }
        else
        {
            error = "自动生成 instruction 失败，请显式提供 --instruction";
        }
        return std::nullopt;
    }
    if (config.loopHz <= 0.0 || config.videoPollHz <= 0.0)
    {
        error = "频率参数必须为正数";
        return std::nullopt;
    }
    if (config.cmdVxMax <= 0.0 || config.cmdVyMax <= 0.0 || config.cmdWzMax <= 0.0)
    {
        error = "速度上限参数必须为正数";
        return std::nullopt;
    }
    if (config.webPort <= 0 || config.webPort > 65535)
    {
        error = "--web-port 必须在 1 到 65535 之间";
        return std::nullopt;
    }
    if (config.d435iEnabled)
    {
        if (config.previewUi)
        {
            error = "--d435i 与 --preview-ui 不能同时启用";
            return std::nullopt;
        }
        if (config.d435iRosSetupPath.empty())
        {
            error = "--d435i-ros-setup 不能为空";
            return std::nullopt;
        }
        if (config.d435iBagBaseName.empty())
        {
            error = "--d435i-bag-name 不能为空";
            return std::nullopt;
        }
        if (config.d435iColorProfile.empty() || config.d435iDepthProfile.empty())
        {
            error = "D435i color/depth profile 不能为空";
            return std::nullopt;
        }
        if (config.d435iLaunchFile.empty())
        {
            error = "--d435i-launch-file 不能为空";
            return std::nullopt;
        }
        if (SplitWhitespaceTokens(config.d435iLaunchFile).empty())
        {
            error = "--d435i-launch-file 至少要包含 roslaunch 目标";
            return std::nullopt;
        }
        if (config.d435iNodeWarmupSeconds < 0.0)
        {
            error = "D435i 预热时间不能为负数";
            return std::nullopt;
        }
        if (config.d435iTopicReadyTimeoutSeconds <= 0.0)
        {
            error = "D435i topic timeout 必须为正数";
            return std::nullopt;
        }
        if (config.d435iDepthPreviewMinMeters < 0.0 || config.d435iDepthPreviewMaxMeters <= config.d435iDepthPreviewMinMeters)
        {
            error = "D435i 深度预览范围非法";
            return std::nullopt;
        }
    }
    if (config.inputBackend == Config::InputBackend::Evdev && config.inputDevice.empty())
    {
        const auto detected = collector::input::FindDefaultKeyboardDevice();
        if (!detected.has_value())
        {
            error = "无法在 /dev/input 下自动检测键盘输入设备";
            return std::nullopt;
        }
        config.inputDevice = detected.value();
    }

    return config;
}

class TrajectoryLogger
{
public:
    explicit TrajectoryLogger(const fs::path& outputDir)
        : sessionId_(FormatTimestampForPath(NowSeconds(), "%Y%m%d_%H%M%S")),
          outputDir_(outputDir / sessionId_),
          episodesDir_(outputDir_ / "episodes"),
          imagesDir_(outputDir_ / "images"),
          indexPath_(outputDir_ / "index.json")
    {
        fs::create_directories(outputDir_);
        fs::create_directories(episodesDir_);
        fs::create_directories(imagesDir_);
        RewriteIndexLocked();
    }

    const fs::path& OutputDir() const
    {
        return outputDir_;
    }

    bool HasRecordedEpisodes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return !episodeHistory_.empty();
    }

    void CleanupIfEmpty()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capturing_ || !pendingFrames_.empty())
        {
            return;
        }
        bool hasEpisodeFiles = false;
        bool hasAuxiliaryArtifacts = false;
        std::error_code scanEc;
        if (fs::exists(episodesDir_, scanEc) && !scanEc)
        {
            for (const auto& entry : fs::directory_iterator(episodesDir_, scanEc))
            {
                if (scanEc)
                {
                    break;
                }
                if (!entry.is_regular_file())
                {
                    continue;
                }
                if (entry.path().extension() == ".json")
                {
                    hasEpisodeFiles = true;
                    break;
                }
            }
        }
        const fs::path d435iDir = outputDir_ / "d435i";
        std::error_code auxEc;
        if (fs::exists(d435iDir, auxEc) && !auxEc)
        {
            for (const auto& entry : fs::recursive_directory_iterator(d435iDir, auxEc))
            {
                if (auxEc)
                {
                    break;
                }
                if (entry.is_regular_file())
                {
                    hasAuxiliaryArtifacts = true;
                    break;
                }
            }
        }
        if (!episodeHistory_.empty() || hasEpisodeFiles || hasAuxiliaryArtifacts)
        {
            return;
        }
        std::error_code ec;
        fs::remove_all(outputDir_, ec);
    }

    struct Status
    {
        bool capturing = false;
        bool pendingLabel = false;
        size_t bufferedFrames = 0;
        std::string pendingEpisodeId;
        std::string lastEpisodeId;
        double startTimestamp = 0.0;
        double endTimestamp = 0.0;
    };

    Status GetStatus() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Status status;
        status.capturing = capturing_;
        status.pendingLabel = pendingLabel_;
        status.bufferedFrames = pendingFrames_.size();
        status.pendingEpisodeId = pendingEpisodeId_;
        status.lastEpisodeId = lastRecordedEpisodeId_;
        if (!pendingFrames_.empty())
        {
            status.startTimestamp = pendingFrames_.front().timestamp;
            status.endTimestamp = pendingFrames_.back().timestamp;
        }
        return status;
    }

    void BeginSegment(
        const std::string& sceneId,
        const std::string& operatorId,
        const TaskMetadata& taskMetadata,
        const std::optional<TrajectoryMotionGateConfig>& trajectoryGate = std::nullopt)
    {
        const std::string trimmedSceneId = Trim(sceneId);
        const std::string trimmedOperatorId = Trim(operatorId);
        if (trimmedSceneId.empty())
        {
            throw std::runtime_error("scene_id 不能为空");
        }
        if (trimmedOperatorId.empty())
        {
            throw std::runtime_error("operator_id 不能为空");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        pendingFrames_.clear();
        pendingSceneId_ = trimmedSceneId;
        pendingOperatorId_ = trimmedOperatorId;
        pendingTaskMetadata_ = taskMetadata;
        pendingEpisodeId_.clear();
        trajectoryGateConfig_ = trajectoryGate;
        effectiveMotionStarted_ = false;
        stopReady_ = false;
        stopRequested_ = false;
        startCandidateCount_ = 0;
        stopCandidateCount_ = 0;
        preRollFrames_.clear();
        capturing_ = true;
        pendingLabel_ = false;
    }

    bool LogStep(
        double timestamp,
        const LatestState& state,
        const VelocityCommand& rawAction,
        const EffectiveControlAction& controlAction,
        const LatestImage& image,
        double stateTimestamp,
        double rawActionTimestamp,
        double imageTimestamp,
        bool motionInputActive)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!capturing_)
        {
            return false;
        }
        if (!image.valid || image.jpegBytes.empty())
        {
            return false;
        }

        EpisodeFrame frame{
            timestamp,
            stateTimestamp,
            controlAction.timestamp,
            rawActionTimestamp,
            imageTimestamp,
            state.velocityX,
            state.velocityY,
            state.velocityZ,
            state.yawSpeed,
            state.roll,
            state.pitch,
            state.yaw,
            state.positionX,
            state.positionY,
            state.positionZ,
            state.bodyHeight,
            state.errorCode,
            state.mode,
            state.gaitType,
            rawAction.vx,
            rawAction.vy,
            rawAction.wz,
            controlAction.command.vx,
            controlAction.command.vy,
            controlAction.command.wz,
            motionInputActive,
            image.jpegBytes,
        };

        if (!trajectoryGateConfig_.has_value())
        {
            pendingFrames_.push_back(std::move(frame));
            return true;
        }

        PushPreRollFrameLocked(frame);
        if (!effectiveMotionStarted_)
        {
            if (ShouldStartEffectiveMotionLocked(frame))
            {
                effectiveMotionStarted_ = true;
                pendingFrames_.insert(pendingFrames_.end(), preRollFrames_.begin(), preRollFrames_.end());
                preRollFrames_.clear();
                return true;
            }
            return false;
        }

        pendingFrames_.push_back(std::move(frame));
        if (stopRequested_)
        {
            if (ShouldStopEffectiveMotionLocked(pendingFrames_.back()))
            {
                stopReady_ = true;
            }
        }
        return true;
    }

    size_t EndSegmentForLabel()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!capturing_)
        {
            return 0;
        }
        capturing_ = false;
        stopRequested_ = false;
        stopReady_ = false;
        preRollFrames_.clear();
        startCandidateCount_ = 0;
        stopCandidateCount_ = 0;
        if (trajectoryGateConfig_.has_value())
        {
            TrimLeadingTrailingLowMotionFramesLocked();
        }
        pendingLabel_ = !pendingFrames_.empty();
        return pendingFrames_.size();
    }

    void DiscardPendingSegment()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        capturing_ = false;
        pendingLabel_ = false;
        pendingFrames_.clear();
        pendingEpisodeId_.clear();
        pendingSceneId_.clear();
        pendingOperatorId_.clear();
        pendingTaskMetadata_ = TaskMetadata{};
        trajectoryGateConfig_.reset();
        effectiveMotionStarted_ = false;
        stopReady_ = false;
        stopRequested_ = false;
        startCandidateCount_ = 0;
        stopCandidateCount_ = 0;
        preRollFrames_.clear();
    }

    void RequestTrajectoryStop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capturing_ && trajectoryGateConfig_.has_value())
        {
            stopRequested_ = true;
            stopCandidateCount_ = 0;
        }
    }

    bool IsTrajectoryStopRequested() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopRequested_;
    }

    bool HasEffectiveMotion() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return effectiveMotionStarted_ && !pendingFrames_.empty();
    }

    bool ConsumeTrajectoryStopReady()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stopReady_)
        {
            return false;
        }
        stopReady_ = false;
        return true;
    }

    std::optional<std::string> FinalizePendingSegment(const std::string& fallbackInstruction, const TaskMetadata& labelMetadata)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pendingLabel_ || pendingFrames_.empty())
        {
            return std::nullopt;
        }
        const TaskMetadata resolvedTaskMetadata = ResolveTaskMetadata(pendingTaskMetadata_, fallbackInstruction);
        const std::string segmentStatus = Trim(labelMetadata.segmentStatus);
        const std::string success = Trim(labelMetadata.success);
        const std::string terminationReason = Trim(labelMetadata.terminationReason);
        const std::string trimmedInstruction = Trim(resolvedTaskMetadata.instruction);
        const auto d435iCaptureReference = LoadD435iCaptureReference(outputDir_);
        if (trimmedInstruction.empty())
        {
            throw std::runtime_error("instruction 不能为空");
        }

        std::ostringstream episodeId;
        episodeId << "ep_" << std::setw(6) << std::setfill('0') << nextEpisodeIndex_++;
        pendingEpisodeId_ = episodeId.str();

        const fs::path episodeImageDir = imagesDir_ / pendingEpisodeId_;
        fs::create_directories(episodeImageDir);

        std::ostringstream episodeJson;
        episodeJson << std::fixed << std::setprecision(6);
        episodeJson << "{\n"
                    << "  \"schema_version\": " << JsonString(kSchemaVersion) << ",\n"
                    << "  \"session_id\": " << JsonString(sessionId_) << ",\n"
                    << "  \"episode_id\": " << JsonString(pendingEpisodeId_) << ",\n"
                    << "  \"instruction\": " << JsonString(trimmedInstruction) << ",\n"
                    << "  \"capture_mode\": " << JsonString(resolvedTaskMetadata.captureMode) << ",\n"
                    << "  \"task_family\": " << JsonString(resolvedTaskMetadata.taskFamily) << ",\n"
                    << "  \"target_type\": " << JsonString(resolvedTaskMetadata.targetType) << ",\n"
                    << "  \"target_label\": " << JsonString(resolvedTaskMetadata.targetLabel) << ",\n"
                    << "  \"target_description\": " << JsonString(resolvedTaskMetadata.targetDescription) << ",\n"
                    << "  \"collector_notes\": " << JsonString(resolvedTaskMetadata.collectorNotes) << ",\n"
                    << "  \"instruction_source\": " << JsonString(resolvedTaskMetadata.instructionSource) << ",\n"
                    << "  \"segment_status\": " << JsonString(segmentStatus) << ",\n"
                    << "  \"success\": " << JsonString(success) << ",\n"
                    << "  \"termination_reason\": " << JsonString(terminationReason) << ",\n"
                    << "  \"scene_id\": " << JsonString(pendingSceneId_) << ",\n"
                    << "  \"operator_id\": " << JsonString(pendingOperatorId_) << ",\n"
                    << "  \"source_type\": " << JsonString(kRealSourceType);
        if (d435iCaptureReference.has_value())
        {
            episodeJson << ",\n  \"d435i_capture\": {\n"
                        << "    \"capture_metadata_path\": " << JsonString(d435iCaptureReference->captureMetadataPath) << ",\n"
                        << "    \"bag_path\": " << JsonString(d435iCaptureReference->bagPath) << ",\n"
                        << "    \"depth_topic_semantics\": " << JsonString(d435iCaptureReference->depthTopicSemantics) << "\n"
                        << "  }";
        }
        episodeJson << ",\n"
                    << "  \"frames\": [\n";

        for (size_t index = 0; index < pendingFrames_.size(); ++index)
        {
            const auto& frame = pendingFrames_[index];
            std::ostringstream filename;
            filename << FormatTimestampForPath(frame.timestamp, "%Y%m%d_%H%M%S")
                     << "_" << std::setw(3) << std::setfill('0') << (index + 1) << ".jpg";
            const fs::path imagePath = episodeImageDir / filename.str();
            std::ofstream imageFile(imagePath, std::ios::binary);
            if (!imageFile.is_open())
            {
                throw std::runtime_error("写入图像文件失败");
            }
            imageFile.write(reinterpret_cast<const char*>(frame.jpegBytes.data()), static_cast<std::streamsize>(frame.jpegBytes.size()));
            const std::string imageRelPath = fs::relative(imagePath, outputDir_).string();
            float previousControlActionVx = 0.0f;
            float previousControlActionVy = 0.0f;
            float previousControlActionWz = 0.0f;
            if (index > 0)
            {
                const auto& previousFrame = pendingFrames_[index - 1];
                previousControlActionVx = previousFrame.controlActionVx;
                previousControlActionVy = previousFrame.controlActionVy;
                previousControlActionWz = previousFrame.controlActionWz;
            }

            episodeJson << "    {\n"
                        << "      \"timestamp\": " << frame.timestamp << ",\n"
                        << "      \"image\": " << JsonString(imageRelPath) << ",\n"
                        << "      \"instruction\": " << JsonString(trimmedInstruction) << ",\n"
                        << "      \"state\": {\n"
                        << "        \"vx\": " << frame.stateVx << ",\n"
                        << "        \"vy\": " << frame.stateVy << ",\n"
                        << "        \"vz\": " << frame.stateVz << ",\n"
                        << "        \"wz\": " << frame.stateWz << ",\n"
                        << "        \"roll\": " << frame.stateRoll << ",\n"
                        << "        \"pitch\": " << frame.statePitch << ",\n"
                        << "        \"yaw\": " << frame.stateYaw << ",\n"
                        << "        \"x\": " << frame.statePositionX << ",\n"
                        << "        \"y\": " << frame.statePositionY << ",\n"
                        << "        \"z\": " << frame.statePositionZ << ",\n"
                        << "        \"body_height\": " << frame.stateBodyHeight << ",\n"
                        << "        \"error_code\": " << frame.stateErrorCode << ",\n"
                        << "        \"mode\": " << static_cast<int>(frame.stateMode) << ",\n"
                        << "        \"gait_type\": " << static_cast<int>(frame.stateGaitType) << "\n"
                        << "      },\n"
                        << "      \"raw_action\": {\n"
                        << "        \"vx\": " << frame.rawActionVx << ",\n"
                        << "        \"vy\": " << frame.rawActionVy << ",\n"
                        << "        \"wz\": " << frame.rawActionWz << ",\n"
                        << "        \"camera_pitch\": 0.0,\n"
                        << "        \"keys\": 0\n"
                        << "      },\n"
                        << "      \"control_action\": {\n"
                        << "        \"vx\": " << frame.controlActionVx << ",\n"
                        << "        \"vy\": " << frame.controlActionVy << ",\n"
                        << "        \"wz\": " << frame.controlActionWz << "\n"
                        << "      },\n"
                        << "      \"previous_action\": {\n"
                        << "        \"vx\": " << previousControlActionVx << ",\n"
                        << "        \"vy\": " << previousControlActionVy << ",\n"
                        << "        \"wz\": " << previousControlActionWz << "\n"
                        << "      },\n"
                        << "      \"meta\": {\n"
                        << "        \"schema_version\": " << JsonString(kSchemaVersion) << ",\n"
                        << "        \"session_id\": " << JsonString(sessionId_) << ",\n"
                        << "        \"episode_id\": " << JsonString(pendingEpisodeId_) << ",\n"
                        << "        \"source_type\": " << JsonString(kRealSourceType) << ",\n"
                        << "        \"camera_interface_source\": " << JsonString("unitree_sdk2.go2.video.GetImageSample") << ",\n"
                        << "        \"capture_mode\": " << JsonString(resolvedTaskMetadata.captureMode) << ",\n"
                        << "        \"task_family\": " << JsonString(resolvedTaskMetadata.taskFamily) << ",\n"
                        << "        \"target_type\": " << JsonString(resolvedTaskMetadata.targetType) << ",\n"
                        << "        \"target_label\": " << JsonString(resolvedTaskMetadata.targetLabel) << ",\n"
                        << "        \"target_description\": " << JsonString(resolvedTaskMetadata.targetDescription) << ",\n"
                        << "        \"instruction_source\": " << JsonString(resolvedTaskMetadata.instructionSource) << ",\n"
                        << "        \"segment_status\": " << JsonString(segmentStatus) << ",\n"
                        << "        \"success\": " << JsonString(success) << ",\n"
                        << "        \"termination_reason\": " << JsonString(terminationReason) << ",\n"
                        << "        \"scene_id\": " << JsonString(pendingSceneId_) << ",\n"
                        << "        \"operator_id\": " << JsonString(pendingOperatorId_) << ",\n"
                        << "        \"state_timestamp\": " << frame.stateTimestamp << ",\n"
                        << "        \"action_timestamp\": " << frame.actionTimestamp << ",\n"
                        << "        \"raw_action_timestamp\": " << frame.rawActionTimestamp << ",\n"
                        << "        \"control_action_timestamp\": " << frame.actionTimestamp << ",\n"
                        << "        \"image_timestamp\": " << frame.imageTimestamp << "\n"
                        << "      }\n"
                        << "    }";
            if (index + 1 != pendingFrames_.size())
            {
                episodeJson << ",";
            }
            episodeJson << "\n";
        }
        episodeJson << "  ]\n"
                    << "}\n";

        std::ofstream episodeFile(episodesDir_ / (pendingEpisodeId_ + ".json"), std::ios::out | std::ios::trunc);
        if (!episodeFile.is_open())
        {
            throw std::runtime_error("写入 episode 文件失败");
        }
        episodeFile << episodeJson.str();
        episodeFile.flush();

        EpisodeSummary summary;
        summary.episodeId = pendingEpisodeId_;
        summary.instruction = trimmedInstruction;
        summary.captureMode = resolvedTaskMetadata.captureMode;
        summary.taskFamily = resolvedTaskMetadata.taskFamily;
        summary.targetType = resolvedTaskMetadata.targetType;
        summary.targetLabel = resolvedTaskMetadata.targetLabel;
        summary.targetDescription = resolvedTaskMetadata.targetDescription;
        summary.collectorNotes = resolvedTaskMetadata.collectorNotes;
        summary.instructionSource = resolvedTaskMetadata.instructionSource;
        summary.segmentStatus = segmentStatus;
        summary.success = success;
        summary.terminationReason = terminationReason;
        summary.sceneId = pendingSceneId_;
        summary.operatorId = pendingOperatorId_;
        summary.numFrames = pendingFrames_.size();
        summary.startTimestamp = pendingFrames_.front().timestamp;
        summary.endTimestamp = pendingFrames_.back().timestamp;
        episodeHistory_[summary.episodeId] = summary;
        lastRecordedEpisodeId_ = summary.episodeId;
        RewriteIndexLocked();

        pendingFrames_.clear();
        pendingLabel_ = false;
        pendingSceneId_.clear();
        pendingOperatorId_.clear();
        pendingTaskMetadata_ = TaskMetadata{};

        return summary.episodeId;
    }

private:
    bool IsCommandAboveStartThresholdLocked(const EpisodeFrame& frame) const
    {
        const auto& config = trajectoryGateConfig_.value();
        return std::fabs(frame.rawActionVx) > config.vxStartThreshold ||
               std::fabs(frame.rawActionVy) > config.vyStartThreshold ||
               std::fabs(frame.rawActionWz) > config.wzStartThreshold;
    }

    bool IsCommandBelowStopThresholdLocked(const EpisodeFrame& frame) const
    {
        const auto& config = trajectoryGateConfig_.value();
        return std::fabs(frame.rawActionVx) <= config.vxStopThreshold &&
               std::fabs(frame.rawActionVy) <= config.vyStopThreshold &&
               std::fabs(frame.rawActionWz) <= config.wzStopThreshold;
    }

    bool IsLowMotionFrameLocked(const EpisodeFrame& frame) const
    {
        return !frame.motionInputActive && IsCommandBelowStopThresholdLocked(frame);
    }

    bool ShouldStartEffectiveMotionLocked(const EpisodeFrame& frame)
    {
        if (frame.motionInputActive && IsCommandAboveStartThresholdLocked(frame))
        {
            ++startCandidateCount_;
        }
        else
        {
            startCandidateCount_ = 0;
        }
        return startCandidateCount_ >= trajectoryGateConfig_->startConsecutiveFrames;
    }

    bool ShouldStopEffectiveMotionLocked(const EpisodeFrame& frame)
    {
        if (!frame.motionInputActive && IsCommandBelowStopThresholdLocked(frame))
        {
            ++stopCandidateCount_;
        }
        else
        {
            stopCandidateCount_ = 0;
        }
        return stopCandidateCount_ >= trajectoryGateConfig_->stopConsecutiveFrames;
    }

    void PushPreRollFrameLocked(const EpisodeFrame& frame)
    {
        preRollFrames_.push_back(frame);
        const size_t maxFrames = std::max<size_t>(1, trajectoryGateConfig_->preRollFrames);
        if (preRollFrames_.size() > maxFrames)
        {
            preRollFrames_.erase(preRollFrames_.begin());
        }
    }

    void TrimLeadingTrailingLowMotionFramesLocked()
    {
        if (!trajectoryGateConfig_.has_value() || pendingFrames_.empty())
        {
            return;
        }

        size_t startIndex = 0;
        while (startIndex < pendingFrames_.size() && IsLowMotionFrameLocked(pendingFrames_[startIndex]))
        {
            ++startIndex;
        }

        size_t endIndex = pendingFrames_.size();
        while (endIndex > startIndex && IsLowMotionFrameLocked(pendingFrames_[endIndex - 1]))
        {
            --endIndex;
        }

        if (startIndex == 0 && endIndex == pendingFrames_.size())
        {
            return;
        }

        if (startIndex >= endIndex)
        {
            pendingFrames_.clear();
            return;
        }

        pendingFrames_ = std::vector<EpisodeFrame>(pendingFrames_.begin() + static_cast<std::ptrdiff_t>(startIndex),
                                                   pendingFrames_.begin() + static_cast<std::ptrdiff_t>(endIndex));
    }

    void RewriteIndexLocked() const
    {
        std::vector<EpisodeSummary> summaries;
        summaries.reserve(episodeHistory_.size());
        for (const auto& item : episodeHistory_)
        {
            summaries.push_back(item.second);
        }
        std::sort(
            summaries.begin(),
            summaries.end(),
            [](const EpisodeSummary& lhs, const EpisodeSummary& rhs)
            {
                return lhs.episodeId < rhs.episodeId;
            });

        std::ofstream indexFile(indexPath_, std::ios::out | std::ios::trunc);
        if (!indexFile.is_open())
        {
            throw std::runtime_error("重写 index.json 失败");
        }

        indexFile << std::fixed << std::setprecision(6);
        indexFile << "{\n"
                  << "  \"schema_version\": " << JsonString(kSchemaVersion) << ",\n"
                  << "  \"session_id\": " << JsonString(sessionId_) << ",\n"
                  << "  \"source_type\": " << JsonString(kRealSourceType) << ",\n"
                  << "  \"episodes\": [\n";
        for (size_t index = 0; index < summaries.size(); ++index)
        {
            const auto& summary = summaries[index];
            indexFile << "    {\n"
                      << "      \"episode_id\": " << JsonString(summary.episodeId) << ",\n"
                      << "      \"instruction\": " << JsonString(summary.instruction) << ",\n"
                      << "      \"capture_mode\": " << JsonString(summary.captureMode) << ",\n"
                      << "      \"task_family\": " << JsonString(summary.taskFamily) << ",\n"
                      << "      \"target_type\": " << JsonString(summary.targetType) << ",\n"
                      << "      \"target_label\": " << JsonString(summary.targetLabel) << ",\n"
                      << "      \"target_description\": " << JsonString(summary.targetDescription) << ",\n"
                      << "      \"collector_notes\": " << JsonString(summary.collectorNotes) << ",\n"
                      << "      \"instruction_source\": " << JsonString(summary.instructionSource) << ",\n"
                      << "      \"segment_status\": " << JsonString(summary.segmentStatus) << ",\n"
                      << "      \"success\": " << JsonString(summary.success) << ",\n"
                      << "      \"termination_reason\": " << JsonString(summary.terminationReason) << ",\n"
                      << "      \"scene_id\": " << JsonString(summary.sceneId) << ",\n"
                      << "      \"operator_id\": " << JsonString(summary.operatorId) << ",\n"
                      << "      \"source_type\": " << JsonString(kRealSourceType) << ",\n"
                      << "      \"num_frames\": " << summary.numFrames << ",\n"
                      << "      \"start_timestamp\": " << summary.startTimestamp << ",\n"
                      << "      \"end_timestamp\": " << summary.endTimestamp << "\n"
                      << "    }";
            if (index + 1 != summaries.size())
            {
                indexFile << ",";
            }
            indexFile << "\n";
        }
        indexFile << "  ]\n"
                  << "}\n";
    }

    std::string sessionId_;
    fs::path outputDir_;
    fs::path episodesDir_;
    fs::path imagesDir_;
    fs::path indexPath_;
    mutable std::mutex mutex_;
    std::vector<EpisodeFrame> pendingFrames_;
    std::vector<EpisodeFrame> preRollFrames_;
    std::map<std::string, EpisodeSummary> episodeHistory_;
    int nextEpisodeIndex_ = 1;
    bool capturing_ = false;
    bool pendingLabel_ = false;
    bool effectiveMotionStarted_ = false;
    bool stopReady_ = false;
    bool stopRequested_ = false;
    size_t startCandidateCount_ = 0;
    size_t stopCandidateCount_ = 0;
    std::optional<TrajectoryMotionGateConfig> trajectoryGateConfig_;
    std::string pendingSceneId_;
    std::string pendingOperatorId_;
    TaskMetadata pendingTaskMetadata_;
    std::string pendingEpisodeId_;
    std::string lastRecordedEpisodeId_;
};

class ManagedShellProcess
{
public:
    bool Start(const std::string& command, const fs::path& logPath, std::string& error)
    {
        if (pid_ > 0)
        {
            error = "process already running";
            return false;
        }

        std::error_code ec;
        fs::create_directories(logPath.parent_path(), ec);
        const int fd = ::open(logPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0)
        {
            error = "打开日志文件失败: " + logPath.string();
            return false;
        }

        const pid_t child = ::fork();
        if (child < 0)
        {
            ::close(fd);
            error = "fork 失败";
            return false;
        }
        if (child == 0)
        {
            ::setsid();
            ::dup2(fd, STDOUT_FILENO);
            ::dup2(fd, STDERR_FILENO);
            ::close(fd);
            ::execl("/bin/bash", "bash", "-lc", command.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }

        ::close(fd);
        pid_ = child;
        return true;
    }

    void Stop(int signalValue, std::chrono::milliseconds timeout)
    {
        if (pid_ <= 0)
        {
            return;
        }

        ::kill(-pid_, signalValue);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            int status = 0;
            const pid_t result = ::waitpid(pid_, &status, WNOHANG);
            if (result == pid_)
            {
                pid_ = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        ::kill(-pid_, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        int status = 0;
        if (::waitpid(pid_, &status, WNOHANG) == pid_)
        {
            pid_ = -1;
            return;
        }

        ::kill(-pid_, SIGKILL);
        ::waitpid(pid_, &status, 0);
        pid_ = -1;
    }

    [[nodiscard]] bool Running() const
    {
        return pid_ > 0;
    }

private:
    pid_t pid_ = -1;
};

struct D435iStatus
{
    bool enabled = false;
    bool active = false;
    bool degraded = false;
    std::string degradedReason;
    bool colorValid = false;
    bool colorFresh = false;
    bool depthValid = false;
    bool depthFresh = false;
    uint64_t colorSeq = 0;
    uint64_t depthSeq = 0;
    double colorAgeSeconds = -1.0;
    double depthAgeSeconds = -1.0;
};

class D435iCaptureManager
{
public:
    explicit D435iCaptureManager(const Config& config)
        : config_(config)
    {
        requestedTopics_ = {
            "/camera/color/image_raw",
            DepthImageTopic(),
            "/camera/color/camera_info",
            DepthCameraInfoTopic(),
            "/tf",
            "/tf_static",
        };
        colorPreviewEnabled_ = true;
        depthPreviewEnabled_ = false;
    }

    void Start(const fs::path& sessionDir)
    {
        if (!config_.d435iEnabled || started_)
        {
            return;
        }

        started_ = true;
        sessionDir_ = sessionDir;
        artifactDir_ = sessionDir_ / "d435i";
        bagPath_ = artifactDir_ / (config_.d435iBagBaseName + ".bag");
        launchLogPath_ = artifactDir_ / "rs_camera.launch.log";
        rosbagLogPath_ = artifactDir_ / "rosbag_record.log";
        sidecarLogPath_ = artifactDir_ / "preview_sidecar.log";
        statsPath_ = artifactDir_ / "d435i_bag_stats.json";
        sidecarPort_ = FindFreeLoopbackPort();
        if (sidecarPort_ <= 0)
        {
            degraded_ = true;
            degradedReason_ = "sidecar_port_unavailable";
            return;
        }

        std::error_code ec;
        fs::create_directories(artifactDir_, ec);
        if (ec)
        {
            throw std::runtime_error("创建 D435i 目录失败: " + artifactDir_.string());
        }

        if (config_.d435iAutostartNode)
        {
            std::string launchError;
            if (!cameraLaunch_.Start(BuildLaunchCommand(), launchLogPath_, launchError))
            {
                degraded_ = true;
                degradedReason_ = "launch_failed: " + launchError;
                return;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(std::lround(config_.d435iNodeWarmupSeconds * 1000.0))));
        }

        ProbeResult probe = WaitForTopicsReady();
        resolvedTopics_ = probe.resolvedTopics;
        missingTopics_ = probe.missingTopics;
        if (!probe.ready)
        {
            degraded_ = true;
            degradedReason_ = config_.d435iAutostartNode ? "topic_probe_failed" : "external_topics_unavailable";
            return;
        }

        std::string rosbagError;
        if (!rosbagRecord_.Start(BuildRosbagCommand(), rosbagLogPath_, rosbagError))
        {
            degraded_ = true;
            degradedReason_ = "rosbag_failed: " + rosbagError;
            cameraLaunch_.Stop(SIGINT, std::chrono::seconds(3));
            return;
        }

        std::string sidecarError;
        if (!previewSidecar_.Start(BuildSidecarCommand(), sidecarLogPath_, sidecarError))
        {
            degraded_ = true;
            degradedReason_ = "preview_sidecar_failed: " + sidecarError;
            rosbagRecord_.Stop(SIGINT, std::chrono::seconds(5));
            cameraLaunch_.Stop(SIGINT, std::chrono::seconds(3));
            return;
        }

        if (!WaitForSidecarReady())
        {
            degraded_ = true;
            degradedReason_ = "preview_first_frame_timeout";
            previewSidecar_.Stop(SIGINT, std::chrono::seconds(3));
            rosbagRecord_.Stop(SIGINT, std::chrono::seconds(5));
            cameraLaunch_.Stop(SIGINT, std::chrono::seconds(3));
            return;
        }

        startTimestamp_ = NowSeconds();
        active_ = true;
        pollingRunning_.store(true);
        pollingThread_ = std::thread(&D435iCaptureManager::PollingLoop, this);
    }

    void Stop()
    {
        if (!started_)
        {
            return;
        }

        pollingRunning_.store(false);
        frameUpdatedCv_.notify_all();
        if (pollingThread_.joinable())
        {
            pollingThread_.join();
        }
        rosbagRecord_.Stop(SIGINT, std::chrono::seconds(5));
        previewSidecar_.Stop(SIGINT, std::chrono::seconds(3));
        cameraLaunch_.Stop(SIGINT, std::chrono::seconds(3));
        endTimestamp_ = NowSeconds();
        AnalyzeBag();
        WriteMetadata();
        active_ = false;
        std::lock_guard<std::mutex> lock(statusMutex_);
        cachedStatusTimestamp_ = 0.0;
        cachedStatus_ = D435iStatus{};
    }

    [[nodiscard]] std::vector<uint8_t> FetchColorJpeg() const
    {
        std::lock_guard<std::mutex> lock(frameMutex_);
        return colorFrame_.jpegBytes;
    }

    [[nodiscard]] std::vector<uint8_t> FetchDepthJpeg() const
    {
        std::lock_guard<std::mutex> lock(frameMutex_);
        return depthFrame_.jpegBytes;
    }

    UiImageFrame WaitForNextColorFrame(uint64_t lastSequence, int timeoutMs) const
    {
        return WaitForNextFrame(lastSequence, timeoutMs, true);
    }

    UiImageFrame WaitForNextDepthFrame(uint64_t lastSequence, int timeoutMs) const
    {
        return WaitForNextFrame(lastSequence, timeoutMs, false);
    }

    [[nodiscard]] D435iStatus StatusSnapshot() const
    {
        const double nowSeconds = NowSeconds();
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            if (cachedStatusTimestamp_ > 0.0 &&
                (!active_ || pollingRunning_.load() || nowSeconds - cachedStatusTimestamp_ <= 0.5))
            {
                return cachedStatus_;
            }
        }

        D435iStatus status;
        status.enabled = config_.d435iEnabled;
        status.active = active_;
        status.degraded = degraded_;
        status.degradedReason = degradedReason_;
        if (!active_)
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            cachedStatus_ = status;
            cachedStatusTimestamp_ = nowSeconds;
            return status;
        }

        const auto body = HttpGetLoopbackText(sidecarPort_, "/status.json", 500);
        if (!body.has_value())
        {
            status.degraded = true;
            if (status.degradedReason.empty())
            {
                status.degradedReason = "preview_unreachable";
            }
            std::lock_guard<std::mutex> lock(statusMutex_);
            cachedStatus_ = status;
            cachedStatusTimestamp_ = nowSeconds;
            return status;
        }

        status.colorValid = ExtractJsonBoolFieldLocal(body.value(), "color_valid").value_or(false);
        status.colorFresh = ExtractJsonBoolFieldLocal(body.value(), "color_fresh").value_or(false);
        status.depthValid = ExtractJsonBoolFieldLocal(body.value(), "depth_valid").value_or(false);
        status.depthFresh = ExtractJsonBoolFieldLocal(body.value(), "depth_fresh").value_or(false);
        status.colorSeq = static_cast<uint64_t>(std::max(0.0, ExtractJsonNumberFieldLocal(body.value(), "color_seq").value_or(0.0)));
        status.depthSeq = static_cast<uint64_t>(std::max(0.0, ExtractJsonNumberFieldLocal(body.value(), "depth_seq").value_or(0.0)));
        status.colorAgeSeconds = ExtractJsonNumberFieldLocal(body.value(), "color_age_s").value_or(-1.0);
        status.depthAgeSeconds = ExtractJsonNumberFieldLocal(body.value(), "depth_age_s").value_or(-1.0);
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            cachedStatus_ = status;
            cachedStatusTimestamp_ = nowSeconds;
        }
        return status;
    }

    [[nodiscard]] fs::path RelativeBagPath() const
    {
        if (sessionDir_.empty() || bagPath_.empty())
        {
            return {};
        }
        std::error_code ec;
        const fs::path rel = fs::relative(bagPath_, sessionDir_, ec);
        return ec ? bagPath_.filename() : rel;
    }

private:
    struct ProbeResult
    {
        bool ready = false;
        std::vector<std::string> resolvedTopics;
        std::vector<std::string> missingTopics;
    };

    [[nodiscard]] std::string DepthImageTopic() const
    {
        return config_.d435iAlignDepth ? "/camera/aligned_depth_to_color/image_raw" : "/camera/depth/image_rect_raw";
    }

    [[nodiscard]] std::string DepthCameraInfoTopic() const
    {
        return config_.d435iAlignDepth ? "/camera/aligned_depth_to_color/camera_info" : "/camera/depth/camera_info";
    }

    [[nodiscard]] std::string DepthTopicSemantics() const
    {
        return config_.d435iAlignDepth ? "aligned_depth_to_color" : "raw_unaligned_depth";
    }

    [[nodiscard]] std::string NativeScriptPath(const std::string& name) const
    {
        return (config_.collectorRoot / "native" / name).string();
    }

    [[nodiscard]] std::string BuildLaunchCommand() const
    {
        const std::vector<std::string> launchTokens = SplitWhitespaceTokens(config_.d435iLaunchFile);
        std::ostringstream oss;
        oss << "source " << ShellQuote(config_.d435iRosSetupPath) << " && exec roslaunch";
        for (const auto& token : launchTokens)
        {
            oss << ' ' << ShellQuote(token);
        }
        oss << ' ' << ShellQuote("enable_color:=true")
            << ' ' << ShellQuote("enable_depth:=true")
            << ' ' << ShellQuote("color_width:=" + std::to_string(ProfileWidth(config_.d435iColorProfile)))
            << ' ' << ShellQuote("color_height:=" + std::to_string(ProfileHeight(config_.d435iColorProfile)))
            << ' ' << ShellQuote("color_fps:=" + std::to_string(ProfileFps(config_.d435iColorProfile)))
            << ' ' << ShellQuote("depth_width:=" + std::to_string(ProfileWidth(config_.d435iDepthProfile)))
            << ' ' << ShellQuote("depth_height:=" + std::to_string(ProfileHeight(config_.d435iDepthProfile)))
            << ' ' << ShellQuote("depth_fps:=" + std::to_string(ProfileFps(config_.d435iDepthProfile)))
            << ' ' << ShellQuote(std::string("align_depth:=") + (config_.d435iAlignDepth ? "true" : "false"));
        if (!config_.d435iSerial.empty())
        {
            oss << ' ' << ShellQuote("serial_no:=" + config_.d435iSerial);
        }
        return oss.str();
    }

    [[nodiscard]] std::string BuildProbeCommand() const
    {
        std::ostringstream oss;
        oss << "source " << ShellQuote(config_.d435iRosSetupPath)
            << " && python3 " << ShellQuote(NativeScriptPath("d435i_preview_sidecar.py"))
            << " probe"
            << ' ' << ShellQuote("--color-topic")
            << ' ' << ShellQuote("/camera/color/image_raw")
            << ' ' << ShellQuote("--depth-topic")
            << ' ' << ShellQuote(DepthImageTopic())
            << ' ' << ShellQuote("--color-camera-info-topic")
            << ' ' << ShellQuote("/camera/color/camera_info")
            << ' ' << ShellQuote("--depth-camera-info-topic")
            << ' ' << ShellQuote(DepthCameraInfoTopic())
            << ' ' << ShellQuote("--timeout-s")
            << ' ' << ShellQuote("6.0")
            << ' ' << ShellQuote("--min-depth-m")
            << ' ' << ShellQuote(std::to_string(config_.d435iDepthPreviewMinMeters))
            << ' ' << ShellQuote("--max-depth-m")
            << ' ' << ShellQuote(std::to_string(config_.d435iDepthPreviewMaxMeters));
        if (!colorPreviewEnabled_)
        {
            oss << ' ' << ShellQuote("--disable-color-preview");
        }
        if (!depthPreviewEnabled_)
        {
            oss << ' ' << ShellQuote("--disable-depth-preview");
        }
        return oss.str();
    }

    [[nodiscard]] std::string BuildRosbagCommand() const
    {
        std::ostringstream oss;
        oss << "source " << ShellQuote(config_.d435iRosSetupPath)
            << " && cd " << ShellQuote(artifactDir_.string())
            << " && exec rosbag record -O " << ShellQuote(config_.d435iBagBaseName);
        for (const auto& topic : resolvedTopics_)
        {
            oss << ' ' << ShellQuote(topic);
        }
        return oss.str();
    }

    [[nodiscard]] std::string BuildSidecarCommand() const
    {
        std::ostringstream oss;
        oss << "source " << ShellQuote(config_.d435iRosSetupPath)
            << " && exec python3 " << ShellQuote(NativeScriptPath("d435i_preview_sidecar.py"))
            << " serve"
            << ' ' << ShellQuote("--port")
            << ' ' << ShellQuote(std::to_string(sidecarPort_))
            << ' ' << ShellQuote("--color-topic")
            << ' ' << ShellQuote("/camera/color/image_raw")
            << ' ' << ShellQuote("--depth-topic")
            << ' ' << ShellQuote(DepthImageTopic())
            << ' ' << ShellQuote("--min-depth-m")
            << ' ' << ShellQuote(std::to_string(config_.d435iDepthPreviewMinMeters))
            << ' ' << ShellQuote("--max-depth-m")
            << ' ' << ShellQuote(std::to_string(config_.d435iDepthPreviewMaxMeters));
        if (!colorPreviewEnabled_)
        {
            oss << ' ' << ShellQuote("--disable-color-preview");
        }
        if (!depthPreviewEnabled_)
        {
            oss << ' ' << ShellQuote("--disable-depth-preview");
        }
        return oss.str();
    }

    ProbeResult ProbeTopicsOnce()
    {
        ProbeResult result;
        std::string output;
        int exitCode = -1;
        if (!RunShellCommandCapture(BuildProbeCommand(), output, exitCode))
        {
            return result;
        }

        result.ready = ExtractJsonBoolFieldLocal(output, "ready").value_or(false) && exitCode == 0;

        static const std::vector<std::string> fallback = {
            "/camera/color/image_raw",
            DepthImageTopic(),
            "/camera/color/camera_info",
            DepthCameraInfoTopic(),
            "/tf",
            "/tf_static",
        };
        result.resolvedTopics = ExtractJsonStringArrayFieldLocal(output, "resolved_topics").value_or(fallback);
        result.missingTopics = ExtractJsonStringArrayFieldLocal(output, "missing_topics").value_or(std::vector<std::string>{});
        if (result.missingTopics.empty() && !result.ready)
        {
            result.missingTopics = {
                "/camera/color/image_raw",
                DepthImageTopic(),
            };
        }
        return result;
    }

    ProbeResult WaitForTopicsReady()
    {
        ProbeResult lastProbe;
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(static_cast<int>(std::lround(config_.d435iTopicReadyTimeoutSeconds * 1000.0)));
        while (std::chrono::steady_clock::now() < deadline)
        {
            lastProbe = ProbeTopicsOnce();
            if (lastProbe.ready)
            {
                return lastProbe;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        return lastProbe;
    }

    bool WaitForSidecarReady() const
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (std::chrono::steady_clock::now() < deadline)
        {
            const auto body = HttpGetLoopbackText(sidecarPort_, "/status.json", 300);
            if (body.has_value() &&
                ExtractJsonBoolFieldLocal(body.value(), "color_valid").value_or(false) &&
                ExtractJsonBoolFieldLocal(body.value(), "depth_valid").value_or(false))
            {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
        return false;
    }

    void PollingLoop()
    {
        while (pollingRunning_.load())
        {
            const double nowSeconds = NowSeconds();
            D435iStatus status;
            status.enabled = config_.d435iEnabled;
            status.active = active_;
            status.degraded = degraded_;
            status.degradedReason = degradedReason_;

            const auto body = HttpGetLoopbackText(sidecarPort_, "/status.json", 250);
            if (!body.has_value())
            {
                status.degraded = true;
                if (status.degradedReason.empty())
                {
                    status.degradedReason = "preview_unreachable";
                }
            }
            else
            {
                status.colorValid = ExtractJsonBoolFieldLocal(body.value(), "color_valid").value_or(false);
                status.colorFresh = ExtractJsonBoolFieldLocal(body.value(), "color_fresh").value_or(false);
                status.depthValid = ExtractJsonBoolFieldLocal(body.value(), "depth_valid").value_or(false);
                status.depthFresh = ExtractJsonBoolFieldLocal(body.value(), "depth_fresh").value_or(false);
                status.colorSeq = static_cast<uint64_t>(std::max(0.0, ExtractJsonNumberFieldLocal(body.value(), "color_seq").value_or(0.0)));
                status.depthSeq = static_cast<uint64_t>(std::max(0.0, ExtractJsonNumberFieldLocal(body.value(), "depth_seq").value_or(0.0)));
                status.colorAgeSeconds = ExtractJsonNumberFieldLocal(body.value(), "color_age_s").value_or(-1.0);
                status.depthAgeSeconds = ExtractJsonNumberFieldLocal(body.value(), "depth_age_s").value_or(-1.0);
            }

            UpdateFrameCache(status, nowSeconds);
            {
                std::lock_guard<std::mutex> lock(statusMutex_);
                cachedStatus_ = status;
                cachedStatusTimestamp_ = nowSeconds;
            }

            for (int i = 0; i < 8 && pollingRunning_.load(); ++i)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        }
    }

    void UpdateFrameCache(const D435iStatus& status, double nowSeconds)
    {
        std::vector<std::uint8_t> colorJpeg;
        std::vector<std::uint8_t> depthJpeg;
        bool colorChanged = false;
        bool depthChanged = false;

        {
            std::lock_guard<std::mutex> lock(frameMutex_);
            if (status.colorValid && status.colorSeq > colorFrame_.sequence)
            {
                colorChanged = true;
            }
            if (depthPreviewEnabled_ && status.depthValid && status.depthSeq > depthFrame_.sequence)
            {
                depthChanged = true;
            }
        }

        if (colorChanged)
        {
            colorJpeg = FetchJpeg("/latest/color.jpg");
        }
        if (depthChanged)
        {
            depthJpeg = FetchJpeg("/latest/depth.jpg");
        }

        bool notify = false;
        {
            std::lock_guard<std::mutex> lock(frameMutex_);
            if (colorChanged && !colorJpeg.empty())
            {
                colorFrame_.sequence = status.colorSeq;
                colorFrame_.timestamp = nowSeconds;
                colorFrame_.jpegBytes = std::move(colorJpeg);
                colorFrame_.valid = true;
                notify = true;
            }
            if (depthChanged && !depthJpeg.empty())
            {
                depthFrame_.sequence = status.depthSeq;
                depthFrame_.timestamp = nowSeconds;
                depthFrame_.jpegBytes = std::move(depthJpeg);
                depthFrame_.valid = true;
                notify = true;
            }
        }
        if (notify)
        {
            frameUpdatedCv_.notify_all();
        }
    }

    void AnalyzeBag()
    {
        if (!fs::exists(bagPath_))
        {
            return;
        }

        std::string output;
        int exitCode = -1;
        std::ostringstream oss;
        oss << "source " << ShellQuote(config_.d435iRosSetupPath)
            << " && python3 " << ShellQuote(NativeScriptPath("d435i_bag_analyze.py"))
            << ' ' << ShellQuote("--bag")
            << ' ' << ShellQuote(bagPath_.string())
            << ' ' << ShellQuote("--color-topic")
            << ' ' << ShellQuote("/camera/color/image_raw")
            << ' ' << ShellQuote("--depth-topic")
            << ' ' << ShellQuote(DepthImageTopic());
        if (RunShellCommandCapture(oss.str(), output, exitCode))
        {
            bagStatsJson_ = output;
            bagStatsExitCode_ = exitCode;
            std::ofstream statsOut(statsPath_, std::ios::out | std::ios::trunc);
            if (statsOut.is_open())
            {
                statsOut << output;
            }
        }

        std::string infoOutput;
        int infoExit = -1;
        const std::string infoCommand =
            "source " + ShellQuote(config_.d435iRosSetupPath) +
            " && rosbag info --yaml " + ShellQuote(bagPath_.string());
        if (RunShellCommandCapture(infoCommand, infoOutput, infoExit))
        {
            bagInfoOk_ = infoExit == 0;
        }
    }

    void WriteMetadata() const
    {
        const fs::path metadataPath = artifactDir_ / "d435i_capture.json";
        std::ofstream output(metadataPath, std::ios::out | std::ios::trunc);
        if (!output.is_open())
        {
            return;
        }

        output << std::fixed << std::setprecision(6)
               << "{\n"
               << "  \"enabled\": true,\n"
               << "  \"autostart_node\": " << (config_.d435iAutostartNode ? "true" : "false") << ",\n"
               << "  \"degraded\": " << (degraded_ ? "true" : "false") << ",\n"
               << "  \"degraded_reason\": " << JsonString(degradedReason_) << ",\n"
               << "  \"ros_setup_path\": " << JsonString(config_.d435iRosSetupPath) << ",\n"
               << "  \"bag_path\": " << JsonString(RelativeBagPath().string()) << ",\n"
               << "  \"launch_log\": " << JsonString(launchLogPath_.filename().string()) << ",\n"
               << "  \"rosbag_log\": " << JsonString(rosbagLogPath_.filename().string()) << ",\n"
               << "  \"sidecar_log\": " << JsonString(sidecarLogPath_.filename().string()) << ",\n"
               << "  \"start_timestamp\": " << startTimestamp_ << ",\n"
               << "  \"end_timestamp\": " << endTimestamp_ << ",\n"
               << "  \"align_depth_applied\": " << (config_.d435iAlignDepth ? "true" : "false") << ",\n"
               << "  \"depth_topic_semantics\": " << JsonString(DepthTopicSemantics()) << ",\n"
               << "  \"requested_topics\": " << JsonStringArray(requestedTopics_) << ",\n"
               << "  \"resolved_topics\": " << JsonStringArray(resolvedTopics_) << ",\n"
               << "  \"missing_topics\": " << JsonStringArray(missingTopics_) << ",\n"
               << "  \"bag_close_status\": " << JsonString(bagInfoOk_ ? "ok" : "invalid") << ",\n"
               << "  \"bag_stats_exit_code\": " << bagStatsExitCode_;
        if (!bagStatsJson_.empty())
        {
            output << ",\n  \"bag_stats\": " << bagStatsJson_;
        }
        output << "\n}\n";
    }

    [[nodiscard]] std::vector<uint8_t> FetchJpeg(const std::string& path) const
    {
        if (!active_)
        {
            return {};
        }
        const auto body = HttpGetLoopbackBinary(sidecarPort_, path, 500);
        return body.value_or(std::vector<uint8_t>{});
    }

    UiImageFrame WaitForNextFrame(uint64_t lastSequence, int timeoutMs, bool color) const
    {
        UiImageFrame frame;
        std::unique_lock<std::mutex> lock(frameMutex_);
        const auto hasNewFrame = [&]()
        {
            const UiImageFrame& source = color ? colorFrame_ : depthFrame_;
            return !pollingRunning_.load() ||
                   (source.valid && !source.jpegBytes.empty() && source.sequence > lastSequence);
        };

        if (!hasNewFrame())
        {
            frameUpdatedCv_.wait_for(lock, std::chrono::milliseconds(std::max(timeoutMs, 1)), hasNewFrame);
        }

        const UiImageFrame& source = color ? colorFrame_ : depthFrame_;
        if (!source.valid || source.jpegBytes.empty() || source.sequence <= lastSequence)
        {
            return frame;
        }
        frame = source;
        return frame;
    }

    static int ProfileWidth(const std::string& profile)
    {
        std::stringstream ss(profile);
        std::string item;
        std::getline(ss, item, 'x');
        return std::max(1, std::stoi(item));
    }

    static int ProfileHeight(const std::string& profile)
    {
        std::stringstream ss(profile);
        std::string item;
        std::getline(ss, item, 'x');
        std::getline(ss, item, 'x');
        return std::max(1, std::stoi(item));
    }

    static int ProfileFps(const std::string& profile)
    {
        std::stringstream ss(profile);
        std::string item;
        std::getline(ss, item, 'x');
        std::getline(ss, item, 'x');
        std::getline(ss, item, 'x');
        return std::max(1, std::stoi(item));
    }

    const Config& config_;
    bool started_ = false;
    bool active_ = false;
    bool degraded_ = false;
    bool bagInfoOk_ = false;
    int bagStatsExitCode_ = -1;
    int sidecarPort_ = -1;
    std::string degradedReason_;
    std::string bagStatsJson_;
    double startTimestamp_ = 0.0;
    double endTimestamp_ = 0.0;
    fs::path sessionDir_;
    fs::path artifactDir_;
    fs::path bagPath_;
    fs::path launchLogPath_;
    fs::path rosbagLogPath_;
    fs::path sidecarLogPath_;
    fs::path statsPath_;
    std::vector<std::string> requestedTopics_;
    std::vector<std::string> resolvedTopics_;
    std::vector<std::string> missingTopics_;
    ManagedShellProcess cameraLaunch_;
    ManagedShellProcess rosbagRecord_;
    ManagedShellProcess previewSidecar_;
    bool colorPreviewEnabled_ = true;
    bool depthPreviewEnabled_ = false;
    std::atomic<bool> pollingRunning_{false};
    std::thread pollingThread_;
    mutable std::mutex statusMutex_;
    mutable D435iStatus cachedStatus_;
    mutable double cachedStatusTimestamp_ = 0.0;
    mutable std::mutex frameMutex_;
    mutable std::condition_variable frameUpdatedCv_;
    UiImageFrame colorFrame_;
    UiImageFrame depthFrame_;
};

class CollectorApp
{
public:
    explicit CollectorApp(const Config& config)
        : config_(config),
          logger_(config.outputDir),
          trajectoryGateConfig_(BuildTrajectoryMotionGateConfig(config.videoPollHz))
    {
        editableConfig_.sceneId = config.sceneId;
        editableConfig_.operatorId = config.operatorId;
        editableConfig_.instruction = config.instruction;
        editableConfig_.instructionSource = config.instructionSource;
        editableConfig_.taskFamily = config.taskFamily;
        editableConfig_.targetType = config.targetType;
        editableConfig_.targetDescription = config.targetDescription;
        editableConfig_.collectorNotes = config.collectorNotes;
        editableConfig_.cmdVxMax = config.cmdVxMax;
        editableConfig_.cmdVyMax = config.cmdVyMax;
        editableConfig_.cmdWzMax = config.cmdWzMax;
        startupGateActive_ = false;
        if (config_.d435iEnabled)
        {
            d435iCaptureManager_ = std::make_unique<D435iCaptureManager>(config_);
        }
    }

    ~CollectorApp()
    {
        Shutdown();
    }

    bool ShouldQuit() const
    {
        return quitRequested_.load();
    }

    void RequestQuit()
    {
        quitRequested_.store(true);
    }

    EditableCollectorConfig GetEditableConfigSnapshot() const
    {
        std::lock_guard<std::mutex> lock(editableConfigMutex_);
        return editableConfig_;
    }

    void MaybePrintStartupGateReminder() {}

    void PrintStartupInstructions() const
    {
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2KCollector Startup" << std::endl;
        std::cout << "  采集前检查：确认周围安全、机器人姿态稳定、状态已正常更新" << std::endl;
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            std::cout << "  主控手柄：原生手柄直通已关闭；当前仅保留 collector 映射的离散控制" << std::endl;
            std::cout << "  离散控制：D-pad 对应 W/S/A/D，X/B 对应 Q/E；进入 label_ready 后 D-pad 临时切换为评分 1/2/3/4" << std::endl;
            std::cout << "  录制按键：Start 开始录制  A 结束录制  Y 丢弃当前段，可立即开始录制" << std::endl;
            std::cout << "  安全/恢复：R2 急停  F1 清 fault  F2 切换站立" << std::endl;
        }
        else
        {
            std::cout << "  主控键位：W/S 前后  A/D 横移  Q/E 转向  R 开始录制  T 结束录制  ESC 丢弃当前段" << std::endl;
            std::cout << "  其他键位：C 清 fault  V 切换站立  P 状态  H 帮助" << std::endl;
            std::cout << "  标注快捷键：待标注时按 1/2/3/4 直接完成评分" << std::endl;
            std::cout << "  安全键位：Space 急停（始终有效）" << std::endl;
        }
        std::cout << "  运行参数：启动时固定；如需修改 scene/instruction/control，请重启 collector" << std::endl;
        std::cout << "  录制门控：" << StartupUnlockHint(config_) << std::endl;
        std::cout << "  采集流程：trajectory input_backend=" << InputBackendName(config_.inputBackend) << std::endl;
        if (config_.d435iEnabled)
        {
            std::cout << "  D435i：session 级 rosbag 录制已启用";
            if (config_.d435iAutostartNode)
            {
                std::cout << "（collector 会自动启动 " << config_.d435iLaunchFile << "）";
            }
            std::cout << std::endl;
        }
        if (config_.previewUi)
        {
            std::cout << "  当前运行在离线预览模式：未连接 Go2，页面使用本地示例状态与图像" << std::endl;
        }
        else
        {
            std::cout << "  " << StartupPromptText(config_) << std::endl;
        }
    }

    void ResetTrajectoryStopFlowLocked()
    {
        trajectoryStopRequested_ = false;
        trajectoryStopWaitingForRelease_ = false;
        trajectoryStopFinalizeDeadline_ = std::chrono::steady_clock::time_point{};
        trajectoryStopForceFinalizeDeadline_ = std::chrono::steady_clock::time_point{};
    }

    enum class TrajectoryStopFlowAction
    {
        None,
        StartGracePeriod,
        StartGracePeriodAfterTimeout,
        Finalize,
    };

    static bool HasSteadyDeadline(const std::chrono::steady_clock::time_point& deadline)
    {
        return deadline.time_since_epoch().count() != 0;
    }

    static const char* TrajectoryStopStatusMessage(bool waitingForRelease)
    {
        return waitingForRelease ? "Stopping segment... waiting for input release"
                                 : "Finalizing segment... submit a score";
    }

    void ResetCaptureProgressLocked()
    {
        ResetTrajectoryStopFlowLocked();
    }

    void EnterIdleOrFaultStateLocked()
    {
        captureState_ = safetyState_ == SafetyState::SafeReady ? CaptureState::Idle : CaptureState::Fault;
        ResetCaptureProgressLocked();
    }

    void BeginTrajectoryStopLocked(bool motionInputActive)
    {
        trajectoryStopRequested_ = true;
        trajectoryStopWaitingForRelease_ = motionInputActive;
        trajectoryStopFinalizeDeadline_ =
            motionInputActive ? std::chrono::steady_clock::time_point{}
                              : std::chrono::steady_clock::now() + kTrajectoryFinalizeGracePeriod;
        trajectoryStopForceFinalizeDeadline_ = std::chrono::steady_clock::now() + kTrajectoryStopReleaseTimeout;
    }

    void StartTrajectoryFinalizeGracePeriod(const std::string& reason, bool emitLog)
    {
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing || !trajectoryStopRequested_)
            {
                return;
            }
            trajectoryStopWaitingForRelease_ = false;
            if (trajectoryStopFinalizeDeadline_.time_since_epoch().count() == 0)
            {
                trajectoryStopFinalizeDeadline_ = std::chrono::steady_clock::now() + kTrajectoryFinalizeGracePeriod;
            }
        }
        if (emitLog)
        {
            PrintLine(reason);
        }
    }

    void Start()
    {
        if (lifecycleStarted_.exchange(true))
        {
            return;
        }
        shutdownStarted_.store(false);
        quitRequested_.store(false);

        terminalRawEnabled_ = terminalGuard_.TryEnable();
        inputBackend_ = collector::input::CreateInputBackend(BuildInputBackendConfig());
        inputBackend_->Start();

        if (!config_.previewUi)
        {
            unitree::robot::ChannelFactory::Instance()->Init(0, config_.networkInterface);

            sportClient_ = std::make_unique<unitree::robot::go2::SportClient>();
            sportClient_->SetTimeout(10.0f);
            sportClient_->Init();
            const bool passthroughConfigured =
                SetWirelessControllerPassthroughEnabled(false, "collector startup disables native joystick");
            if (config_.inputBackend == Config::InputBackend::WirelessController && !passthroughConfigured)
            {
                throw std::runtime_error("无线手柄模式启动失败：未能关闭原生手柄直通，已拒绝继续运行以避免控制冲突");
            }

            sportStateSubscriber_ =
                std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>>(kSportStateTopic);
            sportStateSubscriber_->InitChannel(std::bind(&CollectorApp::OnSportState, this, std::placeholders::_1), 1);

            if (config_.inputBackend == Config::InputBackend::WirelessController)
            {
                wirelessControllerSubscriber_ =
                    std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>>(kWirelessControllerTopic);
                wirelessControllerSubscriber_->InitChannel(std::bind(&CollectorApp::OnWirelessController, this, std::placeholders::_1), 1);
            }

            videoClient_ = std::make_unique<unitree::robot::go2::VideoClient>();
            videoClient_->SetTimeout(1.0f);
            videoClient_->Init();
        }
        else
        {
            InitializePreviewMode();
        }

        if (d435iCaptureManager_)
        {
            d435iCaptureManager_->Start(logger_.OutputDir());
        }

        running_.store(true);
        controlThread_ = std::thread(&CollectorApp::ControlLoop, this);
        loggingThread_ = std::thread(&CollectorApp::LoggingLoop, this);
        keyboardThread_ = std::thread(&CollectorApp::KeyboardLoop, this);
        videoThread_ = std::thread(&CollectorApp::VideoLoop, this);

        if (config_.webUiEnabled)
        {
            WebUiServerConfig webConfig;
            webConfig.port = config_.webPort;
            webConfig.assetDir = (config_.collectorRoot / "native" / "web_ui_assets").string();
            webConfig.statusProvider = [this]() { return GetUiStatusSnapshot(); };
            webConfig.startHandler = [this]() { return RequestBeginSegment(); };
            webConfig.stopHandler = [this]() { return RequestStopSegmentForLabel(); };
            webConfig.discardHandler = [this]() { return RequestDiscardSegment("discarded by web ui"); };
            webConfig.estopHandler = [this]() { return RequestEmergencyStopFromUi(); };
            webConfig.clearFaultHandler = [this]() { return RequestClearFaultFromUi(); };
            webConfig.quitHandler = [this]() { return RequestQuitFromUi(); };
            webConfig.submitLabelHandler = [this](const SegmentLabelInput& input)
            {
                return SubmitPendingLabel(input);
            };
            webConfig.latestImageJpegProvider = [this]() { return GetLatestImageJpeg(); };
            webConfig.nextImageFrameProvider = [this](uint64_t lastSequence, int timeoutMs)
            {
                return WaitForNextImageFrame(lastSequence, timeoutMs);
            };
            webConfig.d435iColorJpegProvider = [this]()
            {
                return d435iCaptureManager_ ? d435iCaptureManager_->FetchColorJpeg() : std::vector<uint8_t>{};
            };
            webConfig.d435iDepthJpegProvider = [this]()
            {
                return d435iCaptureManager_ ? d435iCaptureManager_->FetchDepthJpeg() : std::vector<uint8_t>{};
            };
            webConfig.nextD435iColorFrameProvider = [this](uint64_t lastSequence, int timeoutMs)
            {
                return d435iCaptureManager_ ? d435iCaptureManager_->WaitForNextColorFrame(lastSequence, timeoutMs) : UiImageFrame{};
            };
            webConfig.nextD435iDepthFrameProvider = [this](uint64_t lastSequence, int timeoutMs)
            {
                return d435iCaptureManager_ ? d435iCaptureManager_->WaitForNextDepthFrame(lastSequence, timeoutMs) : UiImageFrame{};
            };
            webUiServer_ = std::make_unique<WebUiServer>(webConfig);
            webUiServer_->Start();
        }

        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_.valid = true;
            latestCommand_.timestamp = NowSeconds();
        }

        PrintLine("session 目录：" + logger_.OutputDir().string());
        if (d435iCaptureManager_)
        {
            PrintLine("D435i rosbag：" + (logger_.OutputDir() / "d435i" / (config_.d435iBagBaseName + ".bag")).string());
            const D435iStatus d435iStatus = d435iCaptureManager_->StatusSnapshot();
            if (d435iStatus.degraded)
            {
                PrintLine("D435i degraded：" + d435iStatus.degradedReason);
            }
        }
        PrintLine("输入后端：" + (inputBackend_ ? inputBackend_->BackendName() : InputBackendName(config_.inputBackend)) +
                  (config_.inputDevice.empty() ? "" : " device=" + config_.inputDevice.string()));
        if (config_.webUiEnabled)
        {
            PrintLine("Web UI：http://127.0.0.1:" + std::to_string(config_.webPort));
        }
        if (config_.previewUi)
        {
            PrintLine("preview ui 模式已启动；未连接 Go2，控制与状态均为本地预览数据");
        }
        else if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            PrintLine("collector 已就绪；原生手柄直通已禁用，当前由 collector 接管手柄映射");
            PrintLine("D-pad=WSAD，X/B=QE；待标注时 D-pad 切回 1/2/3/4 评分");
            PrintLine("当前可直接移动和录制");
        }
        else
        {
            PrintLine("collector 已就绪；当前输入后端可直接移动和录制");
        }
        PrintLine("运行参数在启动时固定；如需修改 scene/instruction/control，请重启 collector");
        PrintStartupInstructions();
    }

    void Shutdown()
    {
        if (shutdownStarted_.exchange(true))
        {
            return;
        }
        if (!lifecycleStarted_.exchange(false))
        {
            return;
        }

        running_.store(false);
        imageUpdatedCv_.notify_all();
        if (webUiServer_)
        {
            webUiServer_->Stop();
        }

        if (sportStateSubscriber_)
        {
            sportStateSubscriber_->CloseChannel();
            sportStateSubscriber_.reset();
        }
        if (wirelessControllerSubscriber_)
        {
            wirelessControllerSubscriber_->CloseChannel();
            wirelessControllerSubscriber_.reset();
        }

        if (keyboardThread_.joinable())
        {
            keyboardThread_.join();
        }
        if (controlThread_.joinable())
        {
            controlThread_.join();
        }
        if (loggingThread_.joinable())
        {
            loggingThread_.join();
        }
        if (videoThread_.joinable())
        {
            videoThread_.join();
        }

        if (d435iCaptureManager_)
        {
            d435iCaptureManager_->Stop();
        }

        StopMotion();
        if (inputBackend_)
        {
            inputBackend_->Stop();
            inputBackend_.reset();
        }
        if (!config_.previewUi)
        {
            SetWirelessControllerPassthroughEnabled(true, "collector shutdown restore native joystick");
            unitree::robot::ChannelFactory::Instance()->Release();
        }
        terminalGuard_.Disable();
        logger_.CleanupIfEmpty();
        webUiServer_.reset();
    }

    UiStatusSnapshot GetUiStatusSnapshot() const
    {
        const Snapshot latest = CollectLatestSnapshot();
        const auto loggerStatus = logger_.GetStatus();
        UiStatusSnapshot snapshot;
        CaptureState captureState;
        SafetyState safetyState;
        std::string faultReason;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            captureState = captureState_;
            safetyState = safetyState_;
            faultReason = latchedFaultReason_;
            snapshot.startupGateActive = startupGateActive_;
            snapshot.stopPhase = TrajectoryStopPhaseName(false, trajectoryStopRequested_, trajectoryStopWaitingForRelease_);
        }
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const double nowSeconds = NowSeconds();
        snapshot.running = running_.load();
        snapshot.webUiEnabled = config_.webUiEnabled;
        snapshot.webPort = config_.webPort;
        snapshot.previewMode = config_.previewUi;
        snapshot.startupPrompt =
            config_.previewUi ? "preview ui mode; no Go2 connection required" : StartupPromptText(config_);
        snapshot.sessionDir = logger_.OutputDir().string();
        snapshot.captureMode = kTrajectoryCaptureMode;
        snapshot.bufferedFrames = loggerStatus.bufferedFrames;
        snapshot.segmentDurationSeconds = loggerStatus.bufferedFrames >= 2
                                              ? std::max(0.0, loggerStatus.endTimestamp - loggerStatus.startTimestamp)
                                              : 0.0;
        snapshot.stateValid = latest.state.valid;
        snapshot.imageValid = latest.image.valid;
        snapshot.stateAgeSeconds = AgeSeconds(latest.state.timestamp, nowSeconds);
        snapshot.imageAgeSeconds = AgeSeconds(latest.image.timestamp, nowSeconds);
        snapshot.robotConnected = config_.previewUi ||
                                  (snapshot.stateValid && snapshot.stateAgeSeconds >= 0.0 &&
                                   snapshot.stateAgeSeconds <= kStateTimeoutSeconds);
        snapshot.bodyHeight = latest.state.bodyHeight;
        snapshot.roll = latest.state.roll;
        snapshot.pitch = latest.state.pitch;
        snapshot.yaw = latest.state.yaw;
        snapshot.positionX = latest.state.positionX;
        snapshot.positionY = latest.state.positionY;
        snapshot.positionZ = latest.state.positionZ;
        snapshot.velocityX = latest.state.velocityX;
        snapshot.velocityY = latest.state.velocityY;
        snapshot.velocityZ = latest.state.velocityZ;
        snapshot.yawSpeed = latest.state.yawSpeed;
        snapshot.errorCode = latest.state.errorCode;
        snapshot.mode = latest.state.mode;
        snapshot.gaitType = latest.state.gaitType;
        snapshot.commandVx = latest.action.vx;
        snapshot.commandVy = latest.action.vy;
        snapshot.commandWz = latest.action.wz;
        snapshot.sceneId = editable.sceneId;
        snapshot.operatorId = editable.operatorId;
        snapshot.instruction = editable.instruction;
        snapshot.taskFamily = editable.taskFamily;
        snapshot.targetType = editable.targetType;
        snapshot.targetDescription = editable.targetDescription;
        snapshot.collectorNotes = editable.collectorNotes;
        snapshot.cmdVxMax = editable.cmdVxMax;
        snapshot.cmdVyMax = editable.cmdVyMax;
        snapshot.cmdWzMax = editable.cmdWzMax;
        snapshot.networkInterface = config_.previewUi ? "preview" : config_.networkInterface;
        snapshot.outputDir = config_.outputDir.string();
        snapshot.loopHz = config_.loopHz;
        snapshot.videoPollHz = config_.videoPollHz;
        snapshot.inputBackend = InputBackendName(config_.inputBackend);
        snapshot.inputDevice = config_.inputDevice.string();
        snapshot.defaultsPath = CollectorDefaultsPath(config_.collectorRoot).string();
        if (d435iCaptureManager_)
        {
            const D435iStatus d435iStatus = d435iCaptureManager_->StatusSnapshot();
            snapshot.d435iEnabled = true;
            snapshot.d435iActive = d435iStatus.active;
            snapshot.d435iDegraded = d435iStatus.degraded;
            snapshot.d435iDegradedReason = d435iStatus.degradedReason;
            snapshot.d435iColorValid = d435iStatus.colorValid;
            snapshot.d435iColorFresh = d435iStatus.colorFresh;
            snapshot.d435iDepthValid = d435iStatus.depthValid;
            snapshot.d435iDepthFresh = d435iStatus.depthFresh;
            snapshot.d435iColorSeq = d435iStatus.colorSeq;
            snapshot.d435iDepthSeq = d435iStatus.depthSeq;
            snapshot.d435iColorAgeSeconds = d435iStatus.colorAgeSeconds;
            snapshot.d435iDepthAgeSeconds = d435iStatus.depthAgeSeconds;
            snapshot.d435iBagPath = d435iCaptureManager_->RelativeBagPath().string();
        }
        snapshot.pendingLabelActive = loggerStatus.pendingLabel;
        snapshot.pendingEpisodeId = loggerStatus.pendingEpisodeId;
        snapshot.pendingLabelBufferedFrames = loggerStatus.bufferedFrames;
        snapshot.captureState = CaptureStateName(captureState);
        if (loggerStatus.pendingLabel)
        {
            snapshot.stopPhase = "label_ready";
        }
        snapshot.safetyState = SafetyStateName(safetyState);
        snapshot.faultReason = faultReason;
        snapshot.recording = captureState == CaptureState::Capturing;
        snapshot.actions =
            BuildUiActionAvailability(loggerStatus, captureState, safetyState, snapshot.startupGateActive);
        return snapshot;
    }

    UiActionResult RequestBeginSegment()
    {
        return BeginSegmentInternal(true);
    }

    UiActionResult RequestStopSegmentForLabel()
    {
        return StopSegmentForLabelInternal(true);
    }

    UiActionResult RequestDiscardSegment(const std::string& reason)
    {
        return DiscardSegmentInternal(reason, true);
    }

    UiActionResult RequestEmergencyStopFromUi()
    {
        SafetyState safetyState;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            safetyState = safetyState_;
        }
        if (safetyState == SafetyState::EstopLatched)
        {
            return ActionError("already_estop", "急停已锁定");
        }
        RequestEmergencyStop("web_ui_estop");
        return ActionOk("急停已锁定");
    }

    UiActionResult RequestClearFaultFromUi()
    {
        bool hasFault = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            hasFault = safetyState_ != SafetyState::SafeReady;
        }
        if (!hasFault)
        {
            return ActionError("no_fault", "当前没有 safety fault");
        }
        std::string reason;
        if (!CanClearFault(reason))
        {
            return ActionError("cannot_clear_fault", "无法清除 safety fault：" + reason);
        }
        ClearLatchedFault();
        PrintLine("safety fault 已清除");
        return ActionOk("safety fault 已清除");
    }

    UiActionResult RequestQuitFromUi()
    {
        RequestQuit();
        PrintLine("collector 正在退出");
        return ActionOk("collector 正在退出");
    }

    UiActionResult SubmitPendingLabel(const SegmentLabelInput& input)
    {
        const auto loggerStatus = logger_.GetStatus();
        if (!loggerStatus.pendingLabel)
        {
            return ActionError("no_pending_label", "当前没有待标注区间");
        }

        const std::string segmentStatus = Trim(input.segmentStatus);
        if (!IsValidSegmentStatusValue(segmentStatus))
        {
            return ActionError("validation_error", "segment_status 非法");
        }
        if (segmentStatus == SegmentStatusName(SegmentStatus::Discard))
        {
            return DiscardSegmentInternal("discarded by web label", true);
        }

        const std::string success = Trim(input.success);
        const std::string terminationReason = Trim(input.terminationReason);
        if (!IsValidSuccessValue(success))
        {
            return ActionError("validation_error", "success 非法");
        }
        if (!IsValidTerminationReasonValue(terminationReason))
        {
            return ActionError("validation_error", "termination_reason 非法");
        }

        TaskMetadata labelMetadata;
        labelMetadata.segmentStatus = segmentStatus;
        labelMetadata.success = success;
        labelMetadata.terminationReason = terminationReason;
        return FinalizePendingLabelInternal(labelMetadata, true);
    }

    std::vector<uint8_t> GetLatestImageJpeg() const
    {
        std::lock_guard<std::mutex> lock(imageMutex_);
        if (!latestImage_.valid || latestImage_.jpegBytes.empty())
        {
            return {};
        }
        return latestImage_.jpegBytes;
    }

    UiImageFrame WaitForNextImageFrame(uint64_t lastSequence, int timeoutMs)
    {
        UiImageFrame frame;
        std::unique_lock<std::mutex> lock(imageMutex_);
        const auto hasNewFrame = [&]()
        {
            return !running_.load() ||
                   (latestImage_.valid && !latestImage_.jpegBytes.empty() && latestImage_.sequence > lastSequence);
        };

        if (!hasNewFrame())
        {
            imageUpdatedCv_.wait_for(lock, std::chrono::milliseconds(std::max(timeoutMs, 1)), hasNewFrame);
        }

        if (!running_.load() ||
            !latestImage_.valid ||
            latestImage_.jpegBytes.empty() ||
            latestImage_.sequence <= lastSequence)
        {
            return frame;
        }

        frame.timestamp = latestImage_.timestamp;
        frame.sequence = latestImage_.sequence;
        frame.jpegBytes = latestImage_.jpegBytes;
        frame.valid = true;
        return frame;
    }

private:
    struct Snapshot
    {
        LatestState state;
        LatestImage image;
        VelocityCommand action;
    };

    Snapshot CollectLatestSnapshot() const
    {
        Snapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(stateMutex_);
            snapshot.state = latestState_;
        }
        {
            std::lock_guard<std::mutex> lock(imageMutex_);
            snapshot.image = latestImage_;
        }
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            snapshot.action = latestCommand_;
        }
        return snapshot;
    }

    static bool HasJpegExtension(const fs::path& path)
    {
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch)
        {
            return static_cast<char>(std::tolower(ch));
        });
        return extension == ".jpg" || extension == ".jpeg";
    }

    std::optional<fs::path> FindPreviewImagePath() const
    {
        const std::vector<fs::path> searchRoots = {
            config_.collectorRoot / "data",
            config_.collectorRoot / "data_min10",
        };
        for (const fs::path& root : searchRoots)
        {
            std::error_code existsError;
            if (!fs::exists(root, existsError) || existsError)
            {
                continue;
            }

            std::error_code walkError;
            fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, walkError);
            fs::recursive_directory_iterator end;
            while (it != end && !walkError)
            {
                const fs::path candidate = it->path();
                std::error_code statusError;
                if (it->is_directory(statusError))
                {
                    const std::string name = candidate.filename().string();
                    if (name == "llada_vla_converted" || name == "build")
                    {
                        it.disable_recursion_pending();
                    }
                }
                else if (it->is_regular_file(statusError) && HasJpegExtension(candidate))
                {
                    return candidate;
                }
                it.increment(walkError);
            }
        }
        return std::nullopt;
    }

    void InitializePreviewMode()
    {
        previewImageJpeg_.clear();
        if (const auto previewImagePath = FindPreviewImagePath(); previewImagePath.has_value())
        {
            std::ifstream input(previewImagePath.value(), std::ios::binary);
            if (input.is_open())
            {
                previewImageJpeg_ =
                    std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
            }
            if (!previewImageJpeg_.empty())
            {
                PrintLine("preview ui 复用本地样例图像：" + previewImagePath.value().string());
            }
        }
        if (previewImageJpeg_.empty())
        {
            PrintLine("preview ui 未找到本地 jpg 样例，Camera 区域将保持等待状态");
        }

        VelocityCommand command;
        command.timestamp = NowSeconds();
        command.valid = true;
        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            latestCommand_ = command;
        }
        UpdatePreviewState(command, 0.0);
        PublishPreviewImageFrame();
    }

    void UpdatePreviewState(const VelocityCommand& command, double deltaSeconds)
    {
        const double nowSeconds = NowSeconds();
        std::lock_guard<std::mutex> lock(stateMutex_);
        LatestState state = latestState_;
        state.timestamp = nowSeconds;
        state.valid = true;
        state.roll = 0.03f * std::sin(static_cast<float>(nowSeconds * 0.7));
        state.pitch = 0.02f * std::cos(static_cast<float>(nowSeconds * 0.5));
        state.yaw += command.wz * static_cast<float>(deltaSeconds);
        const float worldVx = command.vx * std::cos(state.yaw) - command.vy * std::sin(state.yaw);
        const float worldVy = command.vx * std::sin(state.yaw) + command.vy * std::cos(state.yaw);
        state.positionX += worldVx * static_cast<float>(deltaSeconds);
        state.positionY += worldVy * static_cast<float>(deltaSeconds);
        state.velocityX = command.vx;
        state.velocityY = command.vy;
        state.velocityZ = 0.0f;
        state.yawSpeed = command.wz;
        state.bodyHeight = 0.30f;
        state.gaitType = (std::fabs(command.vx) > 1e-3f || std::fabs(command.vy) > 1e-3f || std::fabs(command.wz) > 1e-3f)
                             ? 1
                             : 0;
        latestState_ = state;
    }

    void PublishPreviewImageFrame()
    {
        if (previewImageJpeg_.empty())
        {
            return;
        }

        LatestImage image;
        image.timestamp = NowSeconds();
        image.sequence = nextImageSequence_++;
        image.jpegBytes = previewImageJpeg_;
        image.valid = true;
        {
            std::lock_guard<std::mutex> lock(imageMutex_);
            latestImage_ = std::move(image);
        }
        imageUpdatedCv_.notify_all();
    }

    static UiActionAvailability BuildUiActionAvailability(const TrajectoryLogger::Status& loggerStatus,
                                                          CaptureState captureState,
                                                          SafetyState safetyState,
                                                          bool startupGateActive)
    {
        (void)startupGateActive;
        UiActionAvailability actions;
        actions.canStartRecording = safetyState == SafetyState::SafeReady &&
                                    captureState == CaptureState::Idle &&
                                    !loggerStatus.pendingLabel;
        actions.canStopRecording = captureState == CaptureState::Capturing;
        actions.canDiscardSegment = captureState == CaptureState::Capturing ||
                                    loggerStatus.pendingLabel;
        actions.canSubmitLabel = loggerStatus.pendingLabel;
        actions.canEstop = safetyState != SafetyState::EstopLatched;
        actions.canClearFault = safetyState != SafetyState::SafeReady;
        return actions;
    }

    using MotionStateSnapshot = collector::input::MotionState;

    MotionStateSnapshot GetMotionStateSnapshot() const
    {
        if (!inputBackend_)
        {
            return MotionStateSnapshot{};
        }
        return inputBackend_->GetMotionState(NowSeconds(), std::chrono::steady_clock::now());
    }

    static bool IsMotionInputActive(const MotionStateSnapshot& motionState)
    {
        return motionState.forward || motionState.backward || motionState.left ||
               motionState.right || motionState.yawLeft || motionState.yawRight;
    }

    TrajectoryStopFlowAction EvaluateTrajectoryStopFlowLocked(const MotionStateSnapshot& motionState) const
    {
        if (captureState_ != CaptureState::Capturing || !trajectoryStopRequested_)
        {
            return TrajectoryStopFlowAction::None;
        }

        const auto now = std::chrono::steady_clock::now();
        if (trajectoryStopWaitingForRelease_)
        {
            if (!IsMotionInputActive(motionState))
            {
                return TrajectoryStopFlowAction::StartGracePeriod;
            }
            if (HasSteadyDeadline(trajectoryStopForceFinalizeDeadline_) && now >= trajectoryStopForceFinalizeDeadline_)
            {
                return TrajectoryStopFlowAction::StartGracePeriodAfterTimeout;
            }
            return TrajectoryStopFlowAction::None;
        }

        if (HasSteadyDeadline(trajectoryStopFinalizeDeadline_) && now >= trajectoryStopFinalizeDeadline_)
        {
            return TrajectoryStopFlowAction::Finalize;
        }
        return TrajectoryStopFlowAction::None;
    }

    void UpdateTrajectoryStopFlow(const MotionStateSnapshot& motionState)
    {
        const bool loggerStopReady = logger_.ConsumeTrajectoryStopReady();
        TrajectoryStopFlowAction action = TrajectoryStopFlowAction::None;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ == CaptureState::Capturing && trajectoryStopRequested_ && loggerStopReady)
            {
                action = TrajectoryStopFlowAction::Finalize;
            }
            else
            {
                action = EvaluateTrajectoryStopFlowLocked(motionState);
            }
        }

        if (action == TrajectoryStopFlowAction::StartGracePeriodAfterTimeout)
        {
            StartTrajectoryFinalizeGracePeriod("Stopping segment timed out; finalizing segment... submit a score", true);
            return;
        }
        if (action == TrajectoryStopFlowAction::StartGracePeriod)
        {
            StartTrajectoryFinalizeGracePeriod("Finalizing segment... submit a score", true);
            return;
        }
        if (action == TrajectoryStopFlowAction::Finalize)
        {
            CompleteTrajectoryStopIfReady(true);
        }
    }

    void PrintLine(const std::string& line) const
    {
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2K" << line << std::endl;
    }

    std::optional<std::string> PromptForSelection(
        const std::string& title,
        const std::vector<std::pair<std::string, std::string>>& options)
    {
        if (!isatty(STDIN_FILENO))
        {
            PrintLine("stdin 不是 tty，无法进行区间标注");
            return std::nullopt;
        }

        promptCancelRequested_.store(false);
        promptActive_.store(true);
        struct PromptRestore
        {
            std::atomic<bool>& promptActive;

            ~PromptRestore()
            {
                promptActive.store(false);
            }
        } restore{promptActive_};

        while (running_.load())
        {
            {
                std::lock_guard<std::mutex> lock(outputMutex_);
                std::cout << "\r\33[2K" << title << std::endl;
                for (const auto& option : options)
                {
                    std::cout << "  " << option.first << " = " << option.second << std::endl;
                }
                std::cout << "  ESC = discard" << std::endl;
                std::cout << "> " << std::flush;
            }

            while (running_.load())
            {
                if (promptCancelRequested_.exchange(false))
                {
                    return std::nullopt;
                }

                pollfd pfd{STDIN_FILENO, POLLIN, 0};
                const int pollResult = ::poll(&pfd, 1, 100);
                if (pollResult < 0)
                {
                    PrintLine("等待标注输入失败");
                    return std::nullopt;
                }
                if (pollResult == 0 || (pfd.revents & POLLIN) == 0)
                {
                    continue;
                }

                char ch = 0;
                const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
                if (bytesRead <= 0)
                {
                    continue;
                }
                if (ch == kEscapeKey)
                {
                    return std::nullopt;
                }
                if (ch == '\r' || ch == '\n')
                {
                    continue;
                }

                const std::string input(1, ch);
                for (const auto& option : options)
                {
                    if (input == option.first)
                    {
                        return option.second;
                    }
                }
                PrintLine("输入无效，请按编号重新输入，或按 ESC 丢弃");
                break;
            }
        }
        return std::nullopt;
    }

    std::optional<SegmentLabelInput> PromptQuickLabel()
    {
        const auto selection = PromptForSelection(
            "请选择数据评分",
            {
                {"1", "好的成功示范 (clean + success + goal_reached)"},
                {"2", "可用但不完美 (usable + partial + near_goal_stop)"},
                {"3", "失败但有价值 (usable + fail + operator_stop)"},
                {"4", "丢弃 (discard)"},
            });
        if (!selection.has_value())
        {
            return std::nullopt;
        }

        if (selection.value() == "好的成功示范 (clean + success + goal_reached)")
        {
            return BuildPresetLabelFromShortcut('1');
        }
        if (selection.value() == "可用但不完美 (usable + partial + near_goal_stop)")
        {
            return BuildPresetLabelFromShortcut('2');
        }
        if (selection.value() == "失败但有价值 (usable + fail + operator_stop)")
        {
            return BuildPresetLabelFromShortcut('3');
        }
        if (selection.value() == "丢弃 (discard)")
        {
            return BuildPresetLabelFromShortcut('4');
        }
        return std::nullopt;
    }

    static std::optional<SegmentLabelInput> BuildPresetLabelFromShortcut(char shortcut)
    {
        SegmentLabelInput input;
        switch (shortcut)
        {
        case '1':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Clean);
            input.success = SuccessLabelName(SuccessLabel::Success);
            input.terminationReason = TerminationReasonName(TerminationReason::GoalReached);
            return input;
        case '2':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Usable);
            input.success = SuccessLabelName(SuccessLabel::Partial);
            input.terminationReason = TerminationReasonName(TerminationReason::NearGoalStop);
            return input;
        case '3':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Usable);
            input.success = SuccessLabelName(SuccessLabel::Fail);
            input.terminationReason = TerminationReasonName(TerminationReason::OperatorStop);
            return input;
        case '4':
            input.segmentStatus = SegmentStatusName(SegmentStatus::Discard);
            return input;
        default:
            return std::nullopt;
        }
    }

    void PrintHelp() const
    {
        std::lock_guard<std::mutex> lock(outputMutex_);
        std::cout << "\r\33[2KControls:" << std::endl;
        if (inputBackend_)
        {
            inputBackend_->AppendHelp(std::cout);
        }
        std::cout << "  trajectory 模式：按 R 启动采集流程，连续有效动作后开始写入，按 T 请求结束并等待动作回落后标注" << std::endl;
        std::cout << "  该模式下必须提供 --instruction，并作为轨迹级语义标签保存" << std::endl;
        if (config_.previewUi)
        {
            std::cout << "  当前为 preview ui 模式：无需 Go2，状态/图像来自本地预览数据" << std::endl;
        }
        std::cout << "  input_backend=" << InputBackendName(config_.inputBackend)
                  << " capture_flow=" << kTrajectoryCaptureMode << std::endl;
    }

    void PrintStatus() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const Snapshot latest = CollectLatestSnapshot();
        const auto loggerStatus = logger_.GetStatus();
        const double nowSeconds = NowSeconds();
        SafetyState safetyState;
        std::string faultReason;
        CaptureState captureState;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            safetyState = safetyState_;
            faultReason = latchedFaultReason_;
            captureState = captureState_;
        }

        std::ostringstream oss;
        oss << std::boolalpha
            << "capture_state=" << CaptureStateName(captureState)
            << " safety_state=" << SafetyStateName(safetyState)
            << " fault_reason=" << (faultReason.empty() ? "-" : faultReason)
            << " scene_id=" << editable.sceneId
            << " operator_id=" << editable.operatorId
            << " preview_ui=" << config_.previewUi
            << " configured_instruction=" << (editable.instruction.empty() ? "-" : editable.instruction)
            << " task_family=" << (editable.taskFamily.empty() ? "-" : editable.taskFamily)
            << " target_type=" << (editable.targetType.empty() ? "-" : editable.targetType)
            << " target_description=" << (editable.targetDescription.empty() ? "-" : editable.targetDescription)
            << " buffered_frames=" << loggerStatus.bufferedFrames
            << " state=" << latest.state.valid
            << " image=" << latest.image.valid
            << " state_age_s=" << std::fixed << std::setprecision(3) << AgeSeconds(latest.state.timestamp, nowSeconds)
            << " image_age_s=" << AgeSeconds(latest.image.timestamp, nowSeconds)
            << " command=(" << latest.action.vx << "," << latest.action.vy << "," << latest.action.wz << ")";
        if (inputBackend_)
        {
            inputBackend_->AppendStatus(oss, nowSeconds);
        }
        if (config_.inputBackend == Config::InputBackend::WirelessController)
        {
            oss << " wireless_mode=collector_button_map"
                << " native_passthrough=" << (nativeJoystickEnabled_.load() ? "enabled" : "disabled");
        }
        PrintLine(oss.str());
    }

    std::string EvaluateSafetyFault(const Snapshot& snapshot, double nowSeconds) const
    {
        if (!snapshot.state.valid)
        {
            return "state_unavailable";
        }
        const double stateAge = AgeSeconds(snapshot.state.timestamp, nowSeconds);
        if (stateAge < 0.0 || stateAge > kStateTimeoutSeconds)
        {
            return "state_timeout";
        }
        if (std::fabs(snapshot.state.roll) > kMaxSafeAbsRollRad)
        {
            return "roll_limit_exceeded";
        }
        if (std::fabs(snapshot.state.pitch) > kMaxSafeAbsPitchRad)
        {
            return "pitch_limit_exceeded";
        }
        return "";
    }

    bool CanClearFault(std::string& reason) const
    {
        const Snapshot snapshot = CollectLatestSnapshot();
        reason = EvaluateSafetyFault(snapshot, NowSeconds());
        return reason.empty();
    }

    void ClearLatchedFault()
    {
        std::lock_guard<std::mutex> lock(stateMachineMutex_);
        safetyState_ = SafetyState::SafeReady;
        latchedFaultReason_.clear();
        if (captureState_ == CaptureState::Fault)
        {
            captureState_ = CaptureState::Idle;
        }
    }

    void LatchFault(SafetyState state, const std::string& reason)
    {
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            safetyState_ = state;
            latchedFaultReason_ = reason;
            captureState_ = CaptureState::Fault;
            ResetCaptureProgressLocked();
        }
        logger_.DiscardPendingSegment();
        if (inputBackend_)
        {
            inputBackend_->ResetMotionState();
        }
        SetWirelessControllerPassthroughEnabled(false, "safety fault latched");
        StopMotion();
    }

    void RequestEmergencyStop(const std::string& reason)
    {
        LatchFault(SafetyState::EstopLatched, reason);
        PrintLine("急停已锁定：" + reason);
    }

    bool TryClearFault()
    {
        bool hasFault = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            hasFault = safetyState_ != SafetyState::SafeReady;
        }

        if (!hasFault)
        {
            return false;
        }

        std::string reason;
        if (CanClearFault(reason))
        {
            ClearLatchedFault();
            if (UsesWirelessNativePassthrough(config_))
            {
                SetWirelessControllerPassthroughEnabled(true, "fault cleared");
            }
            PrintLine("safety fault 已清除");
        }
        else
        {
            PrintLine("无法清除 safety fault：" + reason);
        }
        return true;
    }

    void ToggleStandState()
    {
        bool segmentActive = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            segmentActive = captureState_ == CaptureState::Capturing;
        }

        if (segmentActive)
        {
            PrintLine("请先结束或丢弃当前采集段");
            return;
        }

        LatestState state;
        {
            std::lock_guard<std::mutex> stateLock(stateMutex_);
            state = latestState_;
        }
        const bool shouldStandDown = state.valid && state.bodyHeight >= kStandingBodyHeightThreshold;

        try
        {
            std::lock_guard<std::mutex> sportLock(sportClientMutex_);
            if (!sportClient_)
            {
                PrintLine("sport client 尚未准备好");
                return;
            }
            sportClient_->StopMove();
            if (shouldStandDown)
            {
                sportClient_->StandDown();
                PrintLine("已请求趴下");
            }
            else
            {
                sportClient_->StandUp();
                PrintLine("已请求站起");
            }
        }
        catch (...)
        {
            PrintLine("切换站立状态失败");
        }
    }

    void ClearFaultOrToggleStandState()
    {
        if (TryClearFault())
        {
            return;
        }
        ToggleStandState();
    }

    TaskMetadata ConfiguredTaskMetadata() const
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        TaskMetadata metadata;
        metadata.instruction = editable.instruction;
        metadata.taskFamily = editable.taskFamily;
        metadata.targetType = editable.targetType;
        metadata.targetLabel = ResolveTargetText(editable.targetType, editable.targetDescription);
        metadata.targetDescription = editable.targetDescription;
        metadata.collectorNotes = editable.collectorNotes;
        metadata.instructionSource = editable.instructionSource;
        return metadata;
    }

    std::optional<UiActionResult> ValidateBeginSegmentLocked(bool hasPendingLabel, bool emitLog) const
    {
        if (safetyState_ != SafetyState::SafeReady)
        {
            if (emitLog)
            {
                PrintLine("当前 safety fault 已锁定，无法启动采集段");
            }
            return ActionError("fault_latched", "当前 safety fault 已锁定");
        }
        if (captureState_ == CaptureState::Capturing)
        {
            if (emitLog)
            {
                PrintLine("当前已有采集段正在进行");
            }
            return ActionError("already_capturing", "当前已有采集段正在进行");
        }
        if (hasPendingLabel)
        {
            if (emitLog)
            {
                PrintLine("当前有待标注区间，请先提交或丢弃");
            }
            return ActionError("pending_label", "当前有待标注区间");
        }
        return std::nullopt;
    }

    void EnterPreparedSegmentStateLocked()
    {
        captureState_ = CaptureState::Capturing;
        ResetCaptureProgressLocked();
    }

    static const char* BeginSegmentReadyMessage()
    {
        return "trajectory 采集已启动；检测到连续有效动作后开始写入，使用停止键结束，使用丢弃键取消";
    }

    void MaybePromptPendingLabelAfterStop()
    {
        if (config_.webUiEnabled || !logger_.GetStatus().pendingLabel)
        {
            return;
        }
        pendingTerminalLabelPrompt_.store(true);
    }

    void MaybeRunPendingTerminalLabelPrompt()
    {
        if (!pendingTerminalLabelPrompt_.exchange(false))
        {
            return;
        }
        if (!running_.load() || config_.webUiEnabled || !logger_.GetStatus().pendingLabel)
        {
            return;
        }
        PromptAndFinalizePendingSegment();
    }

    UiActionResult BeginSegmentInternal(bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const TaskMetadata configuredTaskMetadata = ConfiguredTaskMetadata();
        std::lock_guard<std::mutex> lock(stateMachineMutex_);
        if (const auto validationResult = ValidateBeginSegmentLocked(hasPendingLabel, emitLog))
        {
            return *validationResult;
        }

        try
        {
            logger_.BeginSegment(editable.sceneId, editable.operatorId, configuredTaskMetadata, trajectoryGateConfig_);
            EnterPreparedSegmentStateLocked();
            if (emitLog)
            {
                PrintLine(BeginSegmentReadyMessage());
            }
        }
        catch (const std::exception& ex)
        {
            if (emitLog)
            {
                PrintLine(std::string("启动采集段失败：") + ex.what());
            }
            return ActionError("start_failed", ex.what());
        }

        return ActionOk("segment started");
    }

    UiActionResult FinalizeCaptureForLabelInternal(bool emitLog)
    {
        const size_t frameCount = logger_.EndSegmentForLabel();
        if (frameCount == 0)
        {
            logger_.DiscardPendingSegment();
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            EnterIdleOrFaultStateLocked();
            if (emitLog)
            {
                PrintLine("该段未记录到有效动作帧，已丢弃");
            }
            return ActionError("empty_segment", "该段未记录到有效动作帧");
        }

        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            EnterIdleOrFaultStateLocked();
        }
        if (emitLog)
        {
            PrintLine("Entering labeling state... submit 1/2/3/4 or D-pad score");
        }
        return ActionOk("segment stopped for labeling");
    }

    UiActionResult RequestTrajectoryStopInternal(bool emitLog)
    {
        if (!logger_.HasEffectiveMotion())
        {
            return FinalizeCaptureForLabelInternal(emitLog);
        }

        const bool waitingForRelease = IsMotionInputActive(GetMotionStateSnapshot());
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing)
            {
                return ActionError("not_capturing", "当前没有可结束的采集段");
            }
            if (trajectoryStopRequested_)
            {
                return ActionError("already_stopping", "trajectory 已收到结束请求，正在等待动作回落");
            }
            BeginTrajectoryStopLocked(waitingForRelease);
        }
        logger_.RequestTrajectoryStop();

        if (emitLog)
        {
            PrintLine(TrajectoryStopStatusMessage(waitingForRelease));
        }
        return ActionOk("trajectory stop requested");
    }

    UiActionResult StopSegmentForLabelInternal(bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing)
            {
                if (hasPendingLabel)
                {
                    return ActionError("already_waiting_label", "当前区间正在等待标注");
                }
                if (emitLog)
                {
                    PrintLine("当前没有可结束的采集段");
                }
                return ActionError("not_capturing", "当前没有可结束的采集段");
            }
        }
        return RequestTrajectoryStopInternal(emitLog);
    }

    void CompleteTrajectoryStopIfReady(bool emitLog)
    {
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            if (captureState_ != CaptureState::Capturing || !trajectoryStopRequested_)
            {
                return;
            }
        }
        const UiActionResult result = FinalizeCaptureForLabelInternal(emitLog);
        if (!result.ok)
        {
            return;
        }
        MaybePromptPendingLabelAfterStop();
    }

    UiActionResult DiscardSegmentInternal(const std::string& reason, bool emitLog)
    {
        const bool hasPendingLabel = logger_.GetStatus().pendingLabel;
        bool hasActiveOrPending = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            hasActiveOrPending = captureState_ == CaptureState::Capturing;
            if (!hasActiveOrPending)
            {
                hasActiveOrPending = hasPendingLabel;
            }
            if (!hasActiveOrPending)
            {
                return ActionError("nothing_to_discard", "当前没有可丢弃区间");
            }
            EnterIdleOrFaultStateLocked();
        }
        logger_.DiscardPendingSegment();
        if (emitLog)
        {
            PrintLine("采集段已丢弃：" + reason);
        }
        return ActionOk("segment discarded");
    }

    UiActionResult FinalizePendingLabelInternal(const TaskMetadata& labelMetadata, bool emitLog)
    {
        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const std::string savedInstruction = editable.instruction;
        try
        {
            const auto episodeId = logger_.FinalizePendingSegment(savedInstruction, labelMetadata);
            if (!episodeId.has_value())
            {
                return ActionError("no_pending_label", "当前没有待保存的采集段");
            }
            if (emitLog)
            {
                PrintLine(
                    "已保存 episode " + episodeId.value() +
                    " 指令=" + savedInstruction +
                    " segment_status=" + labelMetadata.segmentStatus +
                    " success=" + labelMetadata.success +
                    " termination_reason=" + labelMetadata.terminationReason);
            }
            return ActionOk("episode saved", episodeId.value());
        }
        catch (const std::exception& ex)
        {
            logger_.DiscardPendingSegment();
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                EnterIdleOrFaultStateLocked();
            }
            if (emitLog)
            {
                PrintLine(std::string("结束采集段失败：") + ex.what());
            }
            return ActionError("finalize_failed", ex.what());
        }
    }

    void PromptAndFinalizePendingSegment()
    {
        const auto quickLabel = PromptQuickLabel();
        if (!quickLabel.has_value())
        {
            DiscardSegmentInternal("cancelled during labeling", true);
            return;
        }

        const UiActionResult result = SubmitPendingLabel(quickLabel.value());
        if (!result.ok)
        {
            PrintLine("标注提交失败：" + result.message);
        }
    }

    void BeginSegment()
    {
        BeginSegmentInternal(true);
    }

    void EndSegment()
    {
        const UiActionResult result = StopSegmentForLabelInternal(true);
        if (!result.ok)
        {
            return;
        }
        MaybePromptPendingLabelAfterStop();
    }

    void CancelCurrentSegment(const std::string& reason)
    {
        DiscardSegmentInternal(reason, true);
    }

    void StopMotion()
    {
        try
        {
            std::lock_guard<std::mutex> lock(sportClientMutex_);
            if (sportClient_)
            {
                sportClient_->StopMove();
            }
        }
        catch (...)
        {
            PrintLine("StopMove 调用失败");
        }
    }

    bool SetWirelessControllerPassthroughEnabled(bool enabled, const std::string& reason)
    {
        if (config_.inputBackend != Config::InputBackend::WirelessController)
        {
            return true;
        }

        const bool alreadyEnabled = nativeJoystickEnabled_.load();
        const bool stateKnown = nativeJoystickStateKnown_.load();
        if (stateKnown && alreadyEnabled == enabled)
        {
            return true;
        }

        std::string error;
        {
            std::lock_guard<std::mutex> lock(sportClientMutex_);
            if (!sportClient_)
            {
                return false;
            }

            try
            {
                const int32_t result = sportClient_->SwitchJoystick(enabled);
                if (result == 0)
                {
                    nativeJoystickEnabled_.store(enabled);
                    nativeJoystickStateKnown_.store(true);
                    if (!enabled)
                    {
                        // When the collector latches a fault, stop any residual
                        // native joystick motion immediately.
                        sportClient_->StopMove();
                    }
                }
                else
                {
                    error = std::string("切换原生手柄直通失败，SwitchJoystick(") +
                            (enabled ? "true" : "false") + ") 返回码=" + std::to_string(result);
                }
            }
            catch (...)
            {
                error = std::string("切换原生手柄直通失败，SwitchJoystick(") +
                        (enabled ? "true" : "false") + ") 调用失败";
            }
        }

        if (!error.empty())
        {
            PrintLine(error);
            return false;
        }

        PrintLine(std::string("原生手柄直通已") + (enabled ? "启用" : "禁用") + "：" + reason);
        return true;
    }

    collector::input::BackendConfig BuildInputBackendConfig() const
    {
        collector::input::BackendConfig backendConfig;
        backendConfig.kind = config_.inputBackend == Config::InputBackend::WirelessController
                                 ? collector::input::BackendKind::WirelessController
                                 : collector::input::BackendKind::Evdev;
        backendConfig.inputDevice = config_.inputDevice;
        backendConfig.terminalRawEnabled = terminalRawEnabled_;
        backendConfig.wirelessTimeoutSeconds = kWirelessControllerTimeoutSeconds;
        backendConfig.keyboardLinearAccelPerSecond = kKeyboardLinearAccelPerSecond;
        backendConfig.keyboardLinearDecelPerSecond = kKeyboardLinearDecelPerSecond;
        backendConfig.keyboardYawAccelPerSecond = kKeyboardYawAccelPerSecond;
        backendConfig.keyboardYawDecelPerSecond = kKeyboardYawDecelPerSecond;
        backendConfig.linearAccelPerSecond = kLinearAccelPerSecond;
        backendConfig.linearDecelPerSecond = kLinearDecelPerSecond;
        backendConfig.yawAccelPerSecond = kYawAccelPerSecond;
        backendConfig.yawDecelPerSecond = kYawDecelPerSecond;
        backendConfig.wirelessDiscreteThreshold = kWirelessControllerDiscreteThreshold;
        backendConfig.wirelessStickDeadZone = kWirelessControllerStickDeadZone;
        backendConfig.wirelessStickSmoothing = kWirelessControllerStickSmoothing;
        backendConfig.wirelessCollectorDeadZone = kWirelessCollectorDeadZone;
        backendConfig.wirelessCollectorReleaseZone = kWirelessCollectorReleaseZone;
        backendConfig.wirelessAxisExponent = kWirelessCollectorAxisExponent;
        return backendConfig;
    }

    VelocityCommand ComputeInputCommand()
    {
        if (!inputBackend_)
        {
            return VelocityCommand{};
        }

        bool stopRequested = false;
        {
            std::lock_guard<std::mutex> lock(stateMachineMutex_);
            stopRequested = trajectoryStopRequested_;
        }

        const EditableCollectorConfig editable = GetEditableConfigSnapshot();
        const auto nowSteady = std::chrono::steady_clock::now();
        const double nowSeconds = NowSeconds();
        collector::input::ComputeCommandOptions options;
        options.stopRequested = stopRequested;
        options.nowSeconds = nowSeconds;
        options.nowSteady = nowSteady;
        options.wirelessNativePassthrough = UsesWirelessNativePassthrough(config_);
        options.cmdVxMax = static_cast<float>(editable.cmdVxMax);
        options.cmdVyMax = static_cast<float>(editable.cmdVyMax);
        options.cmdWzMax = static_cast<float>(editable.cmdWzMax);
        const collector::input::CommandIntent intent = inputBackend_->ComputeCommand(options);

        VelocityCommand command;
        command.timestamp = intent.timestamp;
        command.vx = intent.vx;
        command.vy = intent.vy;
        command.wz = intent.wz;
        command.valid = intent.valid;
        pendingSendVx_ = intent.sendVx;
        pendingSendVy_ = intent.sendVy;
        pendingSendWz_ = intent.sendWz;
        pendingSendMotion_ = intent.shouldSendMotion;
        return command;
    }

    void HandleTeleopEvent(const collector::input::TeleopEvent& event)
    {
        switch (event.type)
        {
        case collector::input::TeleopEventType::StartCapture:
            BeginSegment();
            break;
        case collector::input::TeleopEventType::StopCapture:
            EndSegment();
            break;
        case collector::input::TeleopEventType::CancelSegment:
            CancelCurrentSegment("cancelled by operator");
            break;
        case collector::input::TeleopEventType::EmergencyStop:
            RequestEmergencyStop("键盘急停");
            break;
        case collector::input::TeleopEventType::ClearFault:
            if (!TryClearFault())
            {
                PrintLine(
                    config_.inputBackend == Config::InputBackend::WirelessController
                        ? "当前无 safety fault；如需切换站立请按 Y"
                        : "当前无 safety fault；如需切换站立请按 V");
            }
            break;
        case collector::input::TeleopEventType::ToggleStand:
            ToggleStandState();
            break;
        case collector::input::TeleopEventType::PrintStatus:
            PrintStatus();
            break;
        case collector::input::TeleopEventType::PrintHelp:
            PrintHelp();
            break;
        case collector::input::TeleopEventType::Quit:
            RequestQuit();
            break;
        case collector::input::TeleopEventType::SubmitLabelShortcut:
            SubmitPresetLabelShortcut(event.shortcut);
            break;
        }
    }

    void SubmitPresetLabelShortcut(char shortcut)
    {
        const auto preset = BuildPresetLabelFromShortcut(shortcut);
        if (!preset.has_value())
        {
            return;
        }
        if (!logger_.GetStatus().pendingLabel)
        {
            return;
        }
        const UiActionResult result = SubmitPendingLabel(preset.value());
        if (!result.ok)
        {
            PrintLine("标注提交失败：" + result.message);
        }
    }

    void KeyboardLoop()
    {
        while (running_.load())
        {
            MaybeRunPendingTerminalLabelPrompt();
            if (!inputBackend_)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            const bool labelInputActive = promptActive_.load() || logger_.GetStatus().pendingLabel;
            const auto events = inputBackend_->PollEvents(labelInputActive, std::chrono::milliseconds(100));
            for (const auto& event : events)
            {
                if (promptActive_.load() && event.type == collector::input::TeleopEventType::CancelSegment)
                {
                    promptCancelRequested_.store(true);
                    continue;
                }
                HandleTeleopEvent(event);
            }
        }
    }

    void OnSportState(const void* message)
    {
        const auto* state = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
        LatestState latest;
        latest.timestamp = NowSeconds();
        latest.errorCode = state->error_code();
        latest.mode = state->mode();
        latest.roll = state->imu_state().rpy()[0];
        latest.pitch = state->imu_state().rpy()[1];
        latest.yaw = state->imu_state().rpy()[2];
        latest.positionX = state->position()[0];
        latest.positionY = state->position()[1];
        latest.positionZ = state->position()[2];
        latest.velocityX = state->velocity()[0];
        latest.velocityY = state->velocity()[1];
        latest.velocityZ = state->velocity()[2];
        latest.yawSpeed = state->yaw_speed();
        latest.bodyHeight = state->body_height();
        latest.gaitType = state->gait_type();
        latest.valid = true;

        {
            std::lock_guard<std::mutex> lock(stateMutex_);
            latestState_ = latest;
        }
    }

    void OnWirelessController(const void* message)
    {
        if (!inputBackend_)
        {
            return;
        }
        const auto* controller = static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);
        inputBackend_->HandleWirelessControllerMessage(*controller);
    }

    void ControlLoop()
    {
        const auto period = std::chrono::duration<double>(1.0 / config_.loopHz);
        bool lastMoveActive = false;

        while (running_.load())
        {
            const auto cycleStart = std::chrono::steady_clock::now();

            const MotionStateSnapshot motionState = GetMotionStateSnapshot();
            UpdateTrajectoryStopFlow(motionState);
            VelocityCommand command = ComputeInputCommand();
            CaptureState captureState;
            SafetyState safetyState;
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                captureState = captureState_;
                safetyState = safetyState_;
            }

            if (safetyState != SafetyState::SafeReady)
            {
                command = VelocityCommand{};
                command.timestamp = NowSeconds();
                command.valid = true;
                pendingSendMotion_.store(false);
                pendingSendVx_.store(0.0f);
                pendingSendVy_.store(0.0f);
                pendingSendWz_.store(0.0f);
            }

            {
                std::lock_guard<std::mutex> lock(commandMutex_);
                latestCommand_ = command;
            }

            if (config_.previewUi)
            {
                UpdatePreviewState(command, std::chrono::duration<double>(period).count());
            }

            float sendVx = command.vx;
            float sendVy = command.vy;
            float sendWz = command.wz;
            bool shouldSendMotion = config_.inputBackend != Config::InputBackend::WirelessController;
            if (config_.inputBackend == Config::InputBackend::WirelessController)
            {
                shouldSendMotion = pendingSendMotion_;
                sendVx = pendingSendVx_;
                sendVy = pendingSendVy_;
                sendWz = pendingSendWz_;
            }
            const bool moveActive = std::fabs(sendVx) > 1e-6f ||
                                    std::fabs(sendVy) > 1e-6f ||
                                    std::fabs(sendWz) > 1e-6f;

            if (shouldSendMotion || lastMoveActive)
            {
                try
                {
                    std::lock_guard<std::mutex> lock(sportClientMutex_);
                    if (sportClient_)
                    {
                        if (moveActive)
                        {
                            sportClient_->Move(sendVx, sendVy, sendWz);
                        }
                        else if (lastMoveActive)
                        {
                            sportClient_->StopMove();
                        }
                    }
                }
                catch (...)
                {
                    PrintLine("发送运动指令失败");
                }
                lastMoveActive = moveActive;
            }

            std::this_thread::sleep_until(cycleStart + period);
        }
    }

    void VideoLoop()
    {
        const auto period = std::chrono::duration<double>(1.0 / config_.videoPollHz);
        while (running_.load())
        {
            const auto cycleStart = std::chrono::steady_clock::now();

            if (config_.previewUi)
            {
                PublishPreviewImageFrame();
                std::this_thread::sleep_until(cycleStart + period);
                continue;
            }

            std::vector<uint8_t> jpegBytes;
            int32_t ret = -1;
            try
            {
                if (videoClient_)
                {
                    ret = videoClient_->GetImageSample(jpegBytes);
                }
            }
            catch (...)
            {
                ret = -1;
            }

            if (ret == 0 && !jpegBytes.empty())
            {
                LatestImage image;
                image.timestamp = NowSeconds();
                image.sequence = nextImageSequence_++;
                image.jpegBytes = std::move(jpegBytes);
                image.valid = true;

                {
                    std::lock_guard<std::mutex> lock(imageMutex_);
                    latestImage_ = std::move(image);
                }
                imageUpdatedCv_.notify_all();
            }

            std::this_thread::sleep_until(cycleStart + period);
        }
    }

    void LoggingLoop()
    {
        const auto period = std::chrono::duration<double>(1.0 / config_.loopHz);
        uint64_t lastLoggedImageSequence = 0;
        auto nextWaitLog = std::chrono::steady_clock::now();
        CaptureState previousCaptureState = CaptureState::Idle;

        while (running_.load())
        {
            const auto cycleStart = std::chrono::steady_clock::now();
            const auto wakeDeadline = cycleStart + period;

            LatestImage image;
            std::unique_lock<std::mutex> lock(imageMutex_);
            imageUpdatedCv_.wait_until(
                lock,
                wakeDeadline,
                [&]()
                {
                    return !running_.load() ||
                           (latestImage_.valid && latestImage_.sequence > lastLoggedImageSequence);
                });
            image = latestImage_;
            lock.unlock();

            LatestState state;
            VelocityCommand action;
            {
                std::lock_guard<std::mutex> lock(stateMutex_);
                state = latestState_;
            }
            {
                std::lock_guard<std::mutex> lock(commandMutex_);
                action = latestCommand_;
            }

            CaptureState captureState;
            SafetyState safetyState;
            {
                std::lock_guard<std::mutex> lock(stateMachineMutex_);
                captureState = captureState_;
                safetyState = safetyState_;
            }

            if (captureState == CaptureState::Capturing &&
                previousCaptureState != CaptureState::Capturing &&
                image.valid)
            {
                lastLoggedImageSequence = image.sequence;
            }

            Snapshot snapshot{state, image, action};
            if (safetyState == SafetyState::SafeReady)
            {
                const std::string reason = EvaluateSafetyFault(snapshot, NowSeconds());
                if (!reason.empty())
                {
                    LatchFault(SafetyState::FaultLatched, reason);
                    PrintLine("safety fault 已锁定：" + reason);
                    std::this_thread::sleep_until(wakeDeadline);
                    continue;
                }
            }

            const bool hasFreshImage = image.valid && image.sequence > lastLoggedImageSequence;
            const bool ready = captureState == CaptureState::Capturing && state.valid && hasFreshImage;
            if (ready)
            {
                const MotionStateSnapshot motionState = GetMotionStateSnapshot();
                const bool motionInputActive = IsMotionInputActive(motionState);
                const double sampleTimestamp = image.timestamp;
                const EffectiveControlAction controlAction = ResolveControlAction(action, sampleTimestamp);
                try
                {
                    logger_.LogStep(
                        sampleTimestamp,
                        state,
                        action,
                        controlAction,
                        image,
                        state.timestamp,
                        action.timestamp,
                        image.timestamp,
                        motionInputActive);
                    lastLoggedImageSequence = image.sequence;
                }
                catch (const std::exception& ex)
                {
                    PrintLine(std::string("记录样本失败：") + ex.what());
                }
            }
            else if (captureState == CaptureState::Capturing && std::chrono::steady_clock::now() >= nextWaitLog)
            {
                std::vector<std::string> missing;
                if (!state.valid)
                {
                    missing.emplace_back("state");
                }
                if (!image.valid)
                {
                    missing.emplace_back("image");
                }
                if (!missing.empty())
                {
                    std::ostringstream oss;
                    oss << "等待 ";
                    for (size_t index = 0; index < missing.size(); ++index)
                    {
                        if (index > 0)
                        {
                            oss << " 和 ";
                        }
                        oss << missing[index];
                    }
                    PrintLine(oss.str());
                    nextWaitLog = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                }
            }

            previousCaptureState = captureState;
            std::this_thread::sleep_until(wakeDeadline);
        }
    }

    Config config_;
    TrajectoryLogger logger_;
    TrajectoryMotionGateConfig trajectoryGateConfig_;
    RawTerminalGuard terminalGuard_;
    std::atomic<bool> lifecycleStarted_{false};
    std::atomic<bool> shutdownStarted_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> quitRequested_{false};

    mutable std::mutex outputMutex_;
    mutable std::mutex stateMutex_;
    mutable std::mutex imageMutex_;
    mutable std::mutex commandMutex_;
    mutable std::mutex sportClientMutex_;
    mutable std::mutex stateMachineMutex_;
    mutable std::mutex editableConfigMutex_;
    std::condition_variable imageUpdatedCv_;

    LatestState latestState_;
    LatestImage latestImage_;
    VelocityCommand latestCommand_;
    uint64_t nextImageSequence_ = 1;

    SafetyState safetyState_ = SafetyState::SafeReady;
    std::string latchedFaultReason_;
    CaptureState captureState_ = CaptureState::Idle;
    bool startupGateActive_ = false;
    bool trajectoryStopRequested_ = false;
    bool trajectoryStopWaitingForRelease_ = false;
    std::chrono::steady_clock::time_point trajectoryStopFinalizeDeadline_{};
    std::chrono::steady_clock::time_point trajectoryStopForceFinalizeDeadline_{};
    EditableCollectorConfig editableConfig_;
    std::vector<uint8_t> previewImageJpeg_;

    std::unique_ptr<collector::input::InputBackend> inputBackend_;
    std::atomic<bool> nativeJoystickEnabled_{false};
    std::atomic<bool> nativeJoystickStateKnown_{false};
    std::atomic<bool> pendingSendMotion_{false};
    std::atomic<float> pendingSendVx_{0.0f};
    std::atomic<float> pendingSendVy_{0.0f};
    std::atomic<float> pendingSendWz_{0.0f};
    bool terminalRawEnabled_ = false;

    std::unique_ptr<unitree::robot::go2::SportClient> sportClient_;
    std::unique_ptr<unitree::robot::go2::VideoClient> videoClient_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>> sportStateSubscriber_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>> wirelessControllerSubscriber_;

    std::thread keyboardThread_;
    std::thread controlThread_;
    std::thread videoThread_;
    std::thread loggingThread_;
    std::atomic<bool> promptActive_{false};
    std::atomic<bool> promptCancelRequested_{false};
    std::atomic<bool> pendingTerminalLabelPrompt_{false};
    std::unique_ptr<WebUiServer> webUiServer_;
    std::unique_ptr<D435iCaptureManager> d435iCaptureManager_;
};

} // namespace

int main(int argc, char** argv)
{
    try
    {
        gSignalStopRequested.store(false);
        std::signal(SIGINT, HandleStopSignal);
        std::signal(SIGTERM, HandleStopSignal);

        std::string error;
        const auto config = ParseArgs(argc, argv, error);
        if (!config.has_value())
        {
            std::cerr << "collector 参数错误：" << error << std::endl;
            return 2;
        }

        CollectorApp app(config.value());
        app.Start();

        while (!app.ShouldQuit())
        {
            if (gSignalStopRequested.exchange(false))
            {
                app.RequestQuit();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        app.Shutdown();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "collector 致命错误：" << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "collector 致命错误：未知异常" << std::endl;
    }

    return 1;
}
