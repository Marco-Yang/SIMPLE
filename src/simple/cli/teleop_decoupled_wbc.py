"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Teleoperation using the decoupled whole-body control policy from decoupled_wbc.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import enum
import typer
from typing_extensions import Annotated, TYPE_CHECKING
import gymnasium as gym
import numpy as np
import simple.envs as _  # import all envs

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

from simple.agents.pico_decoupled_agent import PicoDecoupledAgent
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.robots.g1_sonic import G1Sonic

import json
import os
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import time
import tyro
from datetime import datetime
from tqdm import tqdm
from PIL import Image

from decoupled_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH

# ---------------------------------------------------------------------------
# Episode recording state machine
# ---------------------------------------------------------------------------

class RecordingState(enum.Enum):
    WAITING_FOR_LANDING = "waiting_for_landing"
    RECORDING = "recording"
    EPISODE_DONE = "episode_done"


LIFT_THRESHOLD = 0.10  # metres above initial target z to end episode


def _reset_teleop_state(robot, agent) -> None:
    """Reset the teleop state without assuming the robot or agent exposes every optional attribute."""
    elastic_band = getattr(robot, "elastic_band", None)
    if elastic_band is not None:
        try:
            elastic_band.enable = False
        except Exception:
            pass

    if hasattr(agent, "_dropping"):
        agent._dropping = False

    reset_policy = getattr(agent, "reset_policy", None)
    if callable(reset_policy):
        reset_policy()

    wbc_policy = getattr(agent, "_wbc_policy", None)
    if wbc_policy is None:
        return

    lower_body_policy = getattr(wbc_policy, "lower_body_policy", None)
    if lower_body_policy is None:
        return

    try:
        lower_body_policy.use_policy_action = True
    except Exception:
        pass


def _save_episode_env_config(exporter, task, episode_index: int):
    """Write the task state_dict as environment_config into episodes.jsonl."""
    from simple.utils import NumpyArrayEncoder
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    env_conf = task.state_dict()
    lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _init_exporter(save_dir: str, task_prompt: str, robot_model, obj_names: list[str], joint_names: list[str]):
    """Create a Gr00tDataExporter for LeRobot-format recording."""
    try:
        from decoupled_wbc.data.exporter import Gr00tDataExporter
        from decoupled_wbc.data.utils import get_dataset_features, get_modality_config
    except Exception as exc:
        print(f"Warning: recorder exporter unavailable ({exc}); continuing without recording export.")
        return None

    try:
        # Force offline mode to avoid Hub DNS failures during local recording.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        features = get_dataset_features(robot_model)
        features["observation.state"]["names"] = joint_names # state joint names
        modality_config = get_modality_config(robot_model)

        # # Add torso RPY command feature (3D: roll, pitch, yaw)
        # features["observation.torso_rpy_command"] = {
        #     "dtype": "float64",
        #     "shape": (3,),
        #     "names": ["roll", "pitch", "yaw"],
        # }

        # Add object poses feature: each object has 7D (pos xyz + quat wxyz)
        num_objects = len(obj_names)
        if num_objects > 0:
            obj_names_flat = []
            for name in obj_names:
                for suffix in ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]:
                    obj_names_flat.append(f"{name}.{suffix}")
            features["observation.object_poses"] = {
                "dtype": "float64",
                "shape": (num_objects * 7,),
                "names": obj_names_flat,
            }

        exporter = Gr00tDataExporter.create(
            save_root=save_dir,
            fps=50,
            features=features,
            modality_config=modality_config,
            task=task_prompt,
            overwrite_existing=True,
        )
        return exporter
    except Exception as exc:
        print(f"Warning: recorder exporter initialization failed ({exc}); continuing without recording export.")
        return None


def _coerce_ego_view_image(image: np.ndarray) -> np.ndarray:
    """Normalize ego-view frame to uint8 HWC and resize to exporter's expected resolution."""
    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.max(arr)) if arr.size else 1.0
            if max_val <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[:2] != (RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH):
        arr = np.asarray(
            Image.fromarray(arr).resize(
                (RS_VIEW_CAMERA_WIDTH, RS_VIEW_CAMERA_HEIGHT),
                resample=Image.BILINEAR,
            )
        )

    return arr


def _build_frame(agent, obj_names: list[str], observation, privileged_info, action):
    """Assemble one recording frame from current sim state and agent caches."""
    def _action_value(key: str, default):
        if isinstance(action, dict):
            return action.get(key, default)
        if hasattr(action, key):
            return getattr(action, key)
        try:
            return action[key]
        except Exception:
            return default

    proprio = privileged_info["proprio"]
    rm = agent._dwbc_robot_model

    # action: 43-DOF target q in robot_model joint order
    # NOTE: joint matches with rm.joint_names
    action_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=action["target_q"],
        left_hand_actuated_joint_values=action["left_hand_q"], # natural order: thumb/index/middle
        right_hand_actuated_joint_values=action["right_hand_q"],
    )

    # # Extract torso RPY command directly from waist joints (waist_yaw, waist_roll, waist_pitch)
    # # These three joints fully determine the torso orientation relative to pelvis
    # waist_indices = rm.get_joint_group_indices("waist")
    # torso_rpy_command = action_q[waist_indices]

    mjcf_to_natural_order = lambda q: np.concatenate([q[:3], q[5:7], q[3:5]])
    proprio_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=proprio["body_q"],
        left_hand_actuated_joint_values=mjcf_to_natural_order(proprio["left_hand_q"]),
        right_hand_actuated_joint_values=mjcf_to_natural_order(proprio["right_hand_q"]),
    )
    proprio_joints = dict(zip(rm.joint_names, proprio_q))
    from simple.robots.g1_sonic import WHOLE_BODY_JOINTS
    assert np.allclose(observation["joint_qpos"], np.array([proprio_joints[joint] for joint in WHOLE_BODY_JOINTS], dtype=np.float32))

    navigate_cmd = np.asarray(
        _action_value("navigate_cmd", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(-1)
    if navigate_cmd.size < 3:
        navigate_cmd = np.pad(navigate_cmd, (0, 3 - navigate_cmd.size), mode="constant")
    elif navigate_cmd.size > 3:
        navigate_cmd = navigate_cmd[:3]

    base_height_raw = np.asarray(_action_value("base_height_command", 0.74), dtype=np.float64).reshape(-1)
    base_height_cmd = np.array([base_height_raw[0] if base_height_raw.size > 0 else 0.74], dtype=np.float64)

    frame = {
        "observation.images.ego_view": _coerce_ego_view_image(observation["head_stereo_left"]),
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64), # obs_state
        "observation.eef_state": np.asarray(action["action_eef"], dtype=np.float64), # FIXME
        "action": np.asarray(action_q, dtype=np.float64),
        "action.eef": np.asarray(action["action_eef"], dtype=np.float64), # 1-cycle delayed teleop eef
        "observation.img_state_delta": np.array([0.0], dtype=np.float32), # FIXME
        "teleop.navigate_command": navigate_cmd,
        "teleop.base_height_command": base_height_cmd,
    }

    # Object poses: concatenate all object (pos + quat) in order
    if obj_names:
        obj_poses = []
        for name in obj_names:
            obj_poses.append(privileged_info[name])
        frame["observation.object_poses"] = np.concatenate(obj_poses).astype(np.float64)

    return frame


def _load_sonic_config() -> dict:
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPick-v1",
    target: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 50,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    save_dir: Annotated[str, typer.Option()] = "data/teleop_decoupled_wbc",
    num_episodes: Annotated[int, typer.Option()] = 100,
    shard_size: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    record: Annotated[bool, typer.Option()] = False
):
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
    sim_cnt = 0

    sonic_config = _load_sonic_config()

    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=headless,
        max_episode_steps=max_episode_steps,
        sonic_config=sonic_config,
        target=target
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert sonic_env.spec is not None
    if not isinstance(robot, G1Sonic):
        print(f"Warning: robot type {type(robot).__name__} is not G1Sonic; continuing with the available robot instance.")

    agent = PicoDecoupledAgent(robot, sonic_cfg=sonic_config)
    agent.num_episodes = num_episodes

    # --- Recording setup ---
    exporter = None
    rec_state = RecordingState.WAITING_FOR_LANDING
    initial_target_z = None
    episodes_saved = 0
    control_decimal = int(1 / sonic_config["SIMULATE_DT"] / render_hz)
    robot_sim_dt = getattr(robot, "sim_dt", None)
    if robot_sim_dt is None:
        robot_sim_dt = sonic_config["SIMULATE_DT"]
    control_dt = control_decimal * robot_sim_dt  # = 0.02 s (50 Hz)
    low_latency_mode = os.getenv("SIMPLE_LOW_LATENCY_MODE", "0") == "1"

    record_every_n = max(
        1,
        int(os.getenv("SIMPLE_RECORD_EVERY_N", "2" if low_latency_mode else "1")),
    )
    viewer_every_n = max(
        1,
        int(os.getenv("SIMPLE_VIEWER_EVERY_N", "3" if low_latency_mode else "1")),
    )
    stream_every_n = max(
        1,
        int(os.getenv("SIMPLE_STREAM_EVERY_N", "2" if low_latency_mode else "1")),
    )
    update_reward_enabled = os.getenv(
        "SIMPLE_UPDATE_REWARD", "0" if low_latency_mode else "1"
    ) == "1"
    viewer_enabled = os.getenv("SIMPLE_VIEWER_ENABLE", "1") == "1"
    stream_enabled = os.getenv("SIMPLE_STREAM_ENABLE", "1") == "1"
    perf_log_interval = float(os.getenv("SIMPLE_PERF_LOG_INTERVAL_SECS", "5.0"))
    overrun_log_interval = float(
        os.getenv(
            "SIMPLE_OVERRUN_LOG_INTERVAL_SECS",
            os.getenv("SIMPLE_OVERUN_LOG_INTERVAL_SECS", "2.0"),
        )
    )
    last_overrun_log_ts = 0.0
    last_perf_log_ts = time.monotonic()

    def _on_episode_reset():
        """In recording mode, reset the WBC pipeline to a consistent initial pose,
        skip elastic band drop, and engage the RL policy immediately."""
        if not record:
            return
        _reset_teleop_state(robot, agent)

    # stabilized_printed = False
    step_pbar = None  # Progress bar for current recording episode

    # Initialize telemetry
    from decoupled_wbc.control.utils.telemetry import Telemetry
    telemetry = Telemetry(window_size=100)

    # FIXME DEBUG
    # env.unwrapped._telemetry = telemetry
    # env.unwrapped.mujoco._telemetry = telemetry

    observation, privileged_info = env.reset()
    _on_episode_reset()

    # Read obj_names after first reset so layout is populated by domain randomization.
    obj_names = list(sonic_env.mujoco.mj_objects.keys())

    if record:
        # timestamp = datetime.now().strftime("%m%d%H%M%S")
        run_save_dir = (
            f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/level-{dr_level}" #_{timestamp}
        )
        exporter = _init_exporter(
            run_save_dir,
            task.instruction, 
            agent._dwbc_robot_model, 
            obj_names,
            robot.joint_names
        )
        if exporter is not None:
            print(f"\n[Record] Exporter initialized, saving to {run_save_dir}")
            print(f"[Record] Recording {len(obj_names)} objects: {obj_names}")
        else:
            print("\n[Record] Exporter unavailable; run will continue without dataset export.")

    print(
        "[Perf] loop tuning: "
        f"low_latency_mode={low_latency_mode}, viewer_every_n={viewer_every_n}, "
        f"stream_every_n={stream_every_n}, record_every_n={record_every_n}, "
        f"update_reward={update_reward_enabled}, viewer_enabled={viewer_enabled}, "
        f"stream_enabled={stream_enabled}, "
        f"perf_log_interval={perf_log_interval}s"
    )

    try:
        while True:
            step_start = time.monotonic()

            data_frame = {}

            with telemetry.timer("agent.get_action"):
                action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            
            data_frame["observation"] = observation.copy()
            data_frame["action"] = action.copy()
            data_frame["privileged_info"] = privileged_info.copy()

            with telemetry.timer("env.step"):
                # synced_proprio = info.copy()  # capture sim state before action for recording
                # for _ in range(int(control_dt / robot.sim_dt)):
                observation, reward, terminated, truncated, privileged_info = env.step(action)

            # if "proprio" in info:
            #     agent.publish_low_state(info["proprio"])

            if agent.reset_requested:
                # Discard any in-progress recording
                if exporter is not None and rec_state == RecordingState.RECORDING:
                    print("[Record] Reset requested, discarding in-progress episode")
                    exporter.skip_and_start_new_episode()
                    # Close progress bar for discarded episode
                    if step_pbar is not None:
                        step_pbar.close()
                        step_pbar = None

                observation, privileged_info = env.reset()
                # synced_proprio = info.copy()  # capture sim state after reset for recording
                _on_episode_reset()
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                sim_cnt = 0
                # stabilized_printed = False
                rec_state = RecordingState.WAITING_FOR_LANDING
                initial_target_z = None
                print("[TeleopDecoupled] Environment reset complete")

            with telemetry.timer("update_viewer"):
                if viewer_enabled and (not headless) and (sim_cnt % viewer_every_n == 0):
                    sonic_env.update_viewer()

            with telemetry.timer("update_reward"):
                if update_reward_enabled:
                    sonic_env.update_reward()

            with telemetry.timer("update_render"):
                if stream_enabled and sim_cnt % stream_every_n == 0:
                    agent.update_render_caches(observation)

            """ # --- Print once when robot first stabilizes ---
            if robot.stabilized and not stabilized_printed:
                stabilized_printed = True
                obs = robot.prepare_obs()
                print("=" * 60)
                print("[Stabilized] Robot pose (vel converged)", datetime.now().strftime("%-H:%M:%S"), sim_cnt)
                print("-" * 60)
                print(f"  base_pose (qpos[:7]): {obs['floating_base_pose']}")
                print(f"  max |qvel[0:6]|:      {np.max(np.abs(robot.mjData.qvel[0:6])):.4f}")
                print("=" * 60) """

            with telemetry.timer("recode_data"):
                # --- Recording logic (runs at 50Hz) ---
                if exporter is not None:
                    if rec_state == RecordingState.WAITING_FOR_LANDING:
                        # Check if robot has landed (elastic band done) AND
                        # teleop policy is active (operator has re-aligned and
                        # pressed the activation button)
                        elastic_band = getattr(robot, "elastic_band", None)
                        elastic_done = (
                            not getattr(agent, "_dropping", False)
                            and (elastic_band is None or not getattr(elastic_band, "enable", False))
                        )
                        teleop_policy = getattr(agent, "_teleop_policy", None)
                        teleop_active = bool(getattr(teleop_policy, "is_active", False))
                        if elastic_done and teleop_active and getattr(robot, "stabilized", False) and getattr(agent, "_cached_target_q", None) is not None:
                            rec_state = RecordingState.RECORDING
                            initial_target_z = None
                            sim_cnt = 0  # Reset sim counter for new episode
                            # Create progress bar for this recording episode
                            step_pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}", unit="frame",
                                            leave=False, position=1, bar_format="{desc} {n_fmt} {rate_fmt}")
                            print("[Record] Teleop active, starting episode recording")

                    if rec_state == RecordingState.RECORDING:
                        # Optional frame decimation to reduce recording overhead.
                        if sim_cnt % record_every_n == 0:
                            frame = _build_frame(agent, obj_names, **data_frame)
                            exporter.add_frame(frame)
                            if step_pbar is not None:
                                step_pbar.update(1)

                        # Track initial target height
                        if initial_target_z is None and "target" in privileged_info:
                            initial_target_z = privileged_info["target"][2]

                        # Check episode end condition: target lifted >= 10cm
                        # if initial_target_z is not None and "target" in privileged_info:
                        #     current_z = privileged_info["target"][2]
                        #     if current_z - initial_target_z >= LIFT_THRESHOLD:
                        #         rec_state = RecordingState.EPISODE_DONE

                        if terminated or truncated :
                            rec_state = RecordingState.EPISODE_DONE

                    if rec_state == RecordingState.EPISODE_DONE:
                        # Close progress bar for this episode
                        if step_pbar is not None:
                            step_pbar.close()
                            step_pbar = None

                        ep_idx = exporter.episode_buffer["episode_index"]
                        exporter.save_episode()
                        _save_episode_env_config(exporter, task, ep_idx)
                        episodes_saved += 1
                        agent.episodes_saved = episodes_saved
                        print(
                            f"[Record] Episode {episodes_saved} saved "
                            f"(target lifted {LIFT_THRESHOLD}m)"
                        )
                        if episodes_saved >= num_episodes:
                            print(f"[Record] Reached {num_episodes} episodes, stopping")
                            break
                        # Auto-reset for next episode
                        observation, privileged_info = env.reset()
                        _on_episode_reset()
                        obj_names = list(sonic_env.mujoco.mj_objects.keys())
                        sim_cnt = 0
                        # stabilized_printed = False
                        rec_state = RecordingState.WAITING_FOR_LANDING
                        initial_target_z = None
                        continue  # skip sleep / increment for this iteration

            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Throttle overrun logs to avoid heavy stdout overhead in slow loops.
                now = time.monotonic()
                if now - last_overrun_log_ts >= overrun_log_interval:
                    last_overrun_log_ts = now
                    telemetry.log_timing_info(
                        context=f"{sim_cnt: >3}: Control loop overran by {-sleep_time:.3f} seconds",
                        threshold=0.001,
                    )
                # ...

            now = time.monotonic()
            if perf_log_interval > 0 and (now - last_perf_log_ts) >= perf_log_interval:
                last_perf_log_ts = now
                telemetry.log_timing_info(
                    context=f"{sim_cnt: >3}: Periodic perf summary",
                    threshold=10.0,
                    log_averages=True,
                )

            sim_cnt += 1
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
        # Close progress bar if active
        if step_pbar is not None:
            step_pbar.close()
        if exporter is not None and rec_state == RecordingState.RECORDING:
            print("[Record] Saving in-progress episode before exit...")
            try:
                ep_idx = exporter.episode_buffer["episode_index"]
                exporter.save_episode()
                _save_episode_env_config(exporter, task, ep_idx)
                episodes_saved += 1
                agent.episodes_saved = episodes_saved
                print(f"[Record] Episode {episodes_saved} saved (interrupted)")
            except Exception as e:
                print(f"[Record] Failed to save interrupted episode: {e}")
    finally:
        # Ensure progress bar is closed
        if step_pbar is not None:
            step_pbar.close()
        if exporter is not None:
            exporter.stop_video_writers()
            print(f"[Record] Done. {episodes_saved} episodes saved to {run_save_dir}")
        env.close()
        agent.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
