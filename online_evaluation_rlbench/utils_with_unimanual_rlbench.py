import os
import glob
import random

import open3d  # DON'T DELETE THIS!
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers, fetch_text_encoders
from online_evaluation_rlbench.get_stored_demos import get_stored_demos


# ALL_RLBENCH_TASKS = [
#     'basketball_in_hoop', 'beat_the_buzz', 'change_channel', 'change_clock', 'close_box',
#     'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid',
#     'close_microwave', 'hang_frame_on_hanger', 'insert_onto_square_peg', 'insert_usb_in_computer',
#     'lamp_off', 'lamp_on', 'lift_numbered_block', 'light_bulb_in', 'meat_off_grill', 'meat_on_grill',
#     'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill',
#     'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base',
#     'pick_and_lift', 'pick_and_lift_small', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack',
#     'place_shape_in_shape_sorter', 'place_wine_at_rack_location', 'play_jenga',
#     'plug_charger_in_power_supply', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf',
#     'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_on_chopping_board', 'put_money_in_safe',
#     'put_rubbish_in_bin', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target',
#     'scoop_with_spatula', 'screw_nail', 'setup_checkers', 'slide_block_to_color_target',
#     'slide_block_to_target', 'slide_cabinet_open_and_place_cups', 'stack_blocks', 'stack_cups',
#     'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'sweep_to_dustpan_of_size', 'take_frame_off_hanger',
#     'take_lid_off_saucepan', 'take_money_out_safe', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box',
#     'take_toilet_roll_off_stand', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer',
#     'toilet_seat_down', 'toilet_seat_up', 'tower3', 'turn_oven_on', 'turn_tap', 'tv_on', 'unplug_charger',
#     'water_plants', 'wipe_desk'
# ]
# TASK_TO_ID = {task: i for i, task in enumerate(ALL_RLBENCH_TASKS)}


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

            pos = obs.gripper_pose[:3]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            criteria = (dist_pos < 5e-3,)

            if all(criteria) or reward == 1:
                break

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip'):
        self._policy = policy
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)
        self.encoder, _ = fetch_text_encoders('clip')
        self.encoder.cuda()

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)
        self._instr = self.encoder(self._instr)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            prediction_len: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        # Predict trajectory
        out = self._policy(
            None,
            torch.full([1, prediction_len, 1], False).to(rgbs.device),
            rgbs,
            None,
            pcds,
            self._instr,
            gripper[:, :, None, :7],  # (1, nhist, nhand=1, 7)
            run_inference=True
        ).view(1, prediction_len, 8)
        return out
        # torch.save(
        #     [
        #         out,
        #         None,
        #         torch.full([1, prediction_len, 1], False).to(rgbs.device),
        #         rgbs,
        #         None,
        #         pcds,
        #         self._instr,
        #         gripper[:, :, None, :7]
        #     ], "trajectory_n.pt"
        # )
        # jnj

    @property
    def device(self):
        return next(self._policy.parameters()).device


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs.perception_data, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs.perception_data, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs.perception_data, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs_(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        obs_rgb = [
            torch.tensor(state_dict["rgb"][i]).float().permute(2, 0, 1) / 255.0
            for i in range(len(state_dict["rgb"]))
        ]
        obs_pc = [
            torch.tensor(state_dict["pc"][i]).float().permute(2, 0, 1)
            if len(state_dict["pc"]) > 0 else None
            for i in range(len(state_dict["rgb"]))
        ]
        state = torch.cat(obs_rgb + obs_pc, dim=0)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        rgb = torch.stack([
            torch.tensor(obs.perception_data["{}_rgb".format(cam)]).float().permute(2, 0, 1) / 255.0
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W
        pcd = torch.stack([
            torch.tensor(obs.perception_data["{}_point_cloud".format(cam)]).float().permute(2, 0, 1)
            for cam in self.apply_cameras
        ]).unsqueeze(0)  # 1, N, C, H, W
        # action is an array of length 16 = (7+1)*2
        gripper = torch.from_numpy(np.concatenate([
            obs.gripper_pose, [obs.gripper_open]
        ])).float().unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # this instance of TaskEnvironment
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        num_history=1
    ):
        device = actioner.device

        success_rate = 0
        total_reward = 0
        var_demos = get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=0
        )

        for demo_id, demo in enumerate(var_demos):

            grippers = torch.Tensor([]).to(device)
            descriptions, obs = task.reset_to_demo(demo)
            actioner.load_episode(descriptions)

            move = Mover(task, max_tries=max_tries)
            reward = 0.0
            max_reward = 0.0

            for step_id in range(max_steps):

                # Fetch the current observation and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.to(device)
                pcds_input = pcd.to(device)
                gripper = gripper.to(device)

                grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                # Prepare proprioception history
                if num_history < 1:
                    gripper_input = grippers[:, -1]
                else:
                    gripper_input = grippers[:, -num_history:]
                    npad = num_history - gripper_input.shape[1]
                    gripper_input = F.pad(
                        gripper_input, (0, 0, npad, 0), mode='replicate'
                    )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len
                )

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output[-1].cpu().numpy()
                    actions[..., -1] = actions[..., -1].round()

                    # execute
                    for action in actions:
                        obs, reward, terminate = move(action, collision_checking=False)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                    if terminate:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

            total_reward += max_reward
            if reward == 0:
                step_id += 1

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", demo_id + 1
            )

        # Compensate for failed demos
        valid = len(var_demos) > 0

        return success_rate, valid, len(var_demos)

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        cameras = {}
        for name in camera_names:
            cameras[name] = used_cams
        print(camera_names)
        # kwargs = {}
        # for n in camera_names:
        #     kwargs[n] = used_cams

        # obs_config = ObservationConfig(
        #     front_camera=kwargs.get("front", unused_cams),
        #     left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
        #     right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
        #     wrist_camera=kwargs.get("wrist", unused_cams),
        #     overhead_camera=kwargs.get("overhead", unused_cams),
        #     joint_forces=False,
        #     joint_positions=False,
        #     joint_velocities=True,
        #     task_low_dim_state=False,
        #     gripper_touch_forces=False,
        #     gripper_pose=True,
        #     gripper_open=True,
        #     gripper_matrix=True,
        #     gripper_joint_positions=True
        # )
        obs_config = ObservationConfig(
            camera_configs=cameras,
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True
        )

        return obs_config
