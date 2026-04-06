import hashlib
import json

import numpy as np
import torch
import torchvision.transforms.functional as IF
from PIL import Image
from transformers import AutoModel, AutoProcessor


class XiaomiModel:
    def __init__(
        self,
        model_path,
        split="abc",
        crop_ratio=0.95,
        action_chunk=10,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.model_dtype = torch.bfloat16
        self.crop_ratio = float(crop_ratio)
        self.action_chunk = int(action_chunk)
        self.robot_type = f"calvin_{split}_orig"
        # Default to Xiaomi-style stepping semantics in calvin_env:
        # bare length-7 ndarray -> robot.relative_to_absolute(...)
        self.calvin_action_exec = "relative_array"

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            dtype=self.model_dtype,
        ).to(self.device, dtype=self.model_dtype)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

    def reset(self):
        # Xiaomi inference is stateless across rollout steps.
        return None

    def preprocess_from_calvin_batch(self, sample: dict, frame_idx: int = 0):
        """
        Build MiBoT observation dict from a CALVINDataset __getitem__ result.
        See online_evaluation_calvin.calvin_dataloader_mibot.calvin_sample_to_mibot_obs.
        """
        from online_evaluation_calvin.calvin_dataloader_mibot import calvin_sample_to_mibot_obs

        return calvin_sample_to_mibot_obs(
            sample, frame_idx=frame_idx, crop_ratio=self.crop_ratio
        )

    @torch.no_grad()
    def preprocess(self, obs, env):
        del env
        rgb_static = self._to_pil(obs["rgb_obs"]["rgb_static"])
        rgb_gripper = self._to_pil(obs["rgb_obs"]["rgb_gripper"])
        rgb_static, rgb_gripper = self._center_crop_resize(
            [rgb_static, rgb_gripper], self.crop_ratio
        )

        start_state = obs["robot_obs"][:7]
        start_state = np.concatenate([start_state, np.zeros([25])], axis=-1)
        return {"base": rgb_static, "wrist_left": rgb_gripper, "state": start_state}

    @torch.no_grad()
    def step(self, obs, instruction):
        language = instruction.capitalize()
        seed_payload = {
            "task_id": self.robot_type,
            "state": obs["state"],
            "base": obs["base"],
            "wrist_left": obs["wrist_left"],
            "language": language,
        }
        seed = self._hash_data_to_seed(seed_payload)

        language_with_punc = f"{language}."
        prompt = (
            "<|im_start|>user\n"
            "The following observations are captured from multiple views.\n"
            "# Base View\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "# Left-Wrist View\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            "Generate robot actions for the task:\n"
            f"{language_with_punc} /no_cot<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<cot></cot><|im_end|>\n"
        )

        vl_inputs = self.processor(
            text=[prompt],
            images=[obs["base"], obs["wrist_left"]],
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        model_inputs = {
            **vl_inputs,
            "state": torch.from_numpy(obs["state"]).to(
                self.device, dtype=self.model_dtype
            ).view(1, 1, -1),
            "action_mask": self.processor.get_action_mask(self.robot_type).to(
                self.device, dtype=self.model_dtype
            ),
            "seed": seed,
        }
        outputs = self.model(**model_inputs)
        action = self.processor.decode_action(outputs.actions, robot_type=self.robot_type)

        action = action[0, 0 : self.action_chunk, :7].detach().cpu().numpy()
        action[:, -1:] = np.where(action[:, -1:] > 0, 1, -1)
        return action[None]

    @staticmethod
    def _to_pil(frame):
        if isinstance(frame, Image.Image):
            return frame
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        arr = np.asarray(frame)

        # Drop leading batch dims: (1, H, W, C) -> (H, W, C)
        while arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # (1, H, W) grayscale
        while arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] not in (1, 3, 4):
            arr = arr[0]

        # CHW -> HWC (common if someone passes tensor-style layout)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            pass  # grayscale H x W
        elif arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"Expected image array HxW or HxWxC; got shape={arr.shape}, dtype={arr.dtype}"
            )

        # PIL does not accept int64/float64 directly for RGB
        if np.issubdtype(arr.dtype, np.floating):
            if float(np.nanmax(arr)) <= 1.0 + 1e-5:
                arr = (arr * 255.0).clip(0, 255)
            else:
                arr = arr.clip(0, 255)
            arr = arr.astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)

    @staticmethod
    def _center_crop_resize(images, crop_ratio):
        def _crop(image):
            height, width = image.size[1], image.size[0]
            crop_h, crop_w = int(height * crop_ratio), int(width * crop_ratio)
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2
            cropped = IF.crop(image, top=top, left=left, height=crop_h, width=crop_w)
            return IF.resize(cropped, size=(height, width))

        return [_crop(image) for image in images]

    @staticmethod
    def _hash_data_to_seed(data, max_bytes=4):
        def custom_encoder(obj):
            if isinstance(obj, np.ndarray):
                return {
                    "__type__": "numpy",
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                    "data": obj.tobytes().hex(),
                }
            if isinstance(obj, Image.Image):
                img_hash = hashlib.md5(obj.tobytes()).hexdigest()
                return {
                    "__type__": "PIL.Image",
                    "mode": obj.mode,
                    "size": obj.size,
                    "content_hash": img_hash,
                }
            if isinstance(obj, set):
                return sorted(list(obj))
            raise TypeError(f"Type {type(obj)} is not JSON serializable")

        json_str = json.dumps(
            data,
            default=custom_encoder,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        hex_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        seed_int = int(hex_hash, 16)
        if max_bytes > 0:
            seed_int = seed_int % (2 ** (8 * max_bytes))
        return seed_int
