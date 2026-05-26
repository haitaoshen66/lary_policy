"""Utils for evaluating VLA-Adapter or fine-tuned VLA-Adapter policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import glob

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoProcessor
from latentvla.data_provider.rlds.dataset import NormalizationType
from transformers import Qwen3VLForConditionalGeneration

# Apply JSON numpy patch for serialization
json_numpy.patch()
# ACTION_TOKEN_BEGIN_IDX  = 151386
# STOP_INDEX = 2 
# IGNORE_INDEX = -100
# NUM_ACTIONS_CHUNK = 8
# ACTION_DIM = 7
# ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS_Q99
# NUM_TOKENS = 64

from latentvla.models.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    STOP_INDEX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK
)

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

def load_component_state_dict_v1(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v

    return new_state_dict



def get_vla(cfg: Any):
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
    from latentvla.extern.modeling_internvl_chat import InternVLChatModel
    from latentvla.models import Baseline, LA_Align_VLA, LA_Cond_VLA, LA_Direct_VLA, LA_Tok_VLA
    from collections import OrderedDict
    from latentvla.models.constants import NUM_ACTIONS_CHUNK
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")
    vla_id = cfg.vla_id
    action_head_type = getattr(cfg, "action_head_type", "l1").lower()
    flow_dit_size = getattr(cfg, "flow_dit_size", "dit-b").lower()
    codebook_size = getattr(cfg, "codebook_size", 16)
    latent_tokens_per_step = getattr(cfg, "latent_tokens_per_step", 4)
    if cfg.vlm_model_id == "Qwen3":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.vlm_model_dir, trust_remote_code=True,
            # attn_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(cfg.vlm_model_dir)
        tokenizer = processor

        # add latent tokens
        if vla_id == "la_direct" or vla_id == "la_cond":
            special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(codebook_size)]}
            num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
            # Latent tokens range: 151665 ~ 151680
            print(f"Latent tokens range: {processor.tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][0])} ~ {processor.tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][-1])}, num = {num_added_toks}")
        
        if vla_id == "la_tok":
            special_tokens_dict = {'additional_special_tokens': [f'<LA_{i}>' for i in range(256)]}
            num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Action tokens range: {processor.tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][0])} ~ {processor.tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][-1])}, num = {num_added_toks}")
            print("New vocab size:", len(processor.tokenizer))

    else:
        model = InternVLChatModel.from_pretrained(
            cfg.vlm_model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_model_dir, trust_remote_code=True, use_fast=False)
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        model.config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        model.tokenizer = tokenizer
        model.vision_model.set_num_images_in_input(2)
        processor = None
    if vla_id in {"baseline", "la_align"}:
        prompt_suffix_text = f"Please predict the next {NUM_ACTIONS_CHUNK*5} robot actions: "
    else:
        prompt_suffix_text = f"Please predict the next {NUM_ACTIONS_CHUNK} robot actions: "
    prompt_suffix_token_ids = processor.tokenizer(
        prompt_suffix_text,
        add_special_tokens=False,
    )["input_ids"]

    if vla_id == "la_direct":
        model = LA_Direct_VLA(vlm=model, num_images=cfg.num_images, use_proprio=cfg.use_proprio, action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version, action_head_type=action_head_type, flow_dit_size=flow_dit_size, prompt_suffix_token_ids=prompt_suffix_token_ids)
    elif vla_id == "la_align":
        model = LA_Align_VLA(vlm=model, num_images=cfg.num_images, use_proprio=cfg.use_proprio, action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version, action_head_type=action_head_type, flow_dit_size=flow_dit_size, prompt_suffix_token_ids=prompt_suffix_token_ids)
    elif vla_id == "la_cond":
        model = LA_Cond_VLA(vlm=model, num_images=cfg.num_images, use_proprio=cfg.use_proprio, action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version, action_head_type=action_head_type, flow_dit_size=flow_dit_size, prompt_suffix_token_ids=prompt_suffix_token_ids)
    elif vla_id == "la_tok":
        model = LA_Tok_VLA(vlm=model, num_images=cfg.num_images, use_proprio=cfg.use_proprio, action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version, action_head_type=action_head_type, flow_dit_size=flow_dit_size, prompt_suffix_token_ids=prompt_suffix_token_ids)
    elif vla_id == "baseline":
        model = Baseline(vlm=model, num_images=cfg.num_images, use_proprio=cfg.use_proprio,action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version, action_head_type=action_head_type, flow_dit_size=flow_dit_size, prompt_suffix_token_ids=prompt_suffix_token_ids)
    else:
        raise ValueError(f"Unsupported vla_id: {vla_id}")
    
    # model = InternVL_VLA(vlm=model, use_proprio=cfg.use_proprio, use_pro_version=cfg.use_pro_version)

    # ckpt_path = cfg.pretrained_checkpoint
    
    ckpt_dir = os.path.join(cfg.pretrained_checkpoint, "checkpoints")
    ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, "step-*-epoch-*-loss=*.pt")))
    ckpt_path = ckpt_list[-1]
    state_dict = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    
    print("unexpeacted:", unexpected)
    print("missing:", missing)
    model.eval()
    model.to("cuda")

    # Load dataset stats for action normalization
    _load_dataset_stats(model, cfg.pretrained_checkpoint)
    print(model.norm_stats)

    return model, tokenizer

def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)

def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any], cfg) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    # print(proprio.shape, norm_stats)
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high = np.array(norm_stats["q99"])
        proprio_low  = np.array(norm_stats["q01"])

        if cfg.task_suite_name == "2_bowls" or cfg.task_suite_name == "3_bowls" or cfg.task_suite_name == "4_bowls" or cfg.task_suite_name == "wipe" or cfg.task_suite_name == "bottle" or cfg.task_suite_name == "mango" or cfg.task_suite_name == "sponge":
            # remove gripper open/close dimension
            proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
        else:
            proprio_high = np.delete(proprio_high, 6)
            proprio_low  = np.delete(proprio_low, 6)

            # proprio = np.delete(proprio, 6)
            mask = np.delete(mask, 6)
        
        # proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images

from latentvla.data_provider.data_utils import (
    load_image,
)


def _prepare_labels_for_action_prediction(labels, input_ids):
    """Creates labels tensor for action prediction if not provided"""
    # Extend labels tensor with fake action labels
    ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
    labels_extension = (
        torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
        * ARBITRARY_ACTION_TOKEN_IDX
    )
    labels = torch.cat([labels, labels_extension], dim=-1)

    # Replace last label token with stop token
    labels[:, -1] = STOP_INDEX

    return labels

def _unnormalize_actions(model, normalized_actions, unnorm_key=None):
    """Unnormalize actions using dataset statistics"""
    action_norm_stats = get_action_stats(model, unnorm_key)

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
        normalized_actions,
    )

    return actions

def get_action_stats(vla, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = _check_unnorm_key(vla.norm_stats, unnorm_key)
        return vla.norm_stats[unnorm_key]["action"]

def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
    if unnorm_key is None:
        assert len(norm_stats) == 1, (
            f"Your model was trained on more than one dataset, "
            f"please pass a `unnorm_key` from the following options to choose the statistics "
            f"used for un-normalizing actions: {norm_stats.keys()}"
        )
        unnorm_key = next(iter(norm_stats.keys()))

    assert unnorm_key in norm_stats, (
        f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
        f"please choose from: {norm_stats.keys()}"
    )
    return unnorm_key

def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    tokenizer: Any,
    obs: Dict[str, Any],
    task_label: str,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():
        vla_id = cfg.vla_id
        # Collect all input images
        primary_image = obs["full_image"]
        all_wrist_images = []
        if cfg.num_images > 1:
            all_wrist_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla([primary_image] + all_wrist_images, cfg)
        
        lang = task_label.lower()
        action_token = "🔍"
        if vla_id == "la_direct":
            action_tokens = action_token * (latent_tokens_per_step * NUM_ACTIONS_CHUNK + 1)
        elif vla_id == "la_cond":
            action_tokens = action_token* (5*NUM_ACTIONS_CHUNK+1)
        elif vla_id in {"baseline", "la_align"}:
            action_tokens = action_token* (5*NUM_ACTIONS_CHUNK+1)
        else:
            action_tokens = action_token* (NUM_ACTIONS_CHUNK+1)
        if vla_id in {"baseline", "la_align"}:
            prompt_suffix = f"Please predict the next {NUM_ACTIONS_CHUNK*5} robot actions: {action_tokens}"
        else:
            prompt_suffix = f"Please predict the next {NUM_ACTIONS_CHUNK} robot actions: {action_tokens}"
        
        content = [{"type": "image", "image": img} for img in all_images]
        content.append({"type": "text", "text": lang + prompt_suffix})
        msg = [{"role": "user", "content": content}]
        batch_inputs = tokenizer.apply_chat_template(
            msg,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")
        image_grid_thw = torch.stack([
            batch_inputs["image_grid_thw"].squeeze(0)
        ], dim=0)
        B, N, _ = image_grid_thw.shape
        # flatten 成 [B*N, 3]
        image_grid_thw = image_grid_thw.reshape(B*N, 3)
        vlm_outputs = vla.vlm(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            pixel_values=batch_inputs["pixel_values"],
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats, cfg)
            proprio = obs["state"]
        from latentvla.models.vla.utils import _gather_action_token_embeddings, gather_non_placeholder_hidden_states

        action_head_type = getattr(cfg, "action_head_type", "l1").lower()

        if action_head_type == "flow_gr00t":
            if vla_id in {"baseline", "la_align"}:
                prompt_suffix_text = f"Please predict the next {NUM_ACTIONS_CHUNK*5} robot actions: "
            else:
                prompt_suffix_text = f"Please predict the next {NUM_ACTIONS_CHUNK} robot actions: "
            prompt_suffix_token_ids = tokenizer.tokenizer(
                prompt_suffix_text,
                add_special_tokens=False,
            )["input_ids"]

            vl_embs, vl_mask = gather_non_placeholder_hidden_states(
                last_hidden=vlm_outputs.hidden_states[-1],
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                placeholder_token_id=vla.action_token_id,
                prompt_suffix_token_ids=prompt_suffix_token_ids,
            )

            proprio_tensor = None
            if cfg.use_proprio:
                proprio_tensor = torch.tensor(proprio, dtype=torch.float32, device="cuda").unsqueeze(0)

            normalized_actions = vla.action_head.predict_action(
                vl_embs=vl_embs,
                state=proprio_tensor,
                encoder_attention_mask=vl_mask,
            )
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()

            action = _unnormalize_actions(vla, normalized_actions, cfg.unnorm_key)
            return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]

        num_patches = 512
        multi_layer_hidden_states = []
        for layer_hidden in vlm_outputs.hidden_states[-12:]:
            B, L, H = layer_hidden.shape
            image_hidden = layer_hidden[:, :num_patches]       # [B, P, H]
            text_hidden = layer_hidden
            if vla_id == "la_direct":
                action_hidden = _gather_action_token_embeddings(
                    last_hidden=text_hidden,
                    input_ids=batch_inputs["input_ids"][:, :],  # 对应 text 区间
                    action_token_id=vla.action_token_id,
                    num_chunk=NUM_ACTIONS_CHUNK * latent_tokens_per_step,
                )  # [B, NUM_ACTIONS_CHUNK, H]
                action_hidden = action_hidden.reshape(B, NUM_ACTIONS_CHUNK, latent_tokens_per_step, H)
                action_hidden = vla.pooling(action_hidden)   
            elif vla_id == "la_cond":
                action_hidden = _gather_action_token_embeddings(
                    last_hidden=text_hidden,
                    input_ids=batch_inputs["input_ids"][:, :],  # 对应 text 区间
                    action_token_id=vla.action_token_id,
                    num_chunk=NUM_ACTIONS_CHUNK*4+NUM_ACTIONS_CHUNK,
                )
                # action_hidden = action_hidden[:, -NUM_ACTIONS_CHUNK:, :]
            else:
                action_hidden = _gather_action_token_embeddings(
                    last_hidden=text_hidden,
                    input_ids=batch_inputs["input_ids"][:, :],  # 对应 text 区间
                    action_token_id=vla.action_token_id,
                )  # [B, NUM_ACTIONS_CHUNK, H]

            image_latent = image_hidden.unsqueeze(1)                  # [B, 1, P, H]
            action_latent = action_hidden.unsqueeze(1)                # [B, 1, A, H]

            all_hidden = torch.cat((image_latent, action_latent), dim=2)
            multi_layer_hidden_states.append(all_hidden)
            
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)
        # print(multi_layer_hidden_states.shape)
        proprio = torch.tensor(proprio, dtype=torch.float32, device="cuda").unsqueeze(0)
        
        normalized_actions = vla.action_head.predict_action(multi_layer_hidden_states, proprio=proprio, proprio_projector=vla.proprio_projector)
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.float().cpu().detach().numpy()

        action = _unnormalize_actions(vla, normalized_actions, cfg.unnorm_key)
        
    # Extract subset of actions for open loop steps
    return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()
