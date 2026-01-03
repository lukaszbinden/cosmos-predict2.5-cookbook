# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for LeRobotDataset with dVRK (da Vinci Research Kit) configuration.

Usage:
    python test_dvrk_dataset.py
"""

import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    LeRobotDataset,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
    compute_rel_actions,
    rotation_6d_to_matrix,
)

# Dataset path for SUTureBot dVRK dataset
DVRK_DATASET_PATH = "/lustre/fsw/portfolios/healthcareeng/users/nigeln/cache/huggingface/lerobot/jhu_lerobot/suturebot_lerobot"


def compute_diff_actions(qpos, action):
    """
    Computes the relative actions with respect to the current position using axis-angle rotation.

    Parameters:
    - qpos: Current pose (array of shape [8] - xyz, xyzw, jaw angle)
    - action: Actions commanded by the user (array of shape [n_actions x 8] - xyz, xyzw, jaw angle)

    Returns:
    - diff_expand: Relative actions with delta translation and delta rotation in axis-angle format.
                Shape: (n_actions, 7) - [delta_translation, delta_rotation, jaw_angle]
    """
    # Compute the delta translation w.r.t da vinci endoscope tip frame (approx the camera frame)
    delta_translation = action[:, 0:3] - qpos[0:3]  # Shape: (n_actions, 3)

    # Extract quaternions from qpos and action
    quat_init = qpos[3:7]          # Shape: (4,)
    quat_actions = action[:, 3:7]  # Shape: (n_actions, 4)

    # Convert quaternions to Rotation objects
    r_init = Rotation.from_quat(quat_init)
    r_actions = Rotation.from_quat(quat_actions)

    # Compute the relative rotations
    diff_rs = r_init.inv() * r_actions  # Shape: (n_actions,)

    # Convert the rotation differences to rotation vectors (axis-angle representation)
    delta_rotation = diff_rs.as_rotvec()  # Shape: (n_actions, 3)

    # Extract the jaw angle from the action (note: jaw angle is not relative)
    jaw_angle = action[:, -1]  # Shape: (n_actions,)

    # Prepare the final diff array
    delta_action = np.zeros((action.shape[0], 7))  # Shape: (n_actions, 7)

    # Populate the diff_expand array
    delta_action[:, 0:3] = delta_translation       # Delta translation
    delta_action[:, 3:6] = delta_rotation          # Delta rotation (axis-angle)
    delta_action[:, 6] = jaw_angle                 # Jaw angle (not relative)

    return delta_action


def test_lerobot_dataset():
    """Test the LeRobotDataset with dVRK config."""
    print("=" * 60)
    print("Testing LeRobotDataset with dVRK config")
    print("=" * 60)

    # Create dataset with dVRK config
    dataset = LeRobotDataset(
        num_frames=13,  # Number of frames to load
        time_division_factor=4,  # Action chunking factor
        dataset_path=DVRK_DATASET_PATH,
        embodiment="dvrk",
        data_split="train",
        downscaled_res=True,  # Use smaller resolution for testing
    )

    print(f"\nDataset length: {len(dataset)}")
    print(f"Number of underlying datasets: {len(dataset.lerobot_datasets)}")

    # Test loading a few samples
    print("\n" + "-" * 40)
    print("Testing sample loading...")
    print("-" * 40)

    num_test_samples = 3
    for i in range(num_test_samples):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Video shape: {sample['video'].shape}")
        print(f"  Action shape: {sample['action'].shape}")
        print(f"  Action dtype: {sample['action'].dtype}")
        print(f"  Prompt: {sample['prompt'][:50]}..." if sample['prompt'] else "  Prompt: (empty)")

        # Check action values
        if sample['action'].numel() > 0:
            action = sample['action']
            print(f"  Action min: {action.min():.4f}, max: {action.max():.4f}")
            print(f"  Action[0]: {action[0].numpy()}")

    return dataset


def test_iteration():
    """Test iterating over multiple samples."""
    print("\n" + "=" * 60)
    print("Testing dataset iteration")
    print("=" * 60)

    dataset = LeRobotDataset(
        num_frames=13,
        time_division_factor=4,
        dataset_path=DVRK_DATASET_PATH,
        embodiment="dvrk",
        data_split="train",
        downscaled_res=True,
    )

    # Create a small dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Use 0 for easier debugging
    )

    print(f"\nTesting dataloader with batch_size=2...")

    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Video shape: {batch['video'].shape}")
        print(f"  Action shape: {batch['action'].shape}")

        if batch_idx >= 2:  # Just test a few batches
            break

    print("\nIteration test complete!")


def test_rel_actions_equivalence():
    """
    Test that compute_rel_actions produces equivalent results to compute_diff_actions.

    compute_diff_actions: Original formulation (quaternion input, single arm)
    compute_rel_actions: New formulation (6D rotation input, dual arm)

    Both should produce the same relative action representation:
    - Global translation delta
    - Local (tooltip frame) rotation delta
    """
    print("\n" + "=" * 60)
    print("Testing compute_rel_actions vs compute_diff_actions equivalence")
    print("=" * 60)

    # Create synthetic test data with known quaternion values
    # Single arm format: [xyz (3), quat_xyzw (4), gripper (1)] = 8
    np.random.seed(42)
    n_actions = 5

    # Generate random positions
    positions = np.random.randn(n_actions, 3) * 0.1

    # Generate random rotations as quaternions
    rotations = Rotation.random(n_actions)
    quats = rotations.as_quat()  # xyzw format

    # Generate random gripper values
    grippers = np.random.rand(n_actions)

    # Build quaternion-format actions for compute_diff_actions (single arm)
    actions_quat = np.zeros((n_actions, 8))
    actions_quat[:, 0:3] = positions
    actions_quat[:, 3:7] = quats
    actions_quat[:, 7] = grippers

    # Build 6D rotation format actions for compute_rel_actions (dual arm)
    # 6D rotation = first two ROWS of rotation matrix (row-major, matching dVRK format)
    rot_matrices = rotations.as_matrix()  # (n_actions, 3, 3)
    rot_6d = rot_matrices[:, :2, :].reshape(n_actions, 6)  # Take first 2 rows, flatten

    # Dual arm format: [arm1 (10), arm2 (10)] = 20
    # We'll use the same data for both arms to simplify comparison
    actions_6d = np.zeros((n_actions, 20))
    for arm in range(2):
        i = arm * 10
        actions_6d[:, i:i+3] = positions
        actions_6d[:, i+3:i+9] = rot_6d
        actions_6d[:, i+9] = grippers

    print(f"\nTest data shape (quat format): {actions_quat.shape}")
    print(f"Test data shape (6D format): {actions_6d.shape}")

    # Run original compute_diff_actions (single arm)
    qpos = actions_quat[0]  # Base pose
    action = actions_quat[1:]  # Target poses
    diff_original = compute_diff_actions(qpos, action)

    print(f"\ncompute_diff_actions output shape: {diff_original.shape}")
    print(f"  (n_targets={n_actions-1}, 7 per arm)")

    # Run new compute_rel_actions (dual arm)
    rel_new = compute_rel_actions(actions_6d)

    print(f"compute_rel_actions output shape: {rel_new.shape}")
    print(f"  (n_targets={n_actions-1}, 20 for dual arm with 6D rotation)")

    # Compare results for arm 1 (indices 0:10 in rel_new)
    print("\n" + "-" * 40)
    print("Comparing outputs (arm 1):")
    print("-" * 40)

    # Translation delta
    trans_diff_orig = diff_original[:, 0:3]
    trans_diff_new = rel_new[:, 0:3]
    trans_error = np.abs(trans_diff_orig - trans_diff_new).max()
    print(f"\nTranslation delta max error: {trans_error:.10f}")

    # Rotation delta - convert 6D back to rotvec for comparison
    rot_diff_orig = diff_original[:, 3:6]  # 3D rotvec
    rot_6d_new = rel_new[:, 3:9]  # 6D rotation
    # Convert 6D to rotation matrix, then to rotvec
    rot_matrix_new = rotation_6d_to_matrix(rot_6d_new)  # [n_targets, 3, 3]
    rot_diff_new = Rotation.from_matrix(rot_matrix_new).as_rotvec()  # [n_targets, 3]
    rot_error = np.abs(rot_diff_orig - rot_diff_new).max()
    print(f"Rotation delta max error: {rot_error:.10f}")

    # Gripper (now at index 9 for arm 1)
    grip_orig = diff_original[:, 6]
    grip_new = rel_new[:, 9]
    grip_error = np.abs(grip_orig - grip_new).max()
    print(f"Gripper max error: {grip_error:.10f}")

    # Print sample values
    print("\n" + "-" * 40)
    print("Sample values (first relative action):")
    print("-" * 40)
    print(f"Original (compute_diff_actions):")
    print(f"  delta_xyz: {diff_original[0, 0:3]}")
    print(f"  delta_rot (rotvec): {diff_original[0, 3:6]}")
    print(f"  gripper:   {diff_original[0, 6]}")
    print(f"\nNew (compute_rel_actions, arm 1):")
    print(f"  delta_xyz: {rel_new[0, 0:3]}")
    print(f"  delta_rot (6D): {rel_new[0, 3:9]}")
    print(f"  delta_rot (converted to rotvec): {rot_diff_new[0]}")
    print(f"  gripper:   {rel_new[0, 9]}")

    # Assert equivalence
    assert trans_error < 1e-6, f"Translation mismatch: {trans_error}"
    assert rot_error < 1e-6, f"Rotation mismatch: {rot_error}"
    assert grip_error < 1e-6, f"Gripper mismatch: {grip_error}"

    print("\n✓ compute_rel_actions produces equivalent results to compute_diff_actions!")


if __name__ == "__main__":
    print("dVRK Dataset Test Script")
    print("Dataset path:", DVRK_DATASET_PATH)

    test_rel_actions_equivalence()
    test_lerobot_dataset()
    test_iteration()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
