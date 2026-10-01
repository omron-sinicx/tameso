---
task_categories:
- robotics
tags:
- LeRobot
- tactile
- peg-in-hole
- soft-robot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# Contactile peg-in-hole demonstrations (TaMeSo)

Teleoperated peg-in-hole demonstrations with a soft-wrist robot and a 3×3 Contactile (PapillArray)
tactile sensor, used to train MAT³ (Masked Tactile Trajectory Transformer) in
[Tactile Memory With Soft Robot: Robust Object Insertion via Masked Encoding and Soft Wrist](https://doi.org/10.1109/LRA.2026.3692097) (IEEE RA-L 2026).

* 64 episodes, 19,888 frames, 50 Hz
* Robot: UR5e with a soft wrist, integrated F/T sensor, Robotiq Hand-E, HTC VIVE tracker on the gripper

## Layout

| revision | format | notes |
|---|---|---|
| `main` (tag `v3.0`) | LeRobot v3.0 (`meta/`, `data/`) | used by the data viewer, recent LeRobot and `mat3` |
| branch `v1.6` | LeRobot v1.6 (`meta_data/`, `train/`, `videos/`, `episodes/`, `episodes_original/`) | original recording, kept for the original research code |

The v3.0 files are converted from the v1.6 files by `mat3.data.convert`: frame order is kept, episodes
are renumbered in storage order (original ids: `meta/source.json`), and `meta/stats.json` holds the
original normalisation statistics. The camera videos are only in the `v1.6` branch.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("omron-sinicx/contactile_200_lota")
```

## Features

| key | dim | description |
|---|---|---|
| `observation.contactile` | 33 | 11 pillars × (x, y, z) force; the first 9 form the 3×3 taxel grid |
| `observation.ft` | 6 | force / torque |
| `observation.eef.position`, `observation.eef.rotation_ortho6` | 3 + 6 | end-effector pose |
| `observation.eef.rotation_axis_angle` | 3 | end-effector rotation (axis-angle) |
| `observation.vive_tracker_pose` | 7 | gripper pose from the motion tracker (xyz + quaternion) |
| `observation.qpos`, `observation.qvel` | 6 + 6 | joint positions / velocities |
| `action.position_cmd`, `action.rotation_cmd` | 3 + 3 | commanded position / rotation (zyx Euler) |
| `action.position`, `action.rotation_ortho6` | 3 + 6 | target pose |
| `action.gripper` | 1 | gripper command |
| `action.stiffness_diag.trans`, `action.stiffness_diag.rot` | 3 + 3 | stiffness |

## Citation

```bibtex
@article{kamijo2026tactile,
    title   = {Tactile Memory With Soft Robot: Robust Object Insertion via Masked Encoding and Soft Wrist},
    author  = {Kamijo, Tatsuya and Nishimura, Mai and Shibasaki, Nodoka and Siburian, Jeremy and Beltran-Hernandez, Cristian C. and Hamaya, Masashi},
    journal = {IEEE Robotics and Automation Letters},
    volume  = {11},
    number  = {7},
    pages   = {7844--7851},
    year    = {2026},
    doi     = {10.1109/LRA.2026.3692097},
}
```
