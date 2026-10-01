# Copyright (c) 2026 OMRON SINIC X Corporation

"""Offline retrieval metrics (position in mm, rotation as geodesic angle)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_errors(true_euler_zyx: np.ndarray, pred_euler_zyx: np.ndarray) -> np.ndarray:
    """Geodesic angle [rad] between rotations given as ``zyx`` Euler angles, ``[N]``."""
    r_true = Rotation.from_euler("zyx", np.asarray(true_euler_zyx, dtype=np.float64))
    r_pred = Rotation.from_euler("zyx", np.asarray(pred_euler_zyx, dtype=np.float64))
    return (r_pred * r_true.inv()).magnitude()


def action_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """``true`` / ``pred``: ``[N, 6]`` = position cmd [m] (3) + rotation cmd, zyx Euler [rad] (3)."""
    true, pred = np.asarray(true, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    pos_err_mm = (true[:, :3] - pred[:, :3]) * 1000
    rot_err = rotation_errors(true[:, 3:6], pred[:, 3:6])
    pos_mse = float(np.mean(pos_err_mm**2))
    return {
        "num_samples": len(true),
        "overall": {
            "mse": float(np.mean((true - pred) ** 2)),
            "mae": float(np.mean(np.abs(true - pred))),
        },
        "position": {
            "mse": pos_mse,
            "rmse": float(np.sqrt(pos_mse)),
            "mae": float(np.mean(np.abs(pos_err_mm))),
        },
        "rotation": {
            "rmse_rad": float(np.sqrt(np.mean(rot_err**2))),
            "rmse_deg": float(np.rad2deg(np.sqrt(np.mean(rot_err**2)))),
            "mae_rad": float(np.mean(np.abs(rot_err))),
            "mae_deg": float(np.rad2deg(np.mean(np.abs(rot_err)))),
        },
        "position_l2_mm_percentiles": {
            str(p): float(np.percentile(np.linalg.norm(pos_err_mm, axis=1), p))
            for p in (25, 50, 75, 90, 95, 99)
        },
    }
