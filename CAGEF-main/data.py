from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils import (
    TensorList,
    validate_modality_tensors,
    validate_sample_mask,
)

def _load_csv_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    matrix = np.loadtxt(path, delimiter=",", ndmin=2)
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"数据文件不是数值矩阵：{path}")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"数据文件包含 NaN 或 Inf：{path}")
    return matrix


def _load_csv_labels(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"找不到标签文件：{path}")
    labels_raw = np.atleast_1d(np.loadtxt(path, delimiter=","))
    if labels_raw.ndim != 1:
        labels_raw = labels_raw.reshape(-1)
    if not np.all(np.isfinite(labels_raw)):
        raise ValueError(f"标签文件包含 NaN 或 Inf：{path}")
    if not np.allclose(labels_raw, np.round(labels_raw)):
        raise ValueError(f"标签必须是整数：{path}")
    return np.round(labels_raw).astype(np.int64)


def _natural_sort_key(value: str) -> List[object]:
    """按自然顺序排序模态名称，例如 2 排在 10 之前。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def discover_modality_names(
    folder: Path,
    requested_modalities: Optional[Sequence[str]] = None,
) -> List[str]:
    """发现同时具有训练文件和测试文件的模态前缀。"""
    if isinstance(requested_modalities, str):
        requested_modalities = [
            item.strip()
            for item in requested_modalities.split(",")
            if item.strip()
        ]

    train_names = {
        path.name[: -len("_tr.csv")]
        for path in folder.glob("*_tr.csv")
        if path.name != "labels_tr.csv"
    }
    test_names = {
        path.name[: -len("_te.csv")]
        for path in folder.glob("*_te.csv")
        if path.name != "labels_te.csv"
    }

    if requested_modalities:
        names = [
            str(name).strip()
            for name in requested_modalities
            if str(name).strip()
        ]
        if len(set(names)) != len(names):
            raise ValueError(f"--modalities 包含重复名称：{names}")
        missing_train = [name for name in names if name not in train_names]
        missing_test = [name for name in names if name not in test_names]
        if missing_train or missing_test:
            raise FileNotFoundError(
                "指定模态缺少配对数据文件；"
                f"缺少训练文件={missing_train}，缺少测试文件={missing_test}"
            )
    else:
        unmatched_train = sorted(
            train_names - test_names, key=_natural_sort_key
        )
        unmatched_test = sorted(
            test_names - train_names, key=_natural_sort_key
        )
        if unmatched_train or unmatched_test:
            raise FileNotFoundError(
                "模态训练/测试文件不成对；"
                f"仅有训练文件={unmatched_train}，仅有测试文件={unmatched_test}"
            )
        names = sorted(train_names & test_names, key=_natural_sort_key)

    if len(names) < 2:
        raise ValueError(
            "CAGEF 至少需要两个模态；应提供至少两组 "
            "<modality>_tr.csv 和 <modality>_te.csv 文件"
        )
    return names


def load_raw_data(
    data_folder: str,
    requested_modalities: Optional[Sequence[str]] = None,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """加载任意数量的模态数据，并进行完整的一致性检查。"""
    folder = Path(data_folder).expanduser()
    if not folder.is_dir():
        raise NotADirectoryError(f"数据目录不存在：{folder}")

    labels_train = _load_csv_labels(folder / "labels_tr.csv")
    labels_test = _load_csv_labels(folder / "labels_te.csv")
    modality_names = discover_modality_names(folder, requested_modalities)

    train_views: List[np.ndarray] = []
    test_views: List[np.ndarray] = []
    for modality_name in modality_names:
        train_views.append(
            _load_csv_matrix(folder / f"{modality_name}_tr.csv")
        )
        test_views.append(
            _load_csv_matrix(folder / f"{modality_name}_te.csv")
        )

    n_train = labels_train.size
    n_test = labels_test.size
    for modality_name, train_x, test_x in zip(
        modality_names, train_views, test_views
    ):
        if train_x.shape[0] != n_train:
            raise ValueError(
                f"模态 {modality_name} 训练样本数 {train_x.shape[0]} "
                f"与训练标签数 {n_train} 不一致"
            )
        if test_x.shape[0] != n_test:
            raise ValueError(
                f"模态 {modality_name} 测试样本数 {test_x.shape[0]} "
                f"与测试标签数 {n_test} 不一致"
            )
        if train_x.shape[1] != test_x.shape[1]:
            raise ValueError(
                f"模态 {modality_name} 的训练/测试特征维数不一致："
                f"{train_x.shape[1]} 与 {test_x.shape[1]}"
            )
        if train_x.shape[1] == 0:
            raise ValueError(f"模态 {modality_name} 没有任何特征")

    return (
        train_views,
        test_views,
        labels_train,
        labels_test,
        modality_names,
    )


def make_sample_level_mask(
    n_samples: int,
    n_views: int,
    missing_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成样本级缺失掩码。

    missing_rate 表示“不完整样本占比”，不是全矩阵中缺失元素的精确占比。
    每个不完整样本至少保留一个模态、至多保留 n_views - 1 个模态。
    不完整样本内部的平均保留比例尽量接近 1 - missing_rate。
    """
    if n_samples <= 0:
        raise ValueError("n_samples 必须大于 0")
    if n_views < 2:
        raise ValueError("n_views 必须至少为 2")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate 必须位于 [0, 1] 区间")

    mask = np.ones((n_samples, n_views), dtype=np.float32)
    n_incomplete = int(round(n_samples * missing_rate))
    if n_incomplete == 0:
        return mask

    incomplete_rows = rng.choice(n_samples, size=n_incomplete, replace=False)
    incomplete_mask = np.zeros((n_incomplete, n_views), dtype=np.float32)

    # 每个不完整样本先随机保留一个模态。
    first_views = rng.integers(0, n_views, size=n_incomplete)
    incomplete_mask[np.arange(n_incomplete), first_views] = 1.0

    target_observed = int(
        round(n_incomplete * n_views * (1.0 - missing_rate))
    )
    target_observed = int(
        np.clip(
            target_observed,
            n_incomplete,
            n_incomplete * (n_views - 1),
        )
    )
    n_extra = target_observed - n_incomplete

    if n_extra > 0:
        # 每行最多再增加 n_views-2 个模态，使所有被选中的样本始终至少
        # 缺失一个模态。原实现从全部空位置直接抽样，可能把某些行重新
        # 补成完整样本，导致实际不完整样本比例小于 missing_rate。
        row_capacity = n_views - 2
        row_slots = np.repeat(np.arange(n_incomplete), row_capacity)
        selected_slots = rng.choice(
            row_slots.size, size=n_extra, replace=False
        )
        extra_per_row = np.bincount(
            row_slots[selected_slots], minlength=n_incomplete
        )
        for row, extra_count in enumerate(extra_per_row):
            if extra_count == 0:
                continue
            absent_views = np.flatnonzero(incomplete_mask[row] == 0)
            selected_views = rng.choice(
                absent_views, size=int(extra_count), replace=False
            )
            incomplete_mask[row, selected_views] = 1.0

    mask[incomplete_rows] = incomplete_mask
    return mask


def data_augmentation(
    data_list: TensorList,
    sample_mask: torch.Tensor,
    augmentation_rate: float = 0.1,
    noise_std: float = 0.01,
    feature_masks: Optional[TensorList] = None,
) -> TensorList:
    """仅对真实存在的模态添加高斯噪声，绝不把缺失模态意外填成非零值。"""
    if not 0.0 <= augmentation_rate <= 1.0:
        raise ValueError("augmentation_rate 必须位于 [0,1]")
    if noise_std < 0:
        raise ValueError("noise_std 不能为负数")
    batch_size = validate_modality_tensors(data_list, "增强输入")
    validate_sample_mask(sample_mask, batch_size, len(data_list))
    if augmentation_rate <= 0:
        return data_list
    if feature_masks is not None and len(feature_masks) != len(data_list):
        raise ValueError("feature_masks 数量必须与模态数量一致")

    augmented: TensorList = []
    for view, data in enumerate(data_list):
        available = sample_mask[:, view] > 0.5
        selected = (
            torch.rand(data.shape[0], device=data.device) < augmentation_rate
        ) & available
        if selected.any():
            output = data.clone()
            noise = torch.randn_like(output[selected]) * noise_std
            if feature_masks is not None:
                noise = noise * feature_masks[view].to(noise.dtype)
            output[selected] = (output[selected] + noise).clamp(0.0, 1.0)
            if feature_masks is not None:
                output = output * feature_masks[view].to(output.dtype)
            augmented.append(output)
        else:
            augmented.append(data)
    return augmented

