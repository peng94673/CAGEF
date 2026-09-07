from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

TensorList = List[torch.Tensor]

def seed_everything(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        # 保证同一硬件和软件环境中重复实验尽可能一致。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    """解析计算设备；CUDA 不可用时安全回退到 CPU。"""
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[警告] 请求了 CUDA，但当前环境不可用，已自动改用 CPU。")
        return torch.device("cpu")
    if (
        requested.type == "cuda"
        and requested.index is not None
        and requested.index >= torch.cuda.device_count()
    ):
        raise ValueError(
            f"请求的设备 cuda:{requested.index} 不存在；"
            f"当前仅检测到 {torch.cuda.device_count()} 块 CUDA 设备"
        )
    return requested


def choose_attention_heads(hidden_dim: int, preferred: int = 4) -> int:
    """选择可整除 hidden_dim 的最大注意力头数。"""
    for heads in range(min(preferred, hidden_dim), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


def safe_float(value: float) -> Optional[float]:
    """把非有限浮点数转换为 None，避免生成非标准 JSON。"""
    value = float(value)
    return value if math.isfinite(value) else None


def stable_l2_normalize(
    tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """将近零向量映射为零，并阻断其不稳定的归一化梯度。"""
    norm = torch.linalg.vector_norm(tensor, ord=2, dim=dim, keepdim=True)
    normalized = F.normalize(tensor, p=2, dim=dim, eps=eps)
    return torch.where(norm > eps, normalized, torch.zeros_like(normalized))


def safe_cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """使用 float32 计算余弦相似度，避免 AMP 下零向量除零。"""
    if first.shape != second.shape:
        raise ValueError(
            "余弦相似度的两个输入形状必须一致："
            f"{tuple(first.shape)} vs {tuple(second.shape)}"
        )
    compute_first = (
        first.float()
        if first.dtype in (torch.float16, torch.bfloat16)
        else first
    )
    compute_second = (
        second.float()
        if second.dtype in (torch.float16, torch.bfloat16)
        else second
    )
    # 零向量没有可定义的方向。显式令其归一化结果及梯度为 0，避免
    # 1/eps 量级的梯度回传到 FP16 时溢出。
    normalized_first = stable_l2_normalize(compute_first, dim=dim, eps=eps)
    normalized_second = stable_l2_normalize(compute_second, dim=dim, eps=eps)
    return (normalized_first * normalized_second).sum(dim=dim)


def validate_positive_int_list(values: Sequence[int], name: str) -> List[int]:
    result = [int(x) for x in values]
    if not result or any(x <= 0 for x in result):
        raise ValueError(f"{name} 必须至少包含一个正整数，当前值为 {result}")
    return result


def validate_modality_tensors(
    data_list: Sequence[torch.Tensor],
    input_name: str,
    expected_views: Optional[int] = None,
    expected_dims: Optional[Sequence[int]] = None,
) -> int:
    """统一校验多模态张量，并返回批大小。"""
    if not data_list:
        raise ValueError(f"{input_name}模态列表不能为空")
    if expected_views is not None and len(data_list) != expected_views:
        raise ValueError(
            f"期望 {expected_views} 个{input_name}模态，实际为 {len(data_list)}"
        )
    if expected_dims is not None and len(expected_dims) != len(data_list):
        raise ValueError("expected_dims 数量必须与模态数量一致")

    if data_list[0].ndim != 2:
        raise ValueError(
            f"{input_name}模态 0 必须是二维张量 [B,F]，"
            f"当前为 {tuple(data_list[0].shape)}"
        )
    batch_size = data_list[0].shape[0]
    for view, data in enumerate(data_list):
        if data.ndim != 2:
            raise ValueError(
                f"{input_name}模态 {view} 必须是二维张量 [B,F]，"
                f"当前为 {tuple(data.shape)}"
            )
        if data.shape[0] != batch_size:
            raise ValueError(f"所有{input_name}模态的批大小必须一致")
        if expected_dims is not None and data.shape[1] != expected_dims[view]:
            raise ValueError(
                f"{input_name}模态 {view} 的形状应为 "
                f"[B,{expected_dims[view]}]，当前为 {tuple(data.shape)}"
            )
    return batch_size


def validate_sample_mask(
    sample_mask: torch.Tensor,
    batch_size: int,
    n_views: int,
    require_available: bool = True,
) -> None:
    """统一校验样本级二值模态可用性掩码。"""
    expected_shape = (batch_size, n_views)
    if sample_mask.shape != expected_shape:
        raise ValueError(
            f"sample_mask 形状必须为 {expected_shape}，当前为 "
            f"{tuple(sample_mask.shape)}"
        )
    if not torch.isfinite(sample_mask).all():
        raise ValueError("sample_mask 包含 NaN 或 Inf")
    if torch.any((sample_mask != 0) & (sample_mask != 1)):
        raise ValueError("sample_mask 必须是只包含 0 和 1 的二值掩码")
    if require_available and torch.any(sample_mask.sum(dim=1) == 0):
        raise ValueError("每个样本至少需要一个可用模态")
