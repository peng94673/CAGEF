from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from trainer import CAGEFFiveFoldTrainer
from utils import safe_float, seed_everything, validate_positive_int_list

def parse_int_list(value: str, name: str) -> List[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} 必须是逗号分隔的整数列表"
        ) from error
    try:
        return validate_positive_int_list(parsed, name)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CAGEF 图神经网络增强多模态分类："
            "动态模态数量与分层五折交叉验证"
        )
    )
    parser.add_argument("--data_folder", type=str, default="BRCA")
    parser.add_argument(
        "--modalities",
        type=str,
        default="",
        help=(
            "可选的逗号分隔模态前缀，例如 1,2 或 mRNA,methylation；"
            "留空时自动发现全部成对的 *_tr.csv/*_te.csv 文件"
        ),
    )
    parser.add_argument(
        "--missing_rate",
        type=float,
        default=0.1,
        help="不完整样本占比，范围 [0,1]",
    )
    parser.add_argument("--exp", type=str, default="./exp")
    parser.add_argument("--augmentation_rate", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.01)

    parser.add_argument("--hidden_dim", type=str, default="128")
    parser.add_argument("--prediction", type=str, default="64,32")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--graph_dropout",
        type=float,
        default=0.1,
        help="GAT 与跨模态多头注意力的 dropout 率",
    )
    parser.add_argument("--use_gnn", type=int, choices=[0, 1], default=1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--use_amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--lambda_impute", type=float, default=0.1)
    parser.add_argument("--lambda_consist", type=float, default=0.1)
    parser.add_argument("--lambda_graph", type=float, default=0.05)

    parser.add_argument("--num_epoch", type=int, default=2000)
    parser.add_argument(
        "--test_interval",
        "--test_inverval",
        dest="test_interval",
        type=int,
        default=50,
        help="验证间隔；同时兼容旧参数拼写 --test_inverval",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="连续多少次验证无提升后早停；0 表示关闭",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["auto", "accuracy"],
        default="auto",
        help="最佳模型选择指标；auto=二分类AUC/多分类宏平均F1",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-4,
        help="验证指标至少提升多少才重置早停计数",
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=58,
        help="第一次完整实验使用的基础随机种子",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="使用不同基础随机种子重复完整实验的次数",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=100,
        help="相邻两次完整实验的基础随机种子间隔",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def validate_arguments(params: Dict) -> None:
    if not 0.0 <= params["missing_rate"] <= 1.0:
        raise ValueError("missing_rate 必须位于 [0,1]")
    if not 0.0 <= params["augmentation_rate"] <= 1.0:
        raise ValueError("augmentation_rate 必须位于 [0,1]")
    if params["noise_std"] < 0:
        raise ValueError("noise_std 不能为负数")
    if not 0.0 <= params["dropout"] < 1.0:
        raise ValueError("dropout 必须位于 [0,1)")
    if not 0.0 <= params["graph_dropout"] < 1.0:
        raise ValueError("graph_dropout 必须位于 [0,1)")
    if params["lr"] <= 0 or params["weight_decay"] < 0:
        raise ValueError("lr 必须大于 0，weight_decay 不能为负数")
    if params["step_size"] <= 0:
        raise ValueError("step_size 必须大于 0")
    if not 0.0 < params["lr_gamma"] <= 1.0:
        raise ValueError("lr_gamma 必须位于 (0,1]")
    if params["num_epoch"] <= 0 or params["test_interval"] <= 0:
        raise ValueError("num_epoch 和 test_interval 必须大于 0")
    if params["patience"] < 0:
        raise ValueError("patience 不能为负数")
    if params["min_delta"] < 0:
        raise ValueError("min_delta 不能为负数")
    if params["num_folds"] < 2:
        raise ValueError("num_folds 必须至少为 2")
    if params["seed"] < 0:
        raise ValueError("seed 不能为负数")
    if params["num_runs"] <= 0:
        raise ValueError("num_runs 必须大于 0")
    if params["seed_step"] <= 0:
        raise ValueError("seed_step 必须大于 0")
    for key in ("lambda_impute", "lambda_consist", "lambda_graph"):
        if params[key] < 0:
            raise ValueError(f"{key} 不能为负数")


def run_repeated_experiments(params: Dict) -> str:
    """重复执行完整五折实验并汇总预留测试集集成指标。"""
    num_runs = int(params["num_runs"])
    first_seed = int(params["seed"])
    seed_step = int(params["seed_step"])
    run_seeds = [
        first_seed + run_index * seed_step
        for run_index in range(num_runs)
    ]

    dataset_name = Path(params["data_folder"]).resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    repeated_path = (
        Path(params["exp"])
        / (
            f"{dataset_name}_Repeated{num_runs}_"
            f"CV{int(params['num_folds'])}_{timestamp}"
        )
    )
    runs_path = repeated_path / "runs"
    runs_path.mkdir(parents=True, exist_ok=True)

    repeated_config = dict(params)
    repeated_config["run_seeds"] = run_seeds
    with (repeated_path / "repeated_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(repeated_config, file, indent=2, ensure_ascii=False)

    metric_names: Optional[List[str]] = None
    test_metrics_by_run: List[np.ndarray] = []
    run_records: List[Dict[str, object]] = []

    print("\n" + "#" * 76)
    print(f"开始 {num_runs} 次独立重复实验")
    print(f"基础随机种子：{', '.join(str(seed) for seed in run_seeds)}")
    print("#" * 76)

    for run_index, base_seed in enumerate(run_seeds, start=1):
        print("\n" + "#" * 76)
        print(
            f"独立实验 {run_index}/{num_runs}："
            f"base_seed={base_seed}"
        )
        print("#" * 76)

        run_params = dict(params)
        run_params["seed"] = base_seed
        run_params["exp"] = str(runs_path)
        run_params["run_index"] = run_index
        seed_everything(base_seed)

        trainer = CAGEFFiveFoldTrainer(run_params)
        result = trainer.train()
        current_metric_names = list(result["metric_names"])
        current_test_metrics = np.asarray(
            result["test_ensemble_metrics"], dtype=np.float64
        )

        if metric_names is None:
            metric_names = current_metric_names
        elif current_metric_names != metric_names:
            raise RuntimeError(
                "不同重复实验返回的评价指标不一致："
                f"{metric_names} vs {current_metric_names}"
            )

        test_metrics_by_run.append(current_test_metrics)
        run_records.append(
            {
                "run": run_index,
                "base_seed": base_seed,
                "metrics": [safe_float(x) for x in current_test_metrics],
                "experiment_path": str(result["experiment_path"]),
            }
        )

    if metric_names is None or not test_metrics_by_run:
        raise RuntimeError("没有获得任何重复实验结果")

    test_array = np.stack(test_metrics_by_run, axis=0)
    test_means = np.nanmean(test_array, axis=0)
    std_ddof = 1 if num_runs > 1 else 0
    test_stds = np.nanstd(test_array, axis=0, ddof=std_ddof)

    csv_path = repeated_path / "repeated_test_results.csv"
    fieldnames = ["run", "base_seed", *metric_names, "experiment_path"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in run_records:
            row = {
                "run": record["run"],
                "base_seed": record["base_seed"],
                "experiment_path": record["experiment_path"],
            }
            row.update(
                {
                    name: f"{float(value):.10f}"
                    for name, value in zip(
                        metric_names, record["metrics"]
                    )
                    if value is not None
                }
            )
            writer.writerow(row)

        mean_row = {
            "run": "mean",
            "base_seed": "",
            "experiment_path": "",
        }
        mean_row.update(
            {
                name: f"{float(value):.10f}"
                for name, value in zip(metric_names, test_means)
            }
        )
        writer.writerow(mean_row)

        std_row = {
            "run": "std",
            "base_seed": "",
            "experiment_path": "",
        }
        std_row.update(
            {
                name: f"{float(value):.10f}"
                for name, value in zip(metric_names, test_stds)
            }
        )
        writer.writerow(std_row)

    repeated_summary = {
        "dataset": dataset_name,
        "num_runs": num_runs,
        "num_folds": int(params["num_folds"]),
        "run_seeds": run_seeds,
        "metric_names": metric_names,
        "heldout_test_mean": [safe_float(x) for x in test_means],
        "heldout_test_std": [safe_float(x) for x in test_stds],
        "runs": run_records,
    }
    with (repeated_path / "repeated_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(repeated_summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print(f"{num_runs} 次独立实验的预留测试集汇总")
    print("=" * 76)
    for name, mean, std in zip(metric_names, test_means, test_stds):
        print(f"{name}: {mean:.5f} +/- {std:.5f}")
    print(f"\n明细与汇总表：{csv_path}")
    print(f"完整重复实验目录：{repeated_path}")
    return str(repeated_path)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    params = vars(args)
    params["modalities"] = [
        item.strip()
        for item in params["modalities"].split(",")
        if item.strip()
    ] or None
    params["hidden_dim"] = parse_int_list(params["hidden_dim"], "hidden_dim")
    params["prediction"] = parse_int_list(
        params["prediction"], "prediction"
    )
    params["use_gnn"] = bool(params["use_gnn"])
    params["use_amp"] = bool(params["use_amp"])
    validate_arguments(params)
    run_seeds = [
        int(params["seed"]) + run_index * int(params["seed_step"])
        for run_index in range(int(params["num_runs"]))
    ]
    seed_everything(run_seeds[0])

    print("训练配置：")
    print(f"  数据目录：{params['data_folder']}")
    print(
        "  模态选择："
        + (
            ", ".join(params["modalities"])
            if params["modalities"]
            else "自动发现"
        )
    )
    print(f"  不完整样本比例：{params['missing_rate']}")
    print(f"  交叉验证折数：{params['num_folds']}")
    print(f"  独立实验次数：{params['num_runs']}")
    print(
        "  基础随机种子："
        + ", ".join(str(seed) for seed in run_seeds)
    )
    print(f"  使用 GNN：{params['use_gnn']}")
    print(f"  使用 AMP：{params['use_amp']}")
    print(f"  隐层维度：{params['hidden_dim']}")
    print(
        f"  全连接 Dropout：{params['dropout']}；"
        f"图注意力 Dropout：{params['graph_dropout']}"
    )
    print(f"  训练轮数：{params['num_epoch']}")
    print(
        f"  学习率：{params['lr']}；每 {params['step_size']} 轮乘以 "
        f"{params['lr_gamma']}"
    )
    print(
        f"  每 {params['test_interval']} 轮验证；连续 "
        f"{params['patience']} 次无提升早停"
    )
    print(
        f"  模型选择指标：{params['selection_metric']}；"
        f"最小有效提升：{params['min_delta']}"
    )

    run_repeated_experiments(params)

if __name__ == "__main__":
    main()

