from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data import (
    data_augmentation,
    load_raw_data,
    make_sample_level_mask,
)
from model import GNNEhancedCAGEF
from utils import (
    TensorList,
    resolve_device,
    safe_float,
    seed_everything,
    validate_positive_int_list,
)

class CAGEFFiveFoldTrainer:
    def __init__(self, params: Dict) -> None:
        self.params = params
        # 兼容未通过新版命令行解析器构造的旧参数字典。
        self.params.setdefault("graph_dropout", 0.1)
        self.device = resolve_device(params["device"])
        self.params["resolved_device"] = str(self.device)
        self._initialize_dataset()

    def _initialize_dataset(self) -> None:
        (
            self.data_train_raw,
            self.data_test_raw,
            labels_train_raw,
            labels_test_raw,
            self.modality_names,
        ) = load_raw_data(
            self.params["data_folder"],
            requested_modalities=self.params.get("modalities"),
        )
        self.n_views = len(self.modality_names)
        self.params["modality_names"] = list(self.modality_names)
        self.params["n_views"] = self.n_views

        # 类别映射只能由开发/训练集定义，外部测试标签不能参与模型设定。
        train_classes = np.unique(labels_train_raw)
        if train_classes.size < 2:
            raise ValueError("训练集必须至少包含两个类别")
        label_map = {int(old): new for new, old in enumerate(train_classes)}
        unknown_test = sorted(set(map(int, labels_test_raw)) - set(label_map))
        if unknown_test:
            raise ValueError(
                "外部测试集包含训练集中从未出现的标签："
                f"{unknown_test}；无法进行有监督评估"
            )
        self.label_map = label_map
        self.labels_train = np.array(
            [label_map[int(x)] for x in labels_train_raw], dtype=np.int64
        )
        self.labels_test = np.array(
            [label_map[int(x)] for x in labels_test_raw], dtype=np.int64
        )
        self.num_classes = train_classes.size
        self.input_dims = [view.shape[1] for view in self.data_train_raw]
        prediction_dims = validate_positive_int_list(
            self.params["prediction"], "prediction"
        )
        self.prediction_dims = {
            view: list(prediction_dims) for view in range(self.n_views)
        }

        n_folds = int(self.params["num_folds"])
        if n_folds < 2:
            raise ValueError("num_folds 必须至少为 2")
        unique, counts = np.unique(self.labels_train, return_counts=True)
        least_populated = int(counts.min())
        if least_populated < n_folds:
            original_label = int(train_classes[unique[counts.argmin()]])
            raise ValueError(
                f"类别 {original_label} 只有 {least_populated} 个训练样本，"
                f"不足以进行 {n_folds} 折分层交叉验证"
            )

        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=int(self.params["seed"]),
        )
        self.fold_indices = list(
            splitter.split(np.zeros(self.labels_train.size), self.labels_train)
        )

        missing_rate = float(self.params["missing_rate"])
        mask_rng = np.random.default_rng(int(self.params["seed"]) + 500_000)
        self.development_mask = make_sample_level_mask(
            self.labels_train.size, self.n_views, missing_rate, mask_rng
        )
        test_mask_rng = np.random.default_rng(
            int(self.params["seed"]) + 1_000_000
        )
        self.test_mask = make_sample_level_mask(
            self.labels_test.size, self.n_views, missing_rate, test_mask_rng
        )

        print(f"[信息] 类别数：{self.num_classes}")
        print(
            f"[信息] 模态数：{self.n_views}；"
            f"模态顺序：{', '.join(self.modality_names)}"
        )
        print(
            f"[信息] {n_folds} 折分层交叉验证："
            f"开发集 {self.labels_train.size} 例，外部测试集 {self.labels_test.size} 例"
        )
        for fold, (train_idx, val_idx) in enumerate(self.fold_indices, start=1):
            print(
                f"       Fold {fold}: train={len(train_idx)}, "
                f"validation={len(val_idx)}"
            )
        print(f"[信息] 计算设备：{self.device}")
        print(
            "[信息] AMP："
            f"{'开启' if self._amp_enabled() else '关闭'}；梯度裁剪 max_norm=1.0"
        )

    def _amp_enabled(self) -> bool:
        return (
            bool(self.params.get("use_amp", False))
            and self.device.type == "cuda"
        )

    def _setup_fold(self, fold_index: int) -> None:
        train_indices, validation_indices = self.fold_indices[fold_index]
        self.train_data: TensorList = []
        self.validation_data: TensorList = []
        self.test_data: TensorList = []
        self.original_train_data: TensorList = []
        self.nonconstant_feature_masks: TensorList = []

        for raw_train, raw_test in zip(
            self.data_train_raw, self.data_test_raw
        ):
            fold_train = raw_train[train_indices]
            train_min = fold_train.min(axis=0, keepdims=True)
            raw_scale = fold_train.max(axis=0, keepdims=True) - train_min
            nonconstant = raw_scale > 1e-12
            safe_scale = np.where(nonconstant, raw_scale, 1.0)

            train_normalized = (
                (fold_train - train_min) / safe_scale
            ).astype(np.float32)
            validation_normalized = (
                (raw_train[validation_indices] - train_min) / safe_scale
            ).astype(np.float32)
            test_normalized = (
                (raw_test - train_min) / safe_scale
            ).astype(np.float32)

            # 训练折中的常数特征没有可学习信息。验证/测试集即使该列出现
            # 波动，也必须保持为 0；否则未经训练的随机权重会影响预测。
            constant_columns = ~nonconstant.squeeze(0)
            if constant_columns.any():
                train_normalized[:, constant_columns] = 0.0
                validation_normalized[:, constant_columns] = 0.0
                test_normalized[:, constant_columns] = 0.0

            train_tensor = torch.from_numpy(train_normalized).to(self.device)
            validation_tensor = torch.from_numpy(
                validation_normalized
            ).to(self.device)
            test_tensor = torch.from_numpy(test_normalized).to(self.device)
            feature_mask = torch.from_numpy(
                nonconstant.astype(np.float32)
            ).to(self.device)

            # 后续通过重新赋值生成掩码输入，不会原地修改 train_tensor，
            # 因此无需额外 clone 一份完整训练数据。
            self.original_train_data.append(train_tensor)
            self.train_data.append(train_tensor)
            self.validation_data.append(validation_tensor)
            self.test_data.append(test_tensor)
            self.nonconstant_feature_masks.append(feature_mask)

        self.train_labels = torch.from_numpy(
            self.labels_train[train_indices]
        ).long().to(self.device)

        self.train_mask = torch.from_numpy(
            self.development_mask[train_indices]
        ).to(self.device)
        self.validation_mask = torch.from_numpy(
            self.development_mask[validation_indices]
        ).to(self.device)
        self.test_mask_tensor = torch.from_numpy(self.test_mask).to(self.device)

        for view in range(self.n_views):
            self.train_data[view] = (
                self.train_data[view] * self.train_mask[:, view : view + 1]
            )
            self.validation_data[view] = (
                self.validation_data[view]
                * self.validation_mask[:, view : view + 1]
            )
            self.test_data[view] = (
                self.test_data[view] * self.test_mask_tensor[:, view : view + 1]
            )

        self.model = GNNEhancedCAGEF(
            in_dims=self.input_dims,
            hidden_dims=self.params["hidden_dim"],
            num_classes=self.num_classes,
            dropout=self.params["dropout"],
            prediction_dims=self.prediction_dims,
            use_gnn=self.params["use_gnn"],
            fast_path=float(self.params["missing_rate"]) == 0.0,
            graph_dropout=self.params["graph_dropout"],
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.params["step_size"],
            gamma=self.params["lr_gamma"],
        )
        self.scaler = self._create_grad_scaler()

    def _create_grad_scaler(self):
        if not self._amp_enabled():
            return None
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()

    def _autocast_context(self):
        try:
            return torch.amp.autocast(
                device_type="cuda", enabled=self._amp_enabled()
            )
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=self._amp_enabled())

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        augmented = data_augmentation(
            self.train_data,
            self.train_mask,
            augmentation_rate=self.params["augmentation_rate"],
            noise_std=self.params["noise_std"],
            feature_masks=self.nonconstant_feature_masks,
        )

        with self._autocast_context():
            loss, _, components = self.model.training_loss(
                augmented,
                self.train_mask,
                self.train_labels,
                self.original_train_data,
                lambda_impute=self.params["lambda_impute"],
                lambda_consist=self.params["lambda_consist"],
                lambda_graph=self.params["lambda_graph"],
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"训练损失出现 NaN/Inf，损失分量为：{components}"
            )

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "反向传播梯度出现 NaN/Inf；已取消本轮参数更新"
            )
        components["gradient_norm"] = float(gradient_norm.detach().item())

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        return components

    @torch.no_grad()
    def evaluate(
        self, data_list: TensorList, sample_mask: torch.Tensor
    ) -> np.ndarray:
        self.model.eval()
        logits = self.model.infer(data_list, sample_mask)
        probabilities = F.softmax(logits, dim=1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("推理概率出现 NaN 或 Inf")
        return probabilities.cpu().numpy()

    def save_checkpoint(self, fold_path: Path) -> None:
        fold_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), fold_path / "checkpoint.pt")

    def load_checkpoint(self, fold_path: Path) -> None:
        path = fold_path / "checkpoint.pt"
        try:
            state = torch.load(
                path, map_location=self.device, weights_only=True
            )
        except TypeError:
            state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    def _classification_metrics(
        self, labels: np.ndarray, probabilities: np.ndarray
    ) -> Tuple[List[str], List[float]]:
        prediction = probabilities.argmax(axis=1)
        accuracy = accuracy_score(labels, prediction)
        if self.num_classes == 2:
            f1 = f1_score(labels, prediction, zero_division=0)
            try:
                auc = roc_auc_score(labels, probabilities[:, 1])
            except ValueError:
                auc = float("nan")
            return ["ACC", "F1", "AUC"], [accuracy, f1, auc]

        f1_weighted = f1_score(
            labels, prediction, average="weighted", zero_division=0
        )
        f1_macro = f1_score(
            labels, prediction, average="macro", zero_division=0
        )
        return (
            ["ACC", "F1_weighted", "F1_macro"],
            [accuracy, f1_weighted, f1_macro],
        )

    def _selection_score(
        self, metric_names: Sequence[str], metric_values: Sequence[float]
    ) -> Tuple[str, float]:
        """
        返回早停与最佳模型选择指标。

        auto 模式下，二分类优先使用 AUC，多分类优先使用宏平均 F1；
        若目标指标因单类验证集等原因不可计算，则回退到 ACC。
        """
        requested = self.params.get("selection_metric", "auto")
        if requested == "auto":
            target_name = "AUC" if self.num_classes == 2 else "F1_macro"
        else:
            target_name = "ACC"

        metric_map = {
            name: float(value)
            for name, value in zip(metric_names, metric_values)
        }
        score = metric_map.get(target_name, float("nan"))
        if not math.isfinite(score):
            target_name = "ACC"
            score = metric_map["ACC"]
        return target_name, score

    def train(self) -> Dict[str, object]:
        n_folds = int(self.params["num_folds"])
        dataset_name = Path(self.params["data_folder"]).resolve().name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment_path = (
            Path(self.params["exp"])
            / (
                f"{dataset_name}_seed{int(self.params['seed'])}_"
                f"CV{n_folds}_{timestamp}"
            )
        )
        experiment_path.mkdir(parents=True, exist_ok=True)

        config_to_save = dict(self.params)
        config_to_save["label_map"] = {
            str(key): value for key, value in self.label_map.items()
        }
        with (experiment_path / "config.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(config_to_save, file, indent=2, ensure_ascii=False)

        oof_probabilities = np.full(
            (self.labels_train.size, self.num_classes),
            np.nan,
            dtype=np.float64,
        )
        test_probabilities_by_fold: List[np.ndarray] = []
        validation_metrics: List[List[float]] = []
        fold_results: List[Dict] = []

        for fold_index, (train_indices, validation_indices) in enumerate(
            self.fold_indices
        ):
            fold_number = fold_index + 1
            fold_seed = int(self.params["seed"]) + fold_index
            # 必须在创建该折模型之前设置种子。
            seed_everything(fold_seed)
            self._setup_fold(fold_index)
            fold_path = experiment_path / f"fold_{fold_number}"

            print("\n" + "=" * 68)
            print(
                f"Fold {fold_number}/{n_folds}: train={len(train_indices)}, "
                f"validation={len(validation_indices)}, seed={fold_seed}"
            )
            print("=" * 68)

            best_score = -math.inf
            best_selection_metric = None
            best_epoch = 0
            no_improvement = 0
            n_epochs = int(self.params["num_epoch"])
            interval = int(self.params["test_interval"])
            patience = int(self.params["patience"])

            progress = tqdm(
                range(1, n_epochs + 1),
                desc=f"Fold {fold_number}",
                dynamic_ncols=True,
            )
            for epoch in progress:
                components = self.train_epoch()
                self.scheduler.step()
                progress.set_postfix(loss=f"{components['total']:.4f}")

                should_evaluate = (
                    epoch == 1
                    or epoch % interval == 0
                    or epoch == n_epochs
                )
                if not should_evaluate:
                    continue

                validation_probabilities = self.evaluate(
                    self.validation_data, self.validation_mask
                )
                current_metric_names, current_metrics = self._classification_metrics(
                    self.labels_train[validation_indices],
                    validation_probabilities,
                )
                selection_name, selection_score = self._selection_score(
                    current_metric_names, current_metrics
                )
                print(
                    f"Fold {fold_number}, epoch {epoch}: "
                    + ", ".join(
                        f"{key}={value:.5f}"
                        for key, value in zip(
                            current_metric_names,
                            current_metrics,
                        )
                    )
                    + f", selection={selection_name}"
                )

                if selection_score > (
                    best_score + float(self.params["min_delta"])
                ):
                    best_score = selection_score
                    best_selection_metric = selection_name
                    best_epoch = epoch
                    no_improvement = 0
                    self.save_checkpoint(fold_path)
                else:
                    no_improvement += 1

                if patience > 0 and no_improvement >= patience:
                    print(
                        f"[信息] Fold {fold_number} 在 epoch {epoch} 早停；"
                        f"最佳 epoch={best_epoch}"
                    )
                    break

            progress.close()
            self.load_checkpoint(fold_path)

            validation_probabilities = self.evaluate(
                self.validation_data, self.validation_mask
            )
            metric_names, fold_metrics = self._classification_metrics(
                self.labels_train[validation_indices],
                validation_probabilities,
            )
            print(
                f"Fold {fold_number} 最佳验证结果（epoch {best_epoch}）："
                + ", ".join(
                    f"{name}={value:.5f}"
                    for name, value in zip(metric_names, fold_metrics)
                )
            )
            oof_probabilities[validation_indices] = validation_probabilities
            validation_metrics.append(fold_metrics)

            fold_test_probabilities = self.evaluate(
                self.test_data, self.test_mask_tensor
            )
            test_probabilities_by_fold.append(fold_test_probabilities)
            fold_results.append(
                {
                    "fold": fold_number,
                    "seed": fold_seed,
                    "best_epoch": best_epoch,
                    "selection_metric": best_selection_metric,
                    "best_selection_score": safe_float(best_score),
                    "metrics": [safe_float(x) for x in fold_metrics],
                }
            )

            del self.model, self.optimizer, self.scheduler
            self.scaler = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if np.isnan(oof_probabilities).any():
            raise RuntimeError("OOF 预测存在未被任何验证折覆盖的样本")

        fold_array = np.asarray(validation_metrics, dtype=np.float64)
        fold_means = np.nanmean(fold_array, axis=0)
        fold_stds = np.nanstd(fold_array, axis=0, ddof=1)
        _, oof_metrics = self._classification_metrics(
            self.labels_train, oof_probabilities
        )

        ensemble_test_probabilities = np.mean(
            np.stack(test_probabilities_by_fold, axis=0), axis=0
        )
        _, heldout_test_metrics = self._classification_metrics(
            self.labels_test, ensemble_test_probabilities
        )

        print("\n" + "=" * 68)
        print(f"{n_folds} 折交叉验证汇总")
        print("=" * 68)
        for name, mean, std in zip(metric_names, fold_means, fold_stds):
            print(f"折验证 {name}: {mean:.5f} +/- {std:.5f}")
        print(
            "汇总 OOF: "
            + ", ".join(
                f"{name}={value:.5f}"
                for name, value in zip(metric_names, oof_metrics)
            )
        )
        print(
            f"预留测试集 {n_folds} 模型概率集成: "
            + ", ".join(
                f"{name}={value:.5f}"
                for name, value in zip(metric_names, heldout_test_metrics)
            )
        )

        validation_fold = np.empty(self.labels_train.size, dtype=np.int64)
        for fold_index, (_, validation_indices) in enumerate(self.fold_indices):
            validation_fold[validation_indices] = fold_index + 1

        np.savez_compressed(
            experiment_path / "predictions.npz",
            oof_probabilities=oof_probabilities,
            oof_labels=self.labels_train,
            test_probabilities=ensemble_test_probabilities,
            test_labels=self.labels_test,
            modality_names=np.asarray(self.modality_names),
            development_mask=self.development_mask,
            test_mask=self.test_mask,
            validation_fold=validation_fold,
        )
        summary = {
            "n_views": self.n_views,
            "modality_names": list(self.modality_names),
            "metric_names": metric_names,
            "fold_mean": [safe_float(x) for x in fold_means],
            "fold_std": [safe_float(x) for x in fold_stds],
            "oof_metrics": [safe_float(x) for x in oof_metrics],
            "test_ensemble_metrics": [
                safe_float(x) for x in heldout_test_metrics
            ],
            "folds": fold_results,
        }
        with (experiment_path / "summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到：{experiment_path}")
        return {
            "metric_names": list(metric_names),
            "fold_mean": fold_means,
            "fold_std": fold_stds,
            "oof_metrics": np.asarray(oof_metrics, dtype=np.float64),
            "test_ensemble_metrics": np.asarray(
                heldout_test_metrics, dtype=np.float64
            ),
            "experiment_path": str(experiment_path),
        }

