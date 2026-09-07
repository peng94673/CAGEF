from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    TensorList,
    choose_attention_heads,
    safe_cosine_similarity,
    stable_l2_normalize,
    validate_modality_tensors,
    validate_positive_int_list,
    validate_sample_mask,
)

class GraphAttentionLayer(nn.Module):
    """同时支持 [N,F] 和 [B,N,F] 输入的图注意力层。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        alpha: float,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.out_features = int(out_features)
        self.concat = bool(concat)

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.attention_source = nn.Parameter(torch.empty(out_features, 1))
        self.attention_target = nn.Parameter(torch.empty(out_features, 1))
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attention_source, gain=1.414)
        nn.init.xavier_uniform_(self.attention_target, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        transformed = torch.matmul(features, self.weight)
        source_score = torch.matmul(transformed, self.attention_source)
        target_score = torch.matmul(transformed, self.attention_target)
        edge_score = self.leaky_relu(
            source_score + target_score.transpose(-2, -1)
        )

        if adjacency.ndim == 2 and features.ndim == 3:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.shape[-2:] != edge_score.shape[-2:]:
            raise ValueError(
                "邻接矩阵节点数与特征节点数不一致："
                f"{tuple(adjacency.shape)} vs {tuple(features.shape)}"
            )

        # 邻接矩阵不仅用于判断边是否存在，其数值还作为注意力先验。
        # 原实现只判断 adjacency > 0；由于邻接值来自 sigmoid，几乎所有边
        # 都大于 0，导致 relation_weights/relation_bias 不影响 GAT 输出。
        edge_exists = adjacency > 0
        minimum_positive = torch.finfo(edge_score.dtype).tiny
        log_adjacency_prior = torch.log(
            adjacency.to(edge_score.dtype).clamp_min(minimum_positive)
        )
        weighted_score = edge_score + log_adjacency_prior
        negative_large = torch.finfo(edge_score.dtype).min
        masked_score = weighted_score.masked_fill(
            ~edge_exists, negative_large
        )
        attention = F.softmax(masked_score, dim=-1)
        attention = F.dropout(
            attention, p=self.dropout, training=self.training
        )
        output = torch.matmul(attention, transformed)
        return F.elu(output) if self.concat else output


class GAT(nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        alpha: float,
        nheads: int,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.attentions = nn.ModuleList(
            [
                GraphAttentionLayer(
                    nfeat, nhid, dropout=dropout, alpha=alpha, concat=True
                )
                for _ in range(nheads)
            ]
        )
        self.output_attention = GraphAttentionLayer(
            nhid * nheads,
            nclass,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = F.dropout(features, p=self.dropout, training=self.training)
        x = torch.cat(
            [attention(x, adjacency) for attention in self.attentions], dim=-1
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.elu(self.output_attention(x, adjacency))


class ModalityRelationGraph(nn.Module):
    """对每个样本分别构造一个含 n_views 个模态节点的小图。"""

    def __init__(
        self,
        hidden_dim: int,
        n_views: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_views = int(n_views)
        if self.n_views < 2:
            raise ValueError("模态关系图至少需要两个模态节点")
        self.modality_transform = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_views)]
        )
        self.relation_weights = nn.Parameter(torch.ones(self.n_views, self.n_views))
        self.relation_bias = nn.Parameter(torch.zeros(self.n_views, self.n_views))
        self.gat = GAT(
            nfeat=hidden_dim,
            nhid=max(1, hidden_dim // 2),
            nclass=hidden_dim,
            dropout=dropout,
            alpha=0.2,
            nheads=num_heads,
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)

    def _transform_modalities(
        self,
        modality_features: Sequence[torch.Tensor],
        sample_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """使用关系图自身的投影层变换模态，并在变换后清除缺失节点。"""
        batch_size = validate_modality_tensors(
            modality_features,
            "关系图输入",
            expected_views=self.n_views,
            expected_dims=[self.hidden_dim] * self.n_views,
        )
        if sample_mask is not None:
            validate_sample_mask(sample_mask, batch_size, self.n_views)

        transformed = [
            layer(feature)
            for layer, feature in zip(
                self.modality_transform, modality_features
            )
        ]
        if sample_mask is not None:
            # 线性层含偏置，零输入也可能产生非零节点。因此必须在变换后
            # 再次应用掩码。
            transformed = [
                feature * sample_mask[:, view : view + 1].to(feature.dtype)
                for view, feature in enumerate(transformed)
            ]
        return transformed

    def _create_modality_adjacency(
        self,
        modality_features: Sequence[torch.Tensor],
        sample_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        返回 [B,n_views,n_views] 邻接矩阵。

        原代码用 torch.kron(I_B, A) 构造样本优先邻接矩阵，却把节点特征
        按模态优先拼接，导致边连接到错误节点。这里直接使用批量小图，
        同时避免生成 O(B²) 的巨大稠密矩阵。
        """
        nodes = torch.stack(list(modality_features), dim=1)  # [B,V,H]
        # 缺失节点为零向量。显式使用 float32 计算相似度，避免 AMP 下
        # float16 的极小 eps 下溢并引发 NaN。
        similarity_nodes = (
            nodes.float()
            if nodes.dtype in (torch.float16, torch.bfloat16)
            else nodes
        )
        normalized = stable_l2_normalize(similarity_nodes, dim=-1, eps=1e-8)
        similarity = (
            normalized.unsqueeze(2) * normalized.unsqueeze(1)
        ).sum(dim=-1)
        adjacency = torch.sigmoid(
            similarity * self.relation_weights.unsqueeze(0)
            + self.relation_bias.unsqueeze(0)
        )

        eye = torch.eye(
            self.n_views, device=adjacency.device, dtype=adjacency.dtype
        ).unsqueeze(0)
        if sample_mask is not None:
            observed = (sample_mask > 0.5).to(adjacency.dtype)
            observed_pairs = observed.unsqueeze(2) * observed.unsqueeze(1)
            # 缺失模态不能向真实模态发送消息。所有节点仍保留自环，
            # 以避免完全屏蔽的一行产生不稳定 softmax；缺失节点的输出不会
            # 被用作后续注意力的 key/value。
            adjacency = adjacency * observed_pairs
        # 所有节点显式保留自环，防止自身信息丢失，并避免全屏蔽行产生
        # 不稳定 softmax。
        adjacency = adjacency * (1.0 - eye) + eye
        return adjacency

    def project_and_build_adjacency(
        self,
        modality_features: Sequence[torch.Tensor],
        sample_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """公开的关系空间投影与邻接矩阵构建接口。"""
        transformed = self._transform_modalities(
            modality_features, sample_mask=sample_mask
        )
        adjacency = self._create_modality_adjacency(
            transformed, sample_mask=sample_mask
        )
        return transformed, adjacency

    def forward(
        self,
        modality_features: Sequence[torch.Tensor],
        sample_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        transformed, adjacency = self.project_and_build_adjacency(
            modality_features, sample_mask=sample_mask
        )
        nodes = torch.stack(transformed, dim=1)  # 样本优先排列 [B,V,H]
        enhanced = self.gat(nodes, adjacency)
        enhanced = self.residual_norm(enhanced + nodes)
        return [enhanced[:, view, :] for view in range(self.n_views)]


# =============================================================================
# 预测与主模型
# =============================================================================
def xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LinearLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.apply(xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Prediction(nn.Module):
    """跨模态预测自编码器，输入与重建输出维数均为 hidden_dim。"""

    def __init__(
        self,
        prediction_dim: Sequence[int],
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        dims = validate_positive_int_list(prediction_dim, "prediction_dim")
        if len(dims) < 2:
            raise ValueError("prediction_dim 至少应包含输入维数和一个隐层维数")
        self.batchnorm = bool(batchnorm)
        self.encoder = self._build_encoder(dims, activation, batchnorm)
        self.decoder = self._build_decoder(dims, activation, batchnorm)

    @staticmethod
    def _activation(name: str) -> nn.Module:
        mapping = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leakyrelu": nn.LeakyReLU(0.2),
        }
        if name not in mapping:
            raise ValueError(f"不支持的激活函数：{name}")
        return mapping[name]

    @classmethod
    def _build_encoder(
        cls, dims: Sequence[int], activation: str, batchnorm: bool
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if batchnorm and out_dim > 1:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(cls._activation(activation))
        return nn.Sequential(*layers)

    @classmethod
    def _build_decoder(
        cls, dims: Sequence[int], activation: str, batchnorm: bool
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        reversed_dims = list(reversed(dims))
        for index, (in_dim, out_dim) in enumerate(
            zip(reversed_dims[:-1], reversed_dims[1:])
        ):
            layers.append(nn.Linear(in_dim, out_dim))
            is_output = index == len(reversed_dims) - 2
            if not is_output:
                if batchnorm and out_dim > 1:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(cls._activation(activation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 当前模型默认 batchnorm=False；此分支保留小批量兼容性。
        temporarily_eval = self.training and self.batchnorm and x.shape[0] == 1
        if temporarily_eval:
            self.eval()
        latent = self.encoder(x)
        output = self.decoder(latent)
        if temporarily_eval:
            self.train()
        return output, latent


class ConfidenceMechanism(nn.Module):
    def __init__(self, hidden_dim: int, n_views: int) -> None:
        super().__init__()
        mid_dim = max(1, hidden_dim // 2)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + n_views, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, feature_embedding: torch.Tensor, missing_pattern: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat(
            [feature_embedding, missing_pattern.to(feature_embedding.dtype)], dim=1
        )
        return self.network(combined)


class GraphEnhancedPrediction(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_views: int,
        prediction_dims: Dict[int, Sequence[int]],
        use_gnn: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_views = int(n_views)
        if self.n_views < 2:
            raise ValueError("跨模态预测至少需要两个模态")
        self.use_gnn = bool(use_gnn)
        attention_heads = choose_attention_heads(hidden_dim, preferred=4)
        if self.use_gnn:
            self.relation_graph = ModalityRelationGraph(
                hidden_dim,
                n_views=self.n_views,
                num_heads=attention_heads,
                dropout=dropout,
            )

        source_dims = {
            view: [hidden_dim] + list(prediction_dims[view])
            for view in range(self.n_views)
        }
        self.cross_modal_predictors = nn.ModuleDict(
            {
                self.pair_key(source, target): Prediction(source_dims[source])
                for source in range(self.n_views)
                for target in range(self.n_views)
                if source != target
            }
        )
        if self.use_gnn:
            self.relation_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )

    @staticmethod
    def pair_key(source_view: int, target_view: int) -> str:
        return f"view_{source_view}_to_{target_view}"

    def _validate_pair(self, source_view: int, target_view: int) -> None:
        if not 0 <= source_view < self.n_views:
            raise IndexError(f"source_view 越界：{source_view}")
        if not 0 <= target_view < self.n_views:
            raise IndexError(f"target_view 越界：{target_view}")
        if source_view == target_view:
            raise ValueError("source_view 与 target_view 不能相同")

    def _run_predictor(
        self,
        features: torch.Tensor,
        source_view: int,
        target_view: int,
    ) -> torch.Tensor:
        predictor = self.cross_modal_predictors[
            self.pair_key(source_view, target_view)
        ]
        prediction, _ = predictor(features)
        return prediction

    def predict_plain(
        self,
        source_features: torch.Tensor,
        source_view: int,
        target_view: int,
    ) -> torch.Tensor:
        self._validate_pair(source_view, target_view)
        return self._run_predictor(source_features, source_view, target_view)

    def predict_from_enhanced(
        self,
        enhanced_features: Sequence[torch.Tensor],
        source_view: int,
        target_view: int,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_gnn:
            raise RuntimeError(
                "仅当 use_gnn=True 时才能使用图增强跨模态预测"
            )
        self._validate_pair(source_view, target_view)
        batch_size = validate_modality_tensors(
            enhanced_features,
            "图增强输入",
            expected_views=self.n_views,
            expected_dims=[self.hidden_dim] * self.n_views,
        )
        validate_sample_mask(sample_mask, batch_size, self.n_views)
        source = enhanced_features[source_view]
        keys = torch.stack(list(enhanced_features), dim=1)
        query = source.unsqueeze(1)
        key_padding_mask = sample_mask < 0.5
        attended, _ = self.relation_attention(
            query,
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self._run_predictor(
            attended.squeeze(1),
            source_view,
            target_view,
        )


class GNNEhancedCAGEF(nn.Module):
    def __init__(
        self,
        in_dims: Sequence[int],
        hidden_dims: Sequence[int],
        num_classes: int,
        dropout: float,
        prediction_dims: Dict[int, Sequence[int]],
        use_gnn: bool = True,
        fast_path: bool = False,
        graph_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(in_dims) < 2:
            raise ValueError("CAGEF 至少需要两个模态")
        in_dims = validate_positive_int_list(in_dims, "in_dims")
        hidden_dims = validate_positive_int_list(hidden_dims, "hidden_dims")
        if num_classes < 2:
            raise ValueError("分类类别数必须至少为 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")
        if not 0.0 <= graph_dropout < 1.0:
            raise ValueError("graph_dropout 必须位于 [0,1)")

        self.n_views = len(in_dims)
        self.in_dims = list(in_dims)
        self.hidden_dim = hidden_dims[0]
        self.dropout = float(dropout)
        self.use_gnn = bool(use_gnn)
        self.fast_path = bool(fast_path)

        self.feature_attention = nn.ModuleList(
            [LinearLayer(dim, dim) for dim in in_dims]
        )
        self.embeddings = nn.ModuleList(
            [LinearLayer(dim, self.hidden_dim) for dim in in_dims]
        )
        self.sample_attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, max(1, self.hidden_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(1, self.hidden_dim // 2), 1),
                    nn.Sigmoid(),
                )
                for _ in range(self.n_views)
            ]
        )
        self.confidence_net = ConfidenceMechanism(self.hidden_dim, self.n_views)

        classifier: List[nn.Module] = []
        previous_dim = self.n_views * self.hidden_dim
        # 原代码遗漏了 hidden_dim 列表的最后一个元素；这里使用全部后续维度。
        for current_dim in hidden_dims[1:]:
            classifier.extend(
                [
                    LinearLayer(previous_dim, current_dim),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout),
                ]
            )
            previous_dim = current_dim
        classifier.append(LinearLayer(previous_dim, num_classes))
        self.classifier = nn.Sequential(*classifier)
        self.criterion = nn.CrossEntropyLoss()

        expected_prediction_keys = set(range(self.n_views))
        actual_prediction_keys = set(prediction_dims)
        if actual_prediction_keys != expected_prediction_keys:
            raise ValueError(
                "prediction_dims 的模态索引必须完整；"
                f"期望={sorted(expected_prediction_keys)}，"
                f"实际={sorted(actual_prediction_keys)}"
            )
        normalized_prediction_dims = {
            view: validate_positive_int_list(
                prediction_dims[view], f"prediction_dims[{view}]"
            )
            for view in range(self.n_views)
        }
        self.graph_predictor = GraphEnhancedPrediction(
            self.hidden_dim,
            n_views=self.n_views,
            prediction_dims=normalized_prediction_dims,
            use_gnn=self.use_gnn,
            dropout=float(graph_dropout),
        )

    def _encode_modalities(
        self,
        data_list: TensorList,
        input_name: str,
        apply_dropout: bool,
    ) -> Dict[int, torch.Tensor]:
        """共享的特征加权与模态编码流程。"""
        validate_modality_tensors(
            data_list,
            input_name,
            expected_views=self.n_views,
            expected_dims=self.in_dims,
        )
        embeddings: Dict[int, torch.Tensor] = {}
        for view, data in enumerate(data_list):
            feature_weights = torch.sigmoid(self.feature_attention[view](data))
            representation = F.relu(
                self.embeddings[view](data * feature_weights)
            )
            if apply_dropout:
                representation = F.dropout(
                    representation,
                    p=self.dropout,
                    training=self.training,
                )
            embeddings[view] = representation
        return embeddings

    def get_base_embeddings(self, data_list: TensorList) -> Dict[int, torch.Tensor]:
        return self._encode_modalities(data_list, "输入", apply_dropout=True)

    def get_original_features(
        self, original_data_list: TensorList
    ) -> Dict[int, torch.Tensor]:
        """生成无 dropout 的监督目标，并阻断目标分支梯度。"""
        with torch.no_grad():
            return self._encode_modalities(
                original_data_list,
                "原始监督",
                apply_dropout=False,
            )

    def _predict_target_from_source(
        self,
        source_view: int,
        target_view: int,
        indices: torch.Tensor,
        base_embeddings: Dict[int, torch.Tensor],
        enhanced_embeddings: Optional[List[torch.Tensor]],
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        if enhanced_embeddings is None:
            return self.graph_predictor.predict_plain(
                base_embeddings[source_view][indices],
                source_view,
                target_view,
            )
        subset = [features[indices] for features in enhanced_embeddings]
        return self.graph_predictor.predict_from_enhanced(
            subset,
            source_view,
            target_view,
            sample_mask[indices],
        )

    def complete_modalities(
        self,
        base_embeddings: Dict[int, torch.Tensor],
        sample_mask: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        batch_size = sample_mask.shape[0]
        device = sample_mask.device
        observed_mask = sample_mask > 0.5
        observed_count = observed_mask.sum(dim=1, keepdim=True)
        if self.use_gnn:
            enhanced = self.graph_predictor.relation_graph(
                [base_embeddings[view] for view in range(self.n_views)],
                sample_mask=sample_mask,
            )
        else:
            enhanced = None

        completed: Dict[int, torch.Tensor] = {}
        confidences: Dict[int, torch.Tensor] = {}

        for target_view in range(self.n_views):
            target_base = base_embeddings[target_view]
            prediction_sum = torch.zeros_like(target_base)
            missing = ~observed_mask[:, target_view]

            # 按来源模态批量预测，避免对每个缺失样本逐个运行注意力网络。
            for source_view in range(self.n_views):
                if source_view == target_view:
                    continue
                valid = missing & observed_mask[:, source_view]
                indices = torch.where(valid)[0]
                if indices.numel() == 0:
                    continue
                prediction = self._predict_target_from_source(
                    source_view,
                    target_view,
                    indices,
                    base_embeddings,
                    enhanced,
                    sample_mask,
                )
                prediction_sum = prediction_sum.index_add(
                    0, indices, prediction
                )

            has_prediction = observed_count.squeeze(1) > 0
            missing_without_source = missing & ~has_prediction
            if missing_without_source.any():
                bad_rows = torch.where(missing_without_source)[0].tolist()
                raise RuntimeError(
                    f"目标模态 {target_view} 的缺失样本没有可用来源模态："
                    f"样本索引 {bad_rows[:10]}"
                )
            averaged_prediction = prediction_sum / observed_count.to(
                target_base.dtype
            ).clamp_min(1.0)
            observed_representation = (
                enhanced[target_view]
                if enhanced is not None
                else target_base
            )
            target_completed = torch.where(
                missing.unsqueeze(1),
                averaged_prediction,
                observed_representation,
            )

            missing_indices = torch.where(missing)[0]
            if missing_indices.numel() > 0:
                dynamic_confidence = self.confidence_net(
                    target_completed[missing_indices],
                    sample_mask[missing_indices],
                )
                # AMP 下 target_completed 可能为 Float32，而线性网络输出为
                # Float16。index_copy 要求两端 dtype 完全一致，因此占位
                # 张量必须跟随动态置信度的实际输出类型创建。
                confidence = torch.ones(
                    batch_size,
                    1,
                    device=device,
                    dtype=dynamic_confidence.dtype,
                ).index_copy(
                    0, missing_indices, dynamic_confidence
                )
            else:
                confidence = torch.ones(
                    batch_size,
                    1,
                    device=device,
                    dtype=target_completed.dtype,
                )

            completed[target_view] = target_completed
            confidences[target_view] = confidence

        return completed, confidences

    def forward_embeddings(
        self, data_list: TensorList, sample_mask: torch.Tensor
    ) -> Tuple[
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
    ]:
        base = self.get_base_embeddings(data_list)
        batch_size = base[0].shape[0]
        validate_sample_mask(sample_mask, batch_size, self.n_views)
        if self.fast_path and torch.any(sample_mask < 0.5):
            raise ValueError(
                "fast_path 仅适用于所有模态均完整的输入"
            )
        if self.fast_path:
            if self.use_gnn:
                # 完整数据也应真正经过 GNN；否则 use_gnn=True 在
                # missing_rate=0 时不会产生任何作用。
                graph_enhanced = self.graph_predictor.relation_graph(
                    [base[view] for view in range(self.n_views)],
                    sample_mask=sample_mask,
                )
                completed = {
                    view: graph_enhanced[view]
                    for view in range(self.n_views)
                }
            else:
                completed = base
            confidence = {
                view: torch.ones(
                    batch_size,
                    1,
                    device=sample_mask.device,
                    dtype=base[view].dtype,
                )
                for view in range(self.n_views)
            }
        else:
            completed, confidence = self.complete_modalities(base, sample_mask)

        final: Dict[int, torch.Tensor] = {}
        for view in range(self.n_views):
            attention = self.sample_attention[view](completed[view])
            final[view] = completed[view] * attention * confidence[view]
        return final, completed, confidence

    def _imputation_loss(
        self,
        completed: Dict[int, torch.Tensor],
        confidences: Dict[int, torch.Tensor],
        sample_mask: torch.Tensor,
        original_features: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        observed_mask = sample_mask > 0.5
        for target_view in range(self.n_views):
            # forward_embeddings 已保证每个样本至少存在一个模态，因此目标
            # 缺失时必然至少有一个其他来源模态可用。
            valid = ~observed_mask[:, target_view]
            if not valid.any():
                continue

            per_sample_mse = F.mse_loss(
                completed[target_view][valid],
                original_features[target_view][valid],
                reduction="none",
            ).mean(dim=1)
            confidence = confidences[target_view][valid].squeeze(1)

            # 避免原来的 (1-confidence)*MSE 退化为 confidence 恒等于 1。
            confidence_target = torch.exp(-per_sample_mse.detach()).clamp(0.0, 1.0)
            calibration = F.mse_loss(confidence, confidence_target)
            losses.append(per_sample_mse.mean() + 0.1 * calibration)

        if not losses:
            return sample_mask.new_zeros(())
        return torch.stack(losses).mean()

    def _modality_consistency_loss(
        self,
        completed: Dict[int, torch.Tensor],
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        observed_mask = sample_mask > 0.5
        for first in range(self.n_views):
            for second in range(first + 1, self.n_views):
                jointly_observed = (
                    observed_mask[:, first] & observed_mask[:, second]
                )
                if jointly_observed.any():
                    similarity = safe_cosine_similarity(
                        completed[first][jointly_observed],
                        completed[second][jointly_observed],
                        dim=1,
                    )
                    losses.append(1.0 - similarity.mean())
        if not losses:
            return sample_mask.new_zeros(())
        return torch.stack(losses).mean()

    def _graph_consistency_loss(
        self,
        completed: Dict[int, torch.Tensor],
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_gnn:
            return completed[0].new_zeros(())

        relation_graph = self.graph_predictor.relation_graph
        features = [completed[view] for view in range(self.n_views)]
        _, adjacency = relation_graph.project_and_build_adjacency(
            features, sample_mask=sample_mask
        )
        losses: List[torch.Tensor] = []
        observed_mask = sample_mask > 0.5
        for first in range(self.n_views):
            for second in range(first + 1, self.n_views):
                jointly_observed = (
                    observed_mask[:, first] & observed_mask[:, second]
                )
                if not jointly_observed.any():
                    continue

                similarity = safe_cosine_similarity(
                    features[first], features[second], dim=1
                )
                dissimilarity = 1.0 - similarity
                learned_affinity = 0.5 * (
                    adjacency[:, first, second]
                    + adjacency[:, second, first]
                )

                # 特征平滑项不允许通过简单降低邻接权重来趋近于零。
                # detach 后，它只优化特征，不会驱动所有边关闭。
                smoothness = (
                    learned_affinity.detach()[jointly_observed]
                    * dissimilarity[jointly_observed]
                ).mean()

                # 邻接参数拟合由当前特征相似度产生的稳定目标；目标分支
                # detach，避免邻接与特征共同向全零平凡解移动。
                affinity_target = (
                    0.5 * (similarity.detach() + 1.0)
                ).clamp(0.0, 1.0)
                relation_calibration = F.mse_loss(
                    learned_affinity[jointly_observed],
                    affinity_target[jointly_observed],
                )
                losses.append(smoothness + relation_calibration)
        return torch.stack(losses).mean() if losses else features[0].new_zeros(())

    def training_loss(
        self,
        data_list: TensorList,
        sample_mask: torch.Tensor,
        labels: torch.Tensor,
        original_data_list: TensorList,
        lambda_impute: float,
        lambda_consist: float,
        lambda_graph: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        final, completed, confidences = self.forward_embeddings(
            data_list, sample_mask
        )
        multimodal_features = torch.cat(
            [final[view] for view in range(self.n_views)], dim=1
        )
        logits = self.classifier(multimodal_features)
        classification_loss = self.criterion(logits, labels)
        total_loss = classification_loss
        components = {"main_clf": float(classification_loss.detach().item())}

        if not self.fast_path:
            original_features = self.get_original_features(original_data_list)
            imputation = self._imputation_loss(
                completed,
                confidences,
                sample_mask,
                original_features,
            )
            total_loss = total_loss + lambda_impute * imputation
            components["imputation"] = float(imputation.detach().item())

        consistency = self._modality_consistency_loss(completed, sample_mask)
        total_loss = total_loss + lambda_consist * consistency
        components["consistency"] = float(consistency.detach().item())

        if self.use_gnn:
            graph_consistency = self._graph_consistency_loss(
                completed, sample_mask
            )
            total_loss = total_loss + lambda_graph * graph_consistency
            components["graph_consistency"] = float(
                graph_consistency.detach().item()
            )

        components["total"] = float(total_loss.detach().item())
        return total_loss, logits, components

    def infer(
        self, data_list: TensorList, sample_mask: torch.Tensor
    ) -> torch.Tensor:
        final, _, _ = self.forward_embeddings(data_list, sample_mask)
        features = torch.cat(
            [final[view] for view in range(self.n_views)], dim=1
        )
        return self.classifier(features)

    def forward(
        self, data_list: TensorList, sample_mask: torch.Tensor
    ) -> torch.Tensor:
        """标准 PyTorch 前向接口，返回未经过 Softmax 的分类 logits。"""
        return self.infer(data_list, sample_mask)

