# -*- coding: utf-8 -*-

"""
NOVA-SGL: Novelty-Aware Social Graph Learning for Recommendation

Full RecBole implementation for social recommendation datasets such as LastFM.

Main components:
    1. User-item graph propagation
    2. Social user-user graph propagation
    3. Collaborative-social representation fusion
    4. Personalized global/social novelty estimation
    5. User-specific novelty gate
    6. Popularity-bias mitigation
    7. Symmetric social contrastive learning
    8. Pairwise BPR optimization
    9. Full-sort prediction for ranking evaluation

Designed for standard RecBole, not RecBole-GNN.

Example LastFM fields:
    USER_ID_FIELD = "user_id"
    ITEM_ID_FIELD = "artist_id"
    social_file = ".../lastfm.net"
"""

import csv
import os
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class NOVASGL(GeneralRecommender):
    r"""
    NOVA-SGL: Novelty-Aware Social Graph Learning for Recommendation.

    This implementation supports both:
        1. non-social datasets such as MovieLens,
        2. social datasets such as LastFM, Epinions, FilmTrust, Yelp-social.

    If a valid social graph is provided, the model activates:
        - social graph propagation,
        - social exposure novelty,
        - social contrastive loss.

    If no social graph is available, the model automatically falls back to:
        - user-item graph propagation,
        - global novelty,
        - novelty gate,
        - popularity penalty.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NOVASGL, self).__init__(config, dataset)

        # ------------------------------------------------------------
        # Basic RecBole fields
        # ------------------------------------------------------------
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID

        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.device = config["device"]

        # ------------------------------------------------------------
        # Hyperparameters
        # ------------------------------------------------------------
        self.embedding_size = int(self._get_config(config, "embedding_size", 64))

        # User-item graph propagation layers
        self.n_layers = int(self._get_config(config, "n_layers", 2))

        # Social graph propagation layers
        self.social_layers = int(self._get_config(config, "social_layers", 1))

        # L2 regularization
        self.reg_weight = float(self._get_config(config, "reg_weight", 1e-5))

        # Novelty-aware ranking
        self.novelty_weight = float(self._get_config(config, "novelty_weight", 0.10))
        self.pop_penalty_weight = float(
            self._get_config(config, "pop_penalty_weight", 0.05)
        )

        # Global/social novelty mixture
        self.eta_global = float(self._get_config(config, "eta_global", 0.5))
        self.eta_social = float(self._get_config(config, "eta_social", 0.5))

        eta_sum = max(self.eta_global + self.eta_social, 1e-12)
        self.eta_global = self.eta_global / eta_sum
        self.eta_social = self.eta_social / eta_sum

        # Social contrastive learning
        self.cl_weight = float(self._get_config(config, "cl_weight", 0.05))
        self.cl_temp = float(self._get_config(config, "cl_temp", 0.2))
        self.disable_cl_when_no_social = bool(
            self._get_config(config, "disable_cl_when_no_social", True)
        )

        # Novelty gate
        self.gate_hidden_size = int(self._get_config(config, "gate_hidden_size", 32))
        self.long_tail_quantile = float(
            self._get_config(config, "long_tail_quantile", 0.80)
        )

        # Social graph options
        self.social_file = self._get_config(config, "social_file", None)
        self.social_undirected = bool(self._get_config(config, "social_undirected", True))
        self.add_social_self_loop = bool(
            self._get_config(config, "add_social_self_loop", True)
        )

        # Debug print flag
        self.debug_social = bool(self._get_config(config, "debug_social", False))

        # ------------------------------------------------------------
        # Trainable parameters
        # ------------------------------------------------------------
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Learnable fusion coefficient:
        # rho = sigmoid(rho_logit)
        # h_u = rho * z_u^c + (1-rho) * z_u^s
        self.rho_logit = nn.Parameter(torch.tensor(0.0))

        # Gate input = fused user embedding + 3 scalar user descriptors:
        #   user long-tail ratio,
        #   normalized social degree,
        #   user sparsity.
        gate_input_size = self.embedding_size + 3

        self.novelty_gate = nn.Sequential(
            nn.Linear(gate_input_size, self.gate_hidden_size),
            nn.ReLU(),
            nn.Linear(self.gate_hidden_size, 1),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------
        # Build interaction graph, social graph, and novelty statistics
        # ------------------------------------------------------------
        self.inter_csr = self._build_interaction_csr(dataset)

        self.norm_adj_matrix = self._build_user_item_norm_adj(dataset).to(self.device)

        social_binary_csr, social_available = self._build_social_binary_csr(dataset)
        self.social_available = social_available

        self.norm_social_matrix = self._build_social_norm_adj(social_binary_csr).to(
            self.device
        )

        self.social_exposure_csr = self._build_social_exposure_csr(
            social_binary_csr, self.inter_csr
        )

        (
            item_pop_norm,
            item_global_novelty,
            user_longtail_ratio,
            user_social_degree_norm,
            user_sparsity,
        ) = self._build_novelty_statistics(dataset, social_binary_csr)

        self.register_buffer("item_pop_norm", item_pop_norm.to(self.device))
        self.register_buffer("item_global_novelty", item_global_novelty.to(self.device))
        self.register_buffer("user_longtail_ratio", user_longtail_ratio.to(self.device))
        self.register_buffer(
            "user_social_degree_norm", user_social_degree_norm.to(self.device)
        )
        self.register_buffer("user_sparsity", user_sparsity.to(self.device))

        if self.debug_social:
            print("=" * 80)
            print("NOVA-SGL social debug")
            print("=" * 80)
            print("USER_ID_FIELD:", self.USER_ID)
            print("ITEM_ID_FIELD:", self.ITEM_ID)
            print("n_users:", self.n_users)
            print("n_items:", self.n_items)
            print("social_file:", self.social_file)
            print("social_available:", self.social_available)
            print("social_edges:", social_binary_csr.nnz)
            print("social_exposure_nnz:", self.social_exposure_csr.nnz)
            print("=" * 80)

        # Initialize trainable weights
        self.apply(xavier_normal_initialization)

    # ============================================================
    # Config helper
    # ============================================================
    @staticmethod
    def _get_config(config, key, default):
        try:
            return config[key]
        except Exception:
            return default

    # ============================================================
    # Interaction graph
    # ============================================================
    def _build_interaction_csr(self, dataset) -> sp.csr_matrix:
        """
        Build binary user-item interaction matrix R.

        Shape:
            [n_users, n_items]
        """

        inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        inter_matrix.data = np.ones_like(inter_matrix.data, dtype=np.float32)

        inter_csr = inter_matrix.tocsr()
        inter_csr.eliminate_zeros()

        return inter_csr

    def _build_user_item_norm_adj(self, dataset) -> torch.Tensor:
        """
        Build normalized user-item graph adjacency:

            A_hat = D^{-1/2} A D^{-1/2}

        Shape:
            [n_users + n_items, n_users + n_items]
        """

        inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        inter_matrix.data = np.ones_like(inter_matrix.data, dtype=np.float32)

        n_nodes = self.n_users + self.n_items

        # User -> item
        rows_ui = inter_matrix.row
        cols_ui = inter_matrix.col + self.n_users

        # Item -> user
        rows_iu = inter_matrix.col + self.n_users
        cols_iu = inter_matrix.row

        rows = np.concatenate([rows_ui, rows_iu])
        cols = np.concatenate([cols_ui, cols_iu])
        data = np.ones(len(rows), dtype=np.float32)

        adj = sp.coo_matrix(
            (data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32
        ).tocsr()

        norm_adj = self._symmetric_normalize(adj)

        return self._sp_mat_to_torch_sparse_tensor(norm_adj)

    # ============================================================
    # Social graph
    # ============================================================
    def _build_social_binary_csr(self, dataset) -> Tuple[sp.csr_matrix, bool]:
        """
        Build binary user-user social matrix S.

        Priority:
            1. Try dataset.net_graph(form='coo') if available.
            2. Try config['social_file'].
            3. Use empty social graph.

        For LastFM from RUCAIBox/Social-Datasets:
            lastfm.net header:
                source_id:token    target_id:token

        In the runner, use:
            "social_file": r"C:\\NOVA_SGL_RecBole\\data\\lastfm\\lastfm.net"
        """

        social_csr = None
        social_available = False

        # --------------------------------------------------------
        # 1. Try RecBole's SocialDataset graph if available
        # --------------------------------------------------------
        if hasattr(dataset, "net_graph"):
            try:
                social_graph = dataset.net_graph(form="coo").astype(np.float32)
                social_graph.data = np.ones_like(social_graph.data, dtype=np.float32)
                social_csr = social_graph.tocsr()
                social_available = social_csr.nnz > 0
            except Exception:
                social_csr = None
                social_available = False

        # --------------------------------------------------------
        # 2. Fallback: manually load social_file
        # --------------------------------------------------------
        if (social_csr is None or social_csr.nnz == 0) and self.social_file is not None:
            social_csr = self._load_social_file_as_csr(dataset, self.social_file)
            social_available = social_csr.nnz > 0

        # --------------------------------------------------------
        # 3. Empty graph if no social source exists
        # --------------------------------------------------------
        if social_csr is None:
            social_csr = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
            social_available = False

        social_csr = social_csr.astype(np.float32)

        # Remove self-loops before optional undirected conversion.
        social_csr.setdiag(0.0)
        social_csr.eliminate_zeros()

        # Convert to undirected if needed.
        if self.social_undirected:
            social_csr = social_csr.maximum(social_csr.transpose()).tocsr()

        social_csr.data = np.ones_like(social_csr.data, dtype=np.float32)
        social_csr.eliminate_zeros()

        return social_csr, social_available

    def _load_social_file_as_csr(self, dataset, social_file: str) -> sp.csr_matrix:
        """
        Load external .net file into an internal-id user-user sparse matrix.

        Expected header:
            source_id:token    target_id:token

        Important:
            Raw user tokens must be mapped to RecBole internal user IDs.
        """

        if not os.path.exists(social_file):
            raise FileNotFoundError(f"NOVA-SGL social_file does not exist: {social_file}")

        token_to_id = self._get_token_to_id_map(dataset, self.USER_ID)

        rows = []
        cols = []

        with open(social_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")

            # Skip header
            _ = next(reader, None)

            for line in reader:
                if len(line) < 2:
                    continue

                src_raw = str(line[0]).strip()
                tgt_raw = str(line[1]).strip()

                src = self._map_token_to_internal_id(src_raw, token_to_id)
                tgt = self._map_token_to_internal_id(tgt_raw, token_to_id)

                if src is None or tgt is None:
                    continue

                if 0 <= src < self.n_users and 0 <= tgt < self.n_users and src != tgt:
                    rows.append(src)
                    cols.append(tgt)

        data = np.ones(len(rows), dtype=np.float32)

        social_csr = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_users),
            dtype=np.float32,
        ).tocsr()

        social_csr.eliminate_zeros()

        return social_csr

    def _get_token_to_id_map(self, dataset, field: str):
        """
        Robustly obtain RecBole's raw-token to internal-id mapping.
        """

        token_to_id = None

        if hasattr(dataset, "field2token_id"):
            try:
                token_to_id = dataset.field2token_id[field]
            except Exception:
                token_to_id = None

        return token_to_id

    @staticmethod
    def _map_token_to_internal_id(raw_token: str, token_to_id):
        """
        Map raw token from .net file to RecBole internal ID.

        RecBole usually stores field2token_id[field] as a dict-like object.
        This function is intentionally defensive for different RecBole versions.
        """

        if token_to_id is not None:
            # Try string key
            try:
                if raw_token in token_to_id:
                    return int(token_to_id[raw_token])
            except Exception:
                pass

            # Try integer key
            try:
                raw_int = int(raw_token)
                if raw_int in token_to_id:
                    return int(token_to_id[raw_int])
            except Exception:
                pass

            # Try direct indexing
            try:
                return int(token_to_id[raw_token])
            except Exception:
                pass

        # Last fallback:
        # This works only if raw IDs already match RecBole internal IDs.
        try:
            return int(raw_token)
        except Exception:
            return None

    def _build_social_norm_adj(self, social_binary_csr: sp.csr_matrix) -> torch.Tensor:
        """
        Build normalized social adjacency:

            S_hat = D_s^{-1/2} S D_s^{-1/2}

        Self-loops are added only for propagation stability, not for social exposure.
        """

        if self.add_social_self_loop:
            social_matrix = social_binary_csr + sp.eye(
                self.n_users, dtype=np.float32, format="csr"
            )
        else:
            social_matrix = social_binary_csr.copy()

        social_matrix = social_matrix.astype(np.float32)
        social_matrix.eliminate_zeros()

        norm_social = self._symmetric_normalize(social_matrix)

        return self._sp_mat_to_torch_sparse_tensor(norm_social)

    def _build_social_exposure_csr(
        self, social_binary_csr: sp.csr_matrix, inter_csr: sp.csr_matrix
    ) -> sp.csr_matrix:
        """
        Compute social exposure matrix:

            E_social = D_s^{-1} S R

        where:
            S: user-user social adjacency
            R: user-item interaction matrix

        E_social[u, i] approximates the fraction of user u's social neighbors
        that interacted with item i.
        """

        degree = np.asarray(social_binary_csr.sum(axis=1)).reshape(-1)

        inv_degree = np.zeros_like(degree, dtype=np.float32)
        nonzero = degree > 0
        inv_degree[nonzero] = 1.0 / degree[nonzero]

        d_inv = sp.diags(inv_degree)

        row_norm_social = d_inv.dot(social_binary_csr).tocsr()
        exposure = row_norm_social.dot(inter_csr).tocsr()

        if exposure.nnz > 0:
            exposure.data = np.clip(exposure.data, 0.0, 1.0)

        exposure.eliminate_zeros()

        return exposure

    # ============================================================
    # Novelty statistics
    # ============================================================
    def _build_novelty_statistics(
        self, dataset, social_binary_csr: sp.csr_matrix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute:
            item_pop_norm:
                normalized item popularity p_i in [0,1]

            item_global_novelty:
                n_i^g = 1 - p_i

            user_longtail_ratio:
                fraction of user history belonging to long-tail items

            user_social_degree_norm:
                normalized social degree

            user_sparsity:
                1 - normalized interaction count
        """

        inter_feat = dataset.inter_feat

        user_ids = inter_feat[self.USER_ID].cpu().long()
        item_ids = inter_feat[self.ITEM_ID].cpu().long()

        item_pop = torch.bincount(item_ids, minlength=self.n_items).float()
        user_count = torch.bincount(user_ids, minlength=self.n_users).float()

        # Item popularity in [0,1]
        max_pop = item_pop.max().clamp(min=1.0)
        item_pop_norm = item_pop / max_pop

        # Global inverse-popularity novelty
        item_global_novelty = 1.0 - item_pop_norm

        # Long-tail items defined by popularity quantile
        threshold = torch.quantile(item_pop, self.long_tail_quantile)
        long_tail_mask = item_pop <= threshold

        interacted_longtail = long_tail_mask[item_ids].float()

        user_longtail_count = torch.zeros(self.n_users)
        user_longtail_count.index_add_(0, user_ids, interacted_longtail)

        user_longtail_ratio = user_longtail_count / user_count.clamp(min=1.0)

        # Social degree in [0,1]
        social_degree = np.asarray(social_binary_csr.sum(axis=1)).reshape(-1)
        max_social_degree = max(float(social_degree.max()), 1.0)
        user_social_degree_norm = torch.FloatTensor(social_degree / max_social_degree)

        # Sparsity indicator:
        # Higher means fewer interactions.
        max_user_count = user_count.max().clamp(min=1.0)
        user_activity_norm = user_count / max_user_count
        user_sparsity = 1.0 - user_activity_norm

        return (
            item_pop_norm,
            item_global_novelty,
            user_longtail_ratio,
            user_social_degree_norm,
            user_sparsity,
        )

    # ============================================================
    # Sparse matrix utilities
    # ============================================================
    @staticmethod
    def _symmetric_normalize(matrix: sp.csr_matrix) -> sp.coo_matrix:
        """
        Symmetric normalization:
            D^{-1/2} A D^{-1/2}

        Handles zero-degree rows safely.
        """

        matrix = matrix.tocsr()

        rowsum = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float32)

        d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
        nonzero = rowsum > 0.0
        d_inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)

        d_mat = sp.diags(d_inv_sqrt)

        norm_matrix = d_mat.dot(matrix).dot(d_mat).tocoo()
        norm_matrix.eliminate_zeros()

        return norm_matrix    

    @staticmethod
    def _sp_mat_to_torch_sparse_tensor(matrix: sp.coo_matrix) -> torch.Tensor:
        matrix = matrix.tocoo().astype(np.float32)

        indices = torch.LongTensor(np.vstack((matrix.row, matrix.col)))
        values = torch.FloatTensor(matrix.data)
        shape = torch.Size(matrix.shape)

        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    # ============================================================
    # Graph propagation
    # ============================================================
    def forward(self):
        """
        Returns:
            fused_user_embeddings:
                h_u = rho * z_u^c + (1-rho) * z_u^s

            collab_user_embeddings:
                z_u^c from user-item graph

            item_embeddings:
                z_i^c from user-item graph

            social_user_embeddings:
                z_u^s from social graph
        """

        # --------------------------------------------------------
        # User-item graph propagation
        # --------------------------------------------------------
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        all_ui_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj_matrix, ego_embeddings)
            all_ui_embeddings.append(ego_embeddings)

        all_ui_embeddings = torch.stack(all_ui_embeddings, dim=1)
        final_ui_embeddings = torch.mean(all_ui_embeddings, dim=1)

        collab_user_embeddings, item_embeddings = torch.split(
            final_ui_embeddings, [self.n_users, self.n_items]
        )

        # --------------------------------------------------------
        # Social graph propagation
        # --------------------------------------------------------
        social_embeddings = self.user_embedding.weight
        all_social_embeddings = [social_embeddings]

        for _ in range(self.social_layers):
            social_embeddings = torch.sparse.mm(
                self.norm_social_matrix, social_embeddings
            )
            all_social_embeddings.append(social_embeddings)

        all_social_embeddings = torch.stack(all_social_embeddings, dim=1)
        social_user_embeddings = torch.mean(all_social_embeddings, dim=1)

        # --------------------------------------------------------
        # Collaborative-social fusion
        # --------------------------------------------------------
        rho = torch.sigmoid(self.rho_logit)

        fused_user_embeddings = (
            rho * collab_user_embeddings + (1.0 - rho) * social_user_embeddings
        )

        return (
            fused_user_embeddings,
            collab_user_embeddings,
            item_embeddings,
            social_user_embeddings,
        )

    # ============================================================
    # Gate and novelty scoring
    # ============================================================
    def compute_gate(
        self, users: torch.Tensor, fused_user_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        User-specific novelty gate lambda_u in [0,1].
        """

        longtail = self.user_longtail_ratio[users].unsqueeze(1)
        social_degree = self.user_social_degree_norm[users].unsqueeze(1)
        sparsity = self.user_sparsity[users].unsqueeze(1)

        gate_input = torch.cat(
            [fused_user_embeddings, longtail, social_degree, sparsity], dim=1
        )

        gate = self.novelty_gate(gate_input).squeeze(1)

        return gate
    
    def _social_exposure_for_pairs(
        self, users: torch.Tensor, items: torch.Tensor
        ) -> torch.Tensor:
        """
        Return social exposure E_social[u, i] for sampled user-item pairs.

        This version avoids SciPy sparse advanced-indexing problems on Windows
        and newer SciPy versions by explicitly copying NumPy arrays.
        """

        users_np = np.array(
            users.detach().cpu().numpy(), dtype=np.int64, copy=True
        ).reshape(-1)

        items_np = np.array(
            items.detach().cpu().numpy(), dtype=np.int64, copy=True
        ).reshape(-1)

        exposure = self.social_exposure_csr[users_np, items_np]

        exposure = np.asarray(exposure).reshape(-1).astype(np.float32, copy=True)

        return torch.tensor(exposure, dtype=torch.float32, device=self.device)

    def _social_exposure_for_full_sort(self, users: torch.Tensor) -> torch.Tensor:
        """
        Return social exposure matrix E_social[u, :] for full-sort prediction.
        """

        users_np = np.array(
            users.detach().cpu().numpy(), dtype=np.int64, copy=True
        ).reshape(-1)

        exposure = self.social_exposure_csr[users_np, :].toarray()
        exposure = np.asarray(exposure, dtype=np.float32)

        return torch.tensor(exposure, dtype=torch.float32, device=self.device)

    def compute_pair_scores(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        fused_user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NOVA-SGL pairwise score:

            y_ui = relevance
                   + beta * lambda_u * novelty_ui
                   - mu * popularity_i
        """

        u_e = fused_user_embeddings[users]
        i_e = item_embeddings[items]

        relevance_score = torch.sum(u_e * i_e, dim=1)

        gate = self.compute_gate(users, u_e)

        global_novelty = self.item_global_novelty[items]

        social_exposure = self._social_exposure_for_pairs(users, items)
        social_novelty = 1.0 - social_exposure

        personalized_novelty = (
            self.eta_global * global_novelty + self.eta_social * social_novelty
        )

        popularity_penalty = self.item_pop_norm[items]

        final_score = (
            relevance_score
            + self.novelty_weight * gate * personalized_novelty
            - self.pop_penalty_weight * popularity_penalty
        )

        return final_score

    # ============================================================
    # Social contrastive loss
    # ============================================================
    def social_contrastive_loss(
        self,
        users: torch.Tensor,
        collab_user_embeddings: torch.Tensor,
        social_user_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symmetric contrastive alignment between:
            collaborative user view z_u^c
            social user view z_u^s
        """

        if self.cl_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)

        if (not self.social_available) and self.disable_cl_when_no_social:
            return torch.tensor(0.0, device=self.device)

        unique_users = torch.unique(users)

        if unique_users.shape[0] <= 1:
            return torch.tensor(0.0, device=self.device)

        z_c = collab_user_embeddings[unique_users]
        z_s = social_user_embeddings[unique_users]

        z_c = F.normalize(z_c, dim=1)
        z_s = F.normalize(z_s, dim=1)

        logits_c2s = torch.matmul(z_c, z_s.transpose(0, 1)) / self.cl_temp
        logits_s2c = torch.matmul(z_s, z_c.transpose(0, 1)) / self.cl_temp

        labels = torch.arange(unique_users.shape[0], device=self.device)

        loss_c2s = F.cross_entropy(logits_c2s, labels)
        loss_s2c = F.cross_entropy(logits_s2c, labels)

        loss = 0.5 * (loss_c2s + loss_s2c)

        return loss

    # ============================================================
    # Training loss
    # ============================================================
    def calculate_loss(self, interaction):
        """
        BPR ranking loss + social contrastive loss + L2 regularization.
        """

        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        (
            fused_user_embeddings,
            collab_user_embeddings,
            item_embeddings,
            social_user_embeddings,
        ) = self.forward()

        pos_scores = self.compute_pair_scores(
            users, pos_items, fused_user_embeddings, item_embeddings
        )

        neg_scores = self.compute_pair_scores(
            users, neg_items, fused_user_embeddings, item_embeddings
        )

        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        cl_loss = self.social_contrastive_loss(
            users, collab_user_embeddings, social_user_embeddings
        )

        user_e = self.user_embedding(users)
        pos_e = self.item_embedding(pos_items)
        neg_e = self.item_embedding(neg_items)

        reg_loss = (
            user_e.norm(2).pow(2)
            + pos_e.norm(2).pow(2)
            + neg_e.norm(2).pow(2)
        ) / users.shape[0]

        loss = bpr_loss + self.cl_weight * cl_loss + self.reg_weight * reg_loss

        return loss

    # ============================================================
    # Pairwise prediction
    # ============================================================
    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]

        fused_user_embeddings, _, item_embeddings, _ = self.forward()

        scores = self.compute_pair_scores(
            users, items, fused_user_embeddings, item_embeddings
        )

        return scores

    # ============================================================
    # Full-sort prediction
    # ============================================================
    def full_sort_predict(self, interaction):
        """
        Full-sort prediction for all candidate items.
        Required for RecBole full ranking evaluation.
        """

        users = interaction[self.USER_ID]

        fused_user_embeddings, _, item_embeddings, _ = self.forward()

        u_e = fused_user_embeddings[users]

        relevance_scores = torch.matmul(u_e, item_embeddings.transpose(0, 1))

        gate = self.compute_gate(users, u_e).unsqueeze(1)

        global_novelty = self.item_global_novelty.unsqueeze(0)

        social_exposure = self._social_exposure_for_full_sort(users)
        social_novelty = 1.0 - social_exposure

        personalized_novelty = (
            self.eta_global * global_novelty + self.eta_social * social_novelty
        )

        popularity_penalty = self.item_pop_norm.unsqueeze(0)

        final_scores = (
            relevance_scores
            + self.novelty_weight * gate * personalized_novelty
            - self.pop_penalty_weight * popularity_penalty
        )

        return final_scores.view(-1)