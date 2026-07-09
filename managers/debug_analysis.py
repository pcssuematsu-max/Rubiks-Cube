"""Debug and viewer analysis helpers."""

from __future__ import annotations

import numpy as np

class DebugAnalysisManager:
    """AIの内部状態や評価結果を確認するための診断処理を担当する。"""

    def __init__(self, frame):
        self.frame = frame
        self.grad_index = frame.grad_index
        self.grad_mode = frame.grad_mode
        self.grad_layer = frame.grad_layer
        self._last_occlusion_scores = []
        self._last_attention_scores = []
        self._last_attention_relation = None
        self._last_attention_mode = ''
        self._last_attention_error = ''
        self._last_embedding_scores = []
        self._last_embedding_mode = ''
        self._last_embedding_error = ''

    def update_viewer_settings(self, index_text, mode_text, layer_text):
        """UI入力値を読み取り、myviewerの表示対象設定を更新する。"""
        grad_index = self._parse_grad_index(index_text)
        if grad_index is None:
            return False
        self.grad_index = grad_index
        self.grad_mode = mode_text
        self.grad_layer = layer_text
        self._sync_frame_grad_settings()
        return True

    def show_current_viewer(self, ai_index, N):
        """現在のgrad設定に従って、正負の特徴Viewerを更新する。"""
        if self.grad_mode == "SVD":
            self.myviewer(ai_index,self.grad_index,N,SVD = True)
        elif self.grad_mode == "Grad":
            self.myviewer(ai_index,self.grad_index,N,Grad = True,layer = self.grad_layer)
        elif self.grad_mode == "IG":
            self.myviewer(ai_index,self.grad_index,N,IG = True,layer = self.grad_layer)
        elif self.grad_mode == "Contrast":
            self.myviewer(ai_index,self.grad_index,N,Contrast = True)
        elif self.grad_mode == "Occ":
            self.myviewer(ai_index,self.grad_index,N,Occ = True)
        elif self.grad_mode == "PieceOcc":
            self.myviewer(ai_index,self.grad_index,N,PieceOcc = True)
        elif self.grad_mode == "PolicyOcc":
            self.myviewer(ai_index,self.grad_index,N,PolicyOcc = True)
        elif self.grad_mode == "PiecePolicyOcc":
            self.myviewer(ai_index,self.grad_index,N,PiecePolicyOcc = True)
        elif self.grad_mode == "AttnIn":
            self.myviewer(ai_index,self.grad_index,N,AttnIn = True)
        elif self.grad_mode == "AttnOut":
            self.myviewer(ai_index,self.grad_index,N,AttnOut = True)
        elif self.grad_mode == "AttnCentral":
            self.myviewer(ai_index,self.grad_index,N,AttnCentral = True)
        elif self.grad_mode == "EmbNorm":
            self.myviewer(ai_index,self.grad_index,N,EmbNorm = True)
        elif self.grad_mode == "EmbPC1":
            self.myviewer(ai_index,self.grad_index,N,EmbPC1 = True)
        elif self.grad_mode == "W1":
            self.myviewer(ai_index,self.grad_index,N)

    def sum_and_var(self, index):
        """指定AIの各パラメータについて、合計・分散・最大最小・更新量を表示する。"""
        ai = self.frame.AIs[index]
        for key in ai.params.keys():
            print(index,key)
            print("sum:",np.sum(ai.params[key]),"var:",np.var(ai.params[key]))
            print("max:",np.max(ai.params[key]),"min:",np.min(ai.params[key]))
            print("vsum:",np.sum(ai.v[key]))

    def max_val(self, T0 = (), T1 = (), head = '', Top = True, Num = 1):
        """条件に合うmyperm候補をAIで評価し、上位または下位の候補を表示する。"""
        keys = self.frame.search_myperms(T0,T1,head)
        input_data,empty_input = self._build_myperm_inputs(keys)
        for index in range(self.frame.AInum):
            self._print_ai_value_ranking(index,keys,input_data,empty_input,Top,Num)
        self.frame.cube.reset()

    def normalize(self, index):
        """指定AIの重みスケールを整え、BatchNorm系パラメータを初期値に戻す。"""
        ai = self.frame.AIs[index]
        for key in ai.params.keys():
            self._normalize_param(ai,key)
        ai.mark_params_dirty()
        ai.set_perfect_val()

    def re_activate(self, index):
        """更新量が小さいユニットを検出し、バイアスと一部重みを再活性化する。"""
        ai = self.frame.AIs[index]
        for key in ai.params.keys():
            if self._is_reactivation_target(key):
                self._reactivate_param(ai,key)
        ai.mark_params_dirty()

    def show_transformer_attention_analysis(self, ai_index):
        """W1 token 化と Att1 attention をパーツ単位に戻して表示する。"""
        content = self.transformer_attention_analysis_text(ai_index)
        self.frame.show_analysis_text(f'transformer attention analysis (ai={ai_index})', content)

    def show_transformer_embedding_analysis(self, ai_index):
        """Aff1 が作る piece embedding の大きさ・類似関係を表示する。"""
        content = self.transformer_embedding_analysis_text(ai_index)
        self.frame.show_analysis_text(f'transformer embedding analysis (ai={ai_index})', content)

    def w1_embedding_projection(self, ai_index, method = 'PCA', point_mode = 'Feature columns', max_points = 600, perplexity = 20.0, iterations = 300, embedding_source = 'W1', pca_axes = (1,2)):
        """Project W1 or its attention transforms to two dimensions for inspection."""
        ai = self.frame.AIs[ai_index]
        if not hasattr(ai, 'params') or 'W1' not in ai.params:
            raise ValueError(f'AI {ai_index} has no params[\'W1\'].')
        w1 = np.asarray(ai.params['W1'], dtype = 'f')
        if w1.ndim != 2 or w1.shape[1] != self.frame.cube.ips:
            raise ValueError(
                f'W1 shape {w1.shape} is incompatible with cube input size {self.frame.cube.ips}.'
            )
        embedding_matrix, source_name = self._attention_embedding_source(ai, w1, embedding_source)

        normalized_method = method.strip().upper().replace('-', '')
        point_limit = max(2, int(max_points))
        if normalized_method == 'TSNE':
            point_limit = min(point_limit, 800)
        embeddings, labels, piece_types, feature_indices, solve_groups, correct_flags, feature_colors = self._w1_embedding_points(
            embedding_matrix,
            point_mode,
            point_limit,
        )
        if embeddings.shape[0] < 2:
            raise ValueError('At least two W1 embedding points are required.')
        full_metrics = self._embedding_analysis_metrics(embeddings, piece_types, solve_groups)

        if normalized_method == 'PCA':
            axes = tuple(int(axis) for axis in pca_axes)
            if len(axes) != 2 or axes[0] == axes[1] or min(axes) < 1:
                raise ValueError(f'PCA axes must be two different positive components: {pca_axes}')
            all_coordinates, explained = self._pca_projection(embeddings, dimensions = max(axes))
            coordinates = all_coordinates[:,[axes[0] - 1, axes[1] - 1]]
            method_name = 'PCA'
            method_detail = ', '.join(
                f'PC{axis}={float(explained[axis - 1]):.2%}' if axis <= explained.size else f'PC{axis}=n/a'
                for axis in axes
            )
        elif normalized_method == 'TSNE':
            axes = (1,2)
            coordinates, used_perplexity = self._tsne_projection(
                embeddings,
                perplexity = float(perplexity),
                iterations = max(100, int(iterations)),
            )
            explained = np.zeros((0,), dtype = 'f')
            method_name = 't-SNE'
            method_detail = f'perplexity={used_perplexity:.1f}, iterations={max(100, int(iterations))}'
        else:
            raise ValueError(f'Unknown projection method: {method}')

        return {
            'ai_index': ai_index,
            'model_type': 'Transformer' if getattr(ai, 'use_transformer_attention', False) else 'Affine',
            'embedding_source': source_name,
            'embedding_shape': embedding_matrix.shape,
            'method': method_name,
            'method_detail': method_detail,
            'pca_axes': axes,
            'point_mode': point_mode,
            'coordinates': coordinates,
            'embeddings': embeddings,
            'labels': labels,
            'piece_types': piece_types,
            'feature_indices': feature_indices,
            'solve_groups': solve_groups,
            'correct_flags': correct_flags,
            'feature_colors': feature_colors,
            'explained': explained,
            'separation': self._projection_type_separation(coordinates, piece_types),
            'group_separation': self._projection_type_separation(coordinates, solve_groups),
            'full_metrics': full_metrics,
        }

    def _attention_embedding_source(self, ai, w1, embedding_source):
        """Return W1 or a learned attention projection of every W1 column."""
        normalized = embedding_source.upper().replace(' ', '').replace('@', '')
        if normalized == 'W1':
            return w1, 'W1'
        source_keys = {
            'WQ1W1': 'WQ1',
            'WK1W1': 'WK1',
            'WV1W1': 'WV1',
        }
        if normalized not in source_keys:
            raise ValueError(f'Unknown embedding source: {embedding_source}')
        param_key = source_keys[normalized]
        if param_key not in ai.params:
            raise ValueError(f'AI does not expose {param_key}; use a Transformer AI or select W1.')
        projection = np.asarray(ai.params[param_key], dtype = 'f')
        if projection.ndim != 2 or projection.shape[1] != w1.shape[0]:
            raise ValueError(f'{param_key} shape {projection.shape} cannot be applied to W1 shape {w1.shape}.')
        return projection @ w1, f'{param_key} @ W1'

    def _w1_embedding_points(self, w1, point_mode, max_points):
        """Build W1 column vectors or per-piece means with piece metadata."""
        try:
            blocks = self._piece_feature_blocks()
        except AttributeError as error:
            raise ValueError('This puzzle does not expose piece-level W1 feature metadata.') from error
        records_by_piece = []
        use_centroids = point_mode.strip().lower().startswith('piece')
        current_data = np.asarray(self.frame.cube.makedata()).reshape(-1)
        perfect_data = np.asarray(self.frame.cube.perfect_data).reshape(-1)
        solve_groups = self._piece_solve_group_labels(blocks)
        for piece_index, (piece_name, mask) in enumerate(blocks):
            indices = np.flatnonzero(mask) if np.asarray(mask).dtype == bool else np.asarray(mask, dtype = int)
            indices = indices[(indices >= 0) & (indices < w1.shape[1])]
            if indices.size == 0:
                continue
            piece_type = self._embedding_piece_type(piece_name, indices)
            solve_group = self._embedding_solve_group(solve_groups[piece_index], indices)
            piece_is_correct = bool(np.array_equal(current_data[indices], perfect_data[indices]))
            target_local_indices = np.flatnonzero(perfect_data[indices] > 0.5)
            if target_local_indices.size > 0:
                target_local_index = int(target_local_indices[0])
                target_colors = self._w1_feature_colors(piece_type, int(indices[target_local_index]), target_local_index)
            else:
                target_colors = ()
            if use_centroids:
                records_by_piece.append([(
                    np.mean(w1[:,indices], axis = 1),
                    piece_name,
                    piece_type,
                    -1,
                    solve_group,
                    piece_is_correct,
                    target_colors,
                )])
                continue
            piece_records = []
            for local_index, feature_index in enumerate(indices):
                colors = self._w1_feature_colors(piece_type, int(feature_index), local_index)
                value_label = self._w1_feature_value_label(colors, int(feature_index))
                piece_records.append((
                    w1[:,feature_index],
                    f'{piece_name} / {value_label}',
                    piece_type,
                    int(feature_index),
                    solve_group,
                    bool(piece_is_correct and perfect_data[feature_index] > 0.5),
                    colors,
                ))
            records_by_piece.append(piece_records)

        records = self._sample_w1_embedding_records(records_by_piece, max_points)
        if not records:
            return (
                np.zeros((0, w1.shape[0]), dtype = 'f'),
                [],
                [],
                np.zeros((0,), dtype = int),
                [],
                np.zeros((0,), dtype = bool),
                [],
            )
        return (
            np.asarray([record[0] for record in records], dtype = 'f'),
            [record[1] for record in records],
            [record[2] for record in records],
            np.asarray([record[3] for record in records], dtype = int),
            [record[4] for record in records],
            np.asarray([record[5] for record in records], dtype = bool),
            [record[6] for record in records],
        )

    def _piece_solve_group_labels(self, blocks):
        """Map each piece block to the most specific descriptive cube group."""
        group_masks = []
        for group_name, group_vector in getattr(self.frame.cube, 'group_val', {}).items():
            name = str(group_name)
            if len(name) <= 1:
                continue
            group_masks.append((name, np.asarray(group_vector).reshape(-1) > 0))
        labels = []
        for piece_name, piece_mask in blocks:
            piece_indices = np.flatnonzero(piece_mask)
            best_name = piece_name.split('-', 1)[0]
            best_overlap = 0
            for group_name, group_mask in group_masks:
                valid_indices = piece_indices[piece_indices < group_mask.size]
                overlap = int(np.count_nonzero(group_mask[valid_indices]))
                if overlap > best_overlap:
                    best_name = group_name
                    best_overlap = overlap
            labels.append(best_name)
        return labels

    def _embedding_piece_type(self, default_name, feature_indices):
        cube = self.frame.cube
        if feature_indices.size > 0 and hasattr(cube, 'embedding_piece_type'):
            return str(cube.embedding_piece_type(int(feature_indices[0])))
        return default_name.split('-', 1)[0]

    def _embedding_solve_group(self, default_name, feature_indices):
        cube = self.frame.cube
        if feature_indices.size > 0 and hasattr(cube, 'embedding_solve_group'):
            return str(cube.embedding_solve_group(int(feature_indices[0])))
        return default_name

    def _sample_w1_embedding_records(self, records_by_piece, max_points):
        """Sample deterministically while retaining at least one point per piece."""
        all_records = [record for piece_records in records_by_piece for record in piece_records]
        if len(all_records) <= max_points:
            return all_records
        selected_ids = set()
        if len(records_by_piece) <= max_points:
            for piece_records in records_by_piece:
                correct_records = [record for record in piece_records if record[5]]
                selected_record = correct_records[0] if correct_records else piece_records[len(piece_records) // 2]
                selected_ids.add(id(selected_record))
        remaining = [record for record in all_records if id(record) not in selected_ids]
        remaining_count = max_points - len(selected_ids)
        if remaining_count > 0:
            sample_indices = np.linspace(0, len(remaining) - 1, remaining_count, dtype = int)
            selected_ids.update(id(remaining[index]) for index in sample_indices)
        return [record for record in all_records if id(record) in selected_ids][:max_points]

    def _w1_feature_colors(self, piece_type, feature_index, local_index):
        """Return the color symbols represented by one makedata feature."""
        cube = self.frame.cube
        if hasattr(cube, 'embedding_feature_colors'):
            return tuple(str(value) for value in cube.embedding_feature_colors(feature_index))
        feature_map = getattr(cube, 'feature_index_to_piece_color', {})
        if feature_index in feature_map:
            color = feature_map[feature_index][1]
            if isinstance(color, str):
                return (color,) if color in getattr(cube, 'colors', ()) else tuple(color)
            return tuple(str(value) for value in color)
        type_name = piece_type.lower()
        if 'corner' in type_name and local_index < len(getattr(cube, 'corner_colors', ())):
            return tuple(str(cube.corner_colors[local_index]))
        if 'edge' in type_name and local_index < len(getattr(cube, 'edge_colors', ())):
            return tuple(str(cube.edge_colors[local_index]))
        if 'center' in type_name and local_index < len(getattr(cube, 'colors', ())):
            return (str(cube.colors[local_index]),)
        return ()

    def _w1_feature_value_label(self, colors, feature_index):
        """Return full color names for one makedata feature."""
        if colors:
            return '/'.join(self._color_display_name(color) for color in colors)
        cube = self.frame.cube
        if hasattr(cube, 'embedding_feature_value_label'):
            return str(cube.embedding_feature_value_label(feature_index))
        return f'feature={feature_index}'

    def _color_display_name(self, color):
        return {
            'R': 'Red',
            'O': 'Orange',
            'Y': 'Yellow',
            'W': 'White',
            'G': 'Green',
            'B': 'Blue',
            'X': 'Masked',
        }.get(str(color), str(color))

    def _pca_projection(self, embeddings, dimensions = 2):
        """Return PCA scores and explained variance ratios using NumPy only."""
        x = np.asarray(embeddings, dtype = np.float64)
        centered = x - np.mean(x, axis = 0, keepdims = True)
        try:
            u, singular_values, _ = np.linalg.svd(centered, full_matrices = False)
        except np.linalg.LinAlgError as error:
            raise ValueError(f'PCA failed: {error}') from error
        component_count = min(dimensions, singular_values.size)
        coordinates = np.zeros((x.shape[0], dimensions), dtype = 'f')
        if component_count > 0:
            coordinates[:,:component_count] = (u[:,:component_count] * singular_values[:component_count]).astype('f')
        variances = singular_values ** 2
        total = float(np.sum(variances))
        explained = np.zeros_like(singular_values, dtype = 'f')
        if total > 1.0e-12:
            explained = (variances / total).astype('f')
        return coordinates, explained

    def _tsne_projection(self, embeddings, perplexity = 20.0, iterations = 300):
        """Run a small exact t-SNE suitable for W1 analysis without sklearn."""
        x = np.asarray(embeddings, dtype = np.float64)
        point_count = x.shape[0]
        if point_count < 3:
            coordinates, _ = self._pca_projection(x, dimensions = 2)
            return coordinates, 1.0
        pca_dimensions = min(50, x.shape[1], point_count - 1)
        x, _ = self._pca_projection(x, dimensions = pca_dimensions)
        x = x.astype(np.float64)
        used_perplexity = min(max(2.0, perplexity), max(2.0, (point_count - 1) / 3.0))
        probabilities = self._tsne_joint_probabilities(x, used_perplexity)

        y, _ = self._pca_projection(x, dimensions = 2)
        y = y.astype(np.float64)
        y_scale = float(np.std(y))
        if y_scale < 1.0e-12:
            rng = np.random.default_rng(0)
            y = rng.normal(0.0, 1.0e-4, size = (point_count, 2))
        else:
            y *= 1.0e-4 / y_scale
        velocity = np.zeros_like(y)
        gains = np.ones_like(y)
        learning_rate = max(100.0, point_count / 4.0)

        for iteration in range(iterations):
            squared_distances = self._pairwise_squared_distances(y)
            numerator = 1.0 / (1.0 + squared_distances)
            np.fill_diagonal(numerator, 0.0)
            q = numerator / max(float(np.sum(numerator)), 1.0e-12)
            p = probabilities * (4.0 if iteration < 100 else 1.0)
            weighted = (p - q) * numerator
            gradient = 4.0 * (
                np.sum(weighted, axis = 1, keepdims = True) * y - weighted @ y
            )
            different_sign = (gradient > 0.0) != (velocity > 0.0)
            gains = np.where(different_sign, gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, 0.01)
            momentum = 0.5 if iteration < 100 else 0.8
            velocity = momentum * velocity - learning_rate * gains * gradient
            y += velocity
            y -= np.mean(y, axis = 0, keepdims = True)
        return y.astype('f'), used_perplexity

    def _tsne_joint_probabilities(self, x, perplexity):
        """Compute symmetric high-dimensional t-SNE probabilities."""
        distances = self._pairwise_squared_distances(x)
        point_count = x.shape[0]
        conditional = np.zeros((point_count, point_count), dtype = np.float64)
        target_entropy = np.log(perplexity)
        for row in range(point_count):
            mask = np.arange(point_count) != row
            row_distances = distances[row,mask]
            beta = 1.0
            beta_min = -np.inf
            beta_max = np.inf
            for _ in range(50):
                values = np.exp(-row_distances * beta)
                value_sum = max(float(np.sum(values)), 1.0e-300)
                entropy = np.log(value_sum) + beta * float(np.sum(row_distances * values)) / value_sum
                difference = entropy - target_entropy
                if abs(difference) < 1.0e-5:
                    break
                if difference > 0.0:
                    beta_min = beta
                    beta = beta * 2.0 if np.isinf(beta_max) else (beta + beta_max) / 2.0
                else:
                    beta_max = beta
                    beta = beta / 2.0 if np.isinf(beta_min) else (beta + beta_min) / 2.0
            conditional[row,mask] = values / value_sum
        joint = (conditional + conditional.T) / (2.0 * point_count)
        return np.maximum(joint, 1.0e-12)

    def _pairwise_squared_distances(self, x):
        norms = np.sum(x * x, axis = 1, keepdims = True)
        return np.maximum(norms + norms.T - 2.0 * (x @ x.T), 0.0)

    def _embedding_analysis_metrics(self, embeddings, piece_types, solve_groups):
        """Measure full-dimensional clustering against shuffled-label baselines."""
        distances = np.sqrt(self._pairwise_squared_distances(np.asarray(embeddings, dtype = np.float64)))
        return {
            'piece_type': self._embedding_cluster_metric(distances, piece_types, embeddings),
            'solve_group': self._embedding_cluster_metric(distances, solve_groups, embeddings),
        }

    def _embedding_cluster_metric(self, distances, labels, embeddings, permutation_count = 12):
        observed = self._silhouette_from_distances(distances, labels)
        rng = np.random.default_rng(12345)
        labels_array = np.asarray(labels, dtype = object)
        baseline = np.asarray([
            self._silhouette_from_distances(distances, rng.permutation(labels_array))
            for _ in range(permutation_count)
        ], dtype = 'f')
        baseline_mean = float(np.mean(baseline)) if baseline.size else 0.0
        baseline_std = float(np.std(baseline)) if baseline.size else 0.0
        z_score = (observed - baseline_mean) / max(baseline_std, 1.0e-8)
        separation = self._projection_type_separation(embeddings, labels)
        return {
            'silhouette': observed,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'z_score': z_score,
            'separation_ratio': separation['ratio'],
            'class_count': len(set(labels)),
        }

    def _silhouette_from_distances(self, distances, labels):
        """Return mean silhouette score from a precomputed distance matrix."""
        labels = np.asarray(labels, dtype = object)
        categories = list(dict.fromkeys(labels.tolist()))
        if len(categories) < 2 or distances.shape[0] < 2:
            return 0.0
        scores = np.zeros((distances.shape[0],), dtype = np.float64)
        for category in categories:
            own_indices = np.flatnonzero(labels == category)
            if own_indices.size <= 1:
                continue
            own_distances = distances[np.ix_(own_indices, own_indices)]
            within = np.sum(own_distances, axis = 1) / (own_indices.size - 1)
            nearest_other = np.full((own_indices.size,), np.inf, dtype = np.float64)
            for other_category in categories:
                if other_category == category:
                    continue
                other_indices = np.flatnonzero(labels == other_category)
                if other_indices.size == 0:
                    continue
                cross_mean = np.mean(distances[np.ix_(own_indices, other_indices)], axis = 1)
                nearest_other = np.minimum(nearest_other, cross_mean)
            denominator = np.maximum(within, nearest_other)
            valid = np.isfinite(nearest_other) & (denominator > 1.0e-12)
            category_scores = np.zeros_like(within)
            category_scores[valid] = (nearest_other[valid] - within[valid]) / denominator[valid]
            scores[own_indices] = category_scores
        return float(np.mean(scores))

    def _projection_type_separation(self, coordinates, piece_types):
        """Summarize category-center distance relative to within-category spread."""
        categories = sorted(set(piece_types))
        centers = []
        within = []
        for category in categories:
            indices = [index for index, value in enumerate(piece_types) if value == category]
            points = coordinates[indices]
            center = np.mean(points, axis = 0)
            centers.append(center)
            within.extend(np.linalg.norm(points - center, axis = 1).tolist())
        between = []
        for source in range(len(centers)):
            for target in range(source + 1, len(centers)):
                between.append(float(np.linalg.norm(centers[source] - centers[target])))
        within_mean = float(np.mean(within)) if within else 0.0
        between_mean = float(np.mean(between)) if between else 0.0
        ratio = between_mean / max(within_mean, 1.0e-12)
        return {'within': within_mean, 'between': between_mean, 'ratio': ratio}

    def transformer_attention_analysis_text(self, ai_index):
        ai = self.frame.AIs[ai_index]
        if not self._supports_original_attention_analysis(ai):
            return (
                f'AI {ai_index} does not expose Original Transformer attention.\n'
                'Required: params W1/WQ1/WK1/WV1 and layer Att1.\n'
                'Use an Original Rubiks_3_AI with use_transformer_attention=True.'
            )

        x = self.frame.cube.makedata().reshape(-1,1)
        policy_value = ai.predict(x, policy = True, value = True, loss = False, retain_cache = True)
        logits = policy_value[:-1,0]
        value = float(policy_value[-1,0])
        attention_layer = self._original_attention_layer(ai)
        attention = np.asarray(attention_layer.M[0], dtype = 'f')
        token_count = attention.shape[0]
        blocks = self._piece_feature_blocks()
        if getattr(ai,'use_piece_tokens',False):
            piece_names, saliency = self._piece_saliency(ai, x, blocks)
            token_vectors = attention_layer.tokens[:,:,0]
            w1_mass = np.linalg.norm(token_vectors, axis = 1)
            relation = attention[:len(piece_names),:len(piece_names)]
        else:
            piece_names, part_token, saliency = self._attention_piece_projection(ai, x, blocks)
            w1_mass = np.sum(part_token, axis = 1)
            relation = part_token @ attention @ part_token.T
        relation = self._zero_relation_diagonal(relation)
        centrality = np.sum(relation, axis = 0) + np.sum(relation, axis = 1)
        entropy_rows = self._attention_entropy_rows(attention)

        lines = [
            f'AI: {ai_index}',
            f'value: {value:.6f}',
            f'top policy: {self._top_policy_text(logits, top_n = 5)}',
            f'token_count: {token_count}',
            f'attention entropy mean/min/max: {float(np.mean(entropy_rows)):.4f} / {float(np.min(entropy_rows)):.4f} / {float(np.max(entropy_rows)):.4f}',
            f'attention max mean/max: {float(np.mean(np.max(attention, axis = 1))):.4f} / {float(np.max(attention)):.4f}',
            '',
            '[Part Importance: value grad * input]',
        ]
        lines += self._format_ranked_scores(piece_names, saliency, top_n = 20)
        lines += [
            '',
            '[Part Importance: W1-token mass]',
        ]
        lines += self._format_ranked_scores(piece_names, w1_mass, top_n = 20)
        lines += [
            '',
            '[Part Importance: attention centrality]',
        ]
        lines += self._format_ranked_scores(piece_names, centrality, top_n = 20)
        lines += [
            '',
            '[Part Relations: query -> key]',
        ]
        lines += self._format_relation_scores(piece_names, relation, top_n = 30)
        lines += [
            '',
            '[Token Diagnostics]',
        ]
        if getattr(ai,'use_piece_tokens',False):
            lines += self._format_piece_token_diagnostics(attention, token_vectors, piece_names, top_n = 12)
        else:
            lines += self._format_token_diagnostics(ai, x, attention, part_token, piece_names, top_n = 12)
        return '\n'.join(lines)

    def transformer_embedding_analysis_text(self, ai_index):
        ai = self.frame.AIs[ai_index]
        if not self._supports_original_attention_analysis(ai):
            return (
                f'AI {ai_index} does not expose Original Transformer embeddings.\n'
                'Required: params W1/WQ1/WK1/WV1 and layer Aff1.\n'
                'Use an Original Rubiks_3_AI with use_transformer_attention=True.'
            )

        x = self.frame.cube.makedata().reshape(-1,1)
        ai.predict(x, policy = True, value = True, loss = False, retain_cache = True)
        blocks = self._piece_feature_blocks()
        piece_names, embeddings = self._piece_embedding_matrix(ai, x, blocks)
        norms = np.linalg.norm(embeddings, axis = 1)
        cosine = self._embedding_cosine_matrix(embeddings)
        centered = embeddings - np.mean(embeddings, axis = 0, keepdims = True)
        svd_values, explained, pc1_scores = self._embedding_svd_summary(centered)

        lines = [
            f'AI: {ai_index}',
            f'embedding_count: {embeddings.shape[0]}',
            f'embedding_dim: {embeddings.shape[1] if embeddings.ndim == 2 else 0}',
            f'norm mean/min/max: {float(np.mean(norms)):.6f} / {float(np.min(norms)):.6f} / {float(np.max(norms)):.6f}',
            '',
            '[Embedding Norm]',
        ]
        lines += self._format_ranked_scores(piece_names, norms, top_n = 20)
        lines += [
            '',
            '[Embedding PC1 Score]',
        ]
        lines += self._format_ranked_scores(piece_names, pc1_scores, top_n = 12)
        lines += [
            '',
            '[Embedding PC1 Score: negative side]',
        ]
        lines += self._format_ranked_scores(piece_names, -pc1_scores, top_n = 12)
        lines += [
            '',
            '[Nearest Embedding Pairs: cosine]',
        ]
        lines += self._format_embedding_pair_scores(piece_names, cosine, largest = True, top_n = 30)
        lines += [
            '',
            '[Opposite Embedding Pairs: cosine]',
        ]
        lines += self._format_embedding_pair_scores(piece_names, cosine, largest = False, top_n = 20)
        lines += [
            '',
            '[Embedding Type Similarity]',
        ]
        lines += self._format_embedding_type_similarity(piece_names, cosine)
        lines += [
            '',
            '[Embedding SVD]',
            '  singular values: ' + ', '.join(f'{float(value):.5f}' for value in svd_values[:8]),
            '  explained ratio: ' + ', '.join(f'{float(value):.4f}' for value in explained[:8]),
        ]

        relation_names, relation = self._current_attention_relation(ai, x, blocks)
        if relation is not None and len(relation_names) == len(piece_names):
            lines += [
                '',
                '[Attention vs Embedding]',
                f'  pearson(attention, cosine): {self._relation_correlation(relation, cosine): .6f}',
                '',
                '[Strong Attention Relations with Embedding Cosine]',
            ]
            lines += self._format_attention_embedding_pairs(piece_names, relation, cosine, top_n = 20)
        return '\n'.join(lines)

    def _supports_original_attention_analysis(self, ai):
        return (
            hasattr(ai, 'params')
            and all(key in ai.params for key in ('W1','WQ1','WK1','WV1'))
            and hasattr(ai, 'layers')
            and self._original_attention_layer(ai) is not None
        )

    def _original_attention_layer(self, ai):
        if hasattr(ai,'layers') and 'Att1' in ai.layers:
            return ai.layers['Att1']
        if hasattr(ai,'layers') and 'Aff1' in ai.layers and hasattr(ai.layers['Aff1'],'M'):
            return ai.layers['Aff1']
        return None

    def _piece_saliency(self, ai, x, blocks):
        try:
            grad = ai.grad(x, layer = 'WO_V').reshape(-1)
            feature_saliency = np.abs(grad * x.reshape(-1))
        except Exception:
            feature_saliency = np.zeros((x.shape[0],), dtype = 'f')
        piece_names = []
        saliency = []
        max_count = len(getattr(ai.layers['Aff1'],'piece_feature_indices',blocks))
        for name, mask in blocks[:max_count]:
            piece_names.append(name)
            saliency.append(float(np.sum(feature_saliency[mask])))
        return piece_names, np.asarray(saliency,dtype = 'f')

    def _attention_piece_projection(self, ai, x, blocks):
        contribution = np.zeros((len(blocks), ai.params['W1'].shape[0]), dtype = 'f')
        saliency = np.zeros((len(blocks),), dtype = 'f')
        try:
            grad = ai.grad(x, layer = 'WO_V').reshape(-1)
            feature_saliency = np.abs(grad * x.reshape(-1))
        except Exception:
            feature_saliency = np.zeros((x.shape[0],), dtype = 'f')

        for piece_index, (_, mask) in enumerate(blocks):
            if np.any(mask):
                contribution[piece_index] = np.sum(np.abs(ai.params['W1'][:,mask] * x[mask,0].reshape(1,-1)), axis = 1)
                saliency[piece_index] = float(np.sum(feature_saliency[mask]))

        part_token = self._normalize_columns(contribution)
        piece_names = [name for name, _ in blocks]
        return piece_names, part_token, saliency

    def _normalize_columns(self, matrix):
        denom = np.sum(matrix, axis = 0, keepdims = True)
        normalized = np.zeros_like(matrix, dtype = 'f')
        np.divide(matrix, denom, out = normalized, where = denom > 1.0e-8)
        return normalized

    def _zero_relation_diagonal(self, relation):
        relation = relation.copy()
        if relation.size > 0:
            np.fill_diagonal(relation, 0.0)
        return relation

    def _attention_entropy_rows(self, attention):
        return -np.sum(attention * np.log(attention + 1.0e-8), axis = 1)

    def _top_policy_text(self, logits, top_n = 5):
        ordered = np.argsort(logits)[-top_n:][::-1]
        items = []
        for index in ordered:
            move = self.frame.display_move_sequence((self.frame.move_keys[int(index)],))[0]
            items.append(f'{move}:{float(logits[index]):.4f}')
        return ', '.join(items)

    def _format_ranked_scores(self, names, scores, top_n = 20):
        if len(scores) == 0:
            return ['  (none)']
        ordered = np.argsort(scores)[-min(top_n, len(scores)):][::-1]
        return [f'  {rank:02d}. {names[index]:<28} {float(scores[index]): .6f}' for rank, index in enumerate(ordered, 1)]

    def _format_relation_scores(self, names, relation, top_n = 30):
        if relation.size == 0:
            return ['  (none)']
        flat = relation.reshape(-1)
        positive_indices = np.where(flat > 0)[0]
        if positive_indices.size == 0:
            return ['  (none)']
        selected = positive_indices[np.argsort(flat[positive_indices])[-min(top_n, positive_indices.size):][::-1]]
        lines = []
        width = relation.shape[1]
        for rank, flat_index in enumerate(selected, 1):
            source = int(flat_index // width)
            target = int(flat_index % width)
            lines.append(f'  {rank:02d}. {names[source]} -> {names[target]}  {float(relation[source,target]): .6f}')
        return lines

    def _format_token_diagnostics(self, ai, x, attention, part_token, piece_names, top_n = 12):
        aff_out = ai.layers['Aff1'].out[:,0] if getattr(ai.layers['Aff1'], 'out', None) is not None else ai.params['W1'] @ x[:,0]
        token_score = np.abs(aff_out) * (np.sum(attention, axis = 0) + np.sum(attention, axis = 1))
        ordered = np.argsort(token_score)[-min(top_n, token_score.size):][::-1]
        lines = []
        for rank, token_index in enumerate(ordered, 1):
            owner_index = int(np.argmax(part_token[:,token_index])) if part_token.shape[0] > 0 else -1
            owner = piece_names[owner_index] if owner_index >= 0 else 'n/a'
            entropy = -float(np.sum(attention[token_index] * np.log(attention[token_index] + 1.0e-8)))
            max_target = int(np.argmax(attention[token_index]))
            target_owner_index = int(np.argmax(part_token[:,max_target])) if part_token.shape[0] > 0 else -1
            target_owner = piece_names[target_owner_index] if target_owner_index >= 0 else 'n/a'
            lines.append(
                f'  {rank:02d}. token={int(token_index):03d} score={float(token_score[token_index]): .6f} '
                f'act={float(aff_out[token_index]): .6f} entropy={entropy:.4f} '
                f'owner={owner} max_attn_token={max_target:03d} max_attn_part={target_owner} '
                f'max_attn={float(attention[token_index,max_target]):.4f}'
            )
        return lines

    def _format_piece_token_diagnostics(self, attention, token_vectors, piece_names, top_n = 12):
        centrality = np.sum(attention, axis = 0) + np.sum(attention, axis = 1)
        token_norm = np.linalg.norm(token_vectors, axis = 1)
        token_score = token_norm * centrality
        ordered = np.argsort(token_score)[-min(top_n, token_score.size):][::-1]
        lines = []
        for rank, token_index in enumerate(ordered, 1):
            entropy = -float(np.sum(attention[token_index] * np.log(attention[token_index] + 1.0e-8)))
            max_target = int(np.argmax(attention[token_index]))
            owner = piece_names[token_index] if token_index < len(piece_names) else 'n/a'
            target_owner = piece_names[max_target] if max_target < len(piece_names) else 'n/a'
            lines.append(
                f'  {rank:02d}. piece_token={int(token_index):03d} score={float(token_score[token_index]): .6f} '
                f'norm={float(token_norm[token_index]): .6f} entropy={entropy:.4f} '
                f'piece={owner} max_attn_piece={target_owner} max_attn={float(attention[token_index,max_target]):.4f}'
            )
        return lines

    def myviewer(self, AInum, i, N = 1, SVD = False, Grad = False, IG = False, Contrast = False, Occ = False, PieceOcc = False, PolicyOcc = False, PiecePolicyOcc = False, AttnIn = False, AttnOut = False, AttnCentral = False, EmbNorm = False, EmbPC1 = False, layer = "WO_V"):
        """指定した重み・勾配・SVD成分をキューブ状態として可視化する。"""
        vector = self._viewer_vector(AInum,i,SVD,Grad,IG,Contrast,Occ,PieceOcc,PolicyOcc,PiecePolicyOcc,AttnIn,AttnOut,AttnCentral,EmbNorm,EmbPC1,layer)
        positive_state,negative_state = self._viewer_states(vector,N)
        self.frame.grad_viewer_positive.set_color(positive_state)
        self.frame.grad_viewer_negative.set_color(negative_state)
        self.frame.set_grad_viewer_info(
            self._viewer_label_text(AInum),
            self._viewer_range_text(vector,N,positive = True),
            self._viewer_range_text(vector,N,positive = False),
        )
        if Contrast:
            self._log_policy_contrast(AInum)
        if Occ or PieceOcc or PolicyOcc or PiecePolicyOcc:
            if Occ:
                mode_name = 'group'
            elif PieceOcc:
                mode_name = 'piece'
            elif PolicyOcc:
                mode_name = 'policy-group'
            else:
                mode_name = 'policy-piece'
            self._show_occlusion_scores(AInum, mode_name)
        if AttnIn or AttnOut or AttnCentral:
            self._show_attention_scores(AInum)
        if EmbNorm or EmbPC1:
            self._show_embedding_scores(AInum)

    def _build_myperm_inputs(self, keys):
        """myperm候補の手順を入力データ行列に変換する。"""
        input_data = np.zeros((self.frame.cube.ips,len(keys)),dtype = 'f')
        empty_input = np.zeros((self.frame.cube.ips,1),dtype = 'f')
        for index,key in enumerate(keys):
            self._write_myperm_input(input_data,index,key)
        return input_data,empty_input

    def _parse_grad_index(self, index_text):
        """grad index入力を整数へ変換し、変換できない場合はNoneを返す。"""
        try:
            return int(index_text)
        except ValueError:
            return None

    def _sync_frame_grad_settings(self):
        """既存コードとの互換性のため、Frame側のgrad設定にも同じ値を反映する。"""
        self.frame.grad_index = self.grad_index
        self.frame.grad_mode = self.grad_mode
        self.frame.grad_layer = self.grad_layer

    def _viewer_label_text(self, ai_index):
        """GradViewerの対象情報を短く表示する。"""
        return f'AI {ai_index}  mode={self.grad_mode}  idx={self.grad_index}  layer={self.grad_layer}'

    def _viewer_range_text(self, vector, N, positive):
        """色付け対象として選んだ値のrangeを表示用文字列にする。"""
        selected_values = self._selected_viewer_values(vector,N,positive)
        label = 'High value' if positive else 'Low value'
        if selected_values.size == 0:
            return f'{label}: n=0'
        return (
            f'{label}: n={selected_values.size}  '
            f'[{float(np.min(selected_values)): .4g} .. {float(np.max(selected_values)): .4g}]'
        )

    def _selected_viewer_values(self, vector, N, positive):
        """Positive/Negative viewerで実際に選ぶ特徴値を返す。"""
        flat_vector = np.asarray(vector).reshape(-1)
        if flat_vector.size == 0:
            return np.zeros((0,), dtype = 'f')
        count = max(0,min(int(N),flat_vector.size))
        if count == 0:
            return np.zeros((0,), dtype = 'f')
        ordered_indices = np.argsort(flat_vector)
        if positive:
            selected_indices = ordered_indices[-count:]
        else:
            selected_indices = ordered_indices[:count]
        return flat_vector[selected_indices]

    def _write_myperm_input(self, input_data, index, key):
        """1つのmyperm手順を実行し、その途中状態を評価用入力に書き込む。"""
        self.frame.cube.reset()
        for move in self.frame.cube.invert_moves(self.frame.cube.myperms[key]):
            self.frame.cube.make_move(move)
            input_data[:,index] = self.frame.cube.makedata()

    def _print_ai_value_ranking(self, index, keys, input_data, empty_input, Top, Num):
        """1つのAIについて、myperm候補の評価値ランキングを表示する。"""
        ai = self.frame.AIs[index]
        values = ai.predict(input_data,policy = False,value = True).reshape(-1)
        ordered_indices = np.argsort(values)
        selected_indices = self._selected_value_indices(ordered_indices,Top,Num)
        selected_keys = [keys[selected_index] for selected_index in selected_indices]
        selected_values = values[selected_indices]
        print(index,selected_keys,ai.perfect_val - selected_values)
        ai.predict(empty_input,policy = False,value = True).reshape(-1)

    def _selected_value_indices(self, ordered_indices, Top, Num):
        """上位表示か下位表示かに応じて、表示対象のindexを選ぶ。"""
        if Top:
            return ordered_indices[-Num:]
        return ordered_indices[:Num]

    def _normalize_param(self, ai, key):
        """1つのパラメータ配列に対して正規化または初期値リセットを行う。"""
        if key[0] == 'W' and len(key) == 2:
            scale = np.sqrt(np.var(ai.params[key],axis = 1).reshape(-1,1)) * np.sqrt(ai.params[key].shape[1] / 2)
            ai.params[key] /= scale
            ai.params['B' + key[1:]] /= scale.reshape(-1)
            ai.v[key] *= 0
        elif key[:3] == 'BNg':
            ai.params[key][:] = 1
        elif key[:3] == 'BNb':
            ai.params[key][:] = 0

    def _is_reactivation_target(self, key):
        """再活性化の対象になる重みパラメータか判定する。"""
        return key[0] == 'W' and key not in ['WO_P','WO_V','WM_P','WM_V']

    def _reactivate_param(self, ai, key):
        """更新量が小さいユニットに小さなバイアスと対角的な重みを入れる。"""
        weak_indices = np.where(ai.h['B' + key[1:]] < 1.0e-6)[0]
        print(key,weak_indices)
        ai.params['B' + key[1:]][weak_indices] = 0.05
        for weak_index in weak_indices:
            ai.params[key][weak_index,weak_index % ai.params[key].shape[1]] = -1.0

    def _viewer_vector(self, AInum, i, SVD, Grad, IG, Contrast, Occ, PieceOcc, PolicyOcc, PiecePolicyOcc, AttnIn, AttnOut, AttnCentral, EmbNorm, EmbPC1, layer):
        """myviewerで表示する元ベクトルを、指定モードに応じて取得する。"""
        ai = self.frame.AIs[AInum]
        if SVD:
            svd_result = np.linalg.svd(ai.params['W1'])
            return svd_result[2][i]
        if Grad:
            x = self.frame.cube.makedata().reshape(-1,1)
            return ai.grad(x,layer = layer,index = i).reshape(-1)
        if IG:
            x = self.frame.cube.makedata()
            return ai.integrated_grad(x,layer = layer,index = i).reshape(-1)
        if Contrast:
            return self._policy_contrast_vector(ai)
        if Occ:
            return self._occlusion_vector(ai, self._group_feature_blocks())
        if PieceOcc:
            return self._occlusion_vector(ai, self._piece_feature_blocks())
        if PolicyOcc:
            return self._policy_occlusion_vector(ai, self._group_feature_blocks())
        if PiecePolicyOcc:
            return self._policy_occlusion_vector(ai, self._piece_feature_blocks())
        if AttnIn or AttnOut or AttnCentral:
            return self._transformer_attention_vector(ai, AttnIn, AttnOut, AttnCentral)
        if EmbNorm or EmbPC1:
            return self._transformer_embedding_vector(ai, EmbNorm, EmbPC1)
        return ai.params['W1'][i]

    def _policy_contrast_vector(self, ai):
        """Return grad(logit_top1 - logit_top2) with respect to input."""
        x = self.frame.cube.makedata().reshape(-1,1)
        logits = ai.predict(x, policy = True, value = False).reshape(-1)
        ordered = np.argsort(logits)
        top1 = int(ordered[-1])
        top2 = int(ordered[-2]) if len(ordered) >= 2 else top1
        grad_top1 = ai.grad(x, layer = "WO_P", index = top1).reshape(-1)
        grad_top2 = ai.grad(x, layer = "WO_P", index = top2).reshape(-1)
        self._last_policy_contrast = {
            'top1_index': top1,
            'top2_index': top2,
            'top1_move': self.frame.display_move_sequence((self.frame.move_keys[top1],))[0],
            'top2_move': self.frame.display_move_sequence((self.frame.move_keys[top2],))[0],
            'top1_logit': float(logits[top1]),
            'top2_logit': float(logits[top2]),
        }
        return grad_top1 - grad_top2

    def _log_policy_contrast(self, ai_index):
        """Log the top-1 vs top-2 policy contrast metadata."""
        if not hasattr(self, '_last_policy_contrast'):
            return
        info = self._last_policy_contrast
        self.frame.append_log(
            'contrast: ai={ai} {m1}({v1:.4f}) - {m2}({v2:.4f})'.format(
                ai = ai_index,
                m1 = info['top1_move'],
                v1 = info['top1_logit'],
                m2 = info['top2_move'],
                v2 = info['top2_logit'],
            )
        )

    def _occlusion_vector(self, ai, blocks):
        """Return a feature vector where each block is weighted by its value-drop under occlusion."""
        x = self.frame.cube.makedata().reshape(-1,1)
        base_value = float(ai.predict(x,policy = False,value = True)[0][0])
        vector = np.zeros((self.frame.cube.ips,),dtype = 'f')
        self._last_occlusion_scores = []
        for key, mask in blocks:
            occluded_x = x.copy()
            occluded_x[mask,0] = self.frame.cube.perfect_data[mask]
            occluded_value = float(ai.predict(occluded_x,policy = False,value = True)[0][0])
            score = base_value - occluded_value
            vector[mask] += score
            self._last_occlusion_scores.append((key, score))
        return vector

    def _show_occlusion_scores(self, ai_index, mode_name):
        """Log and show the most influential occlusion blocks."""
        if not self._last_occlusion_scores:
            return
        ordered_scores = sorted(self._last_occlusion_scores, key = lambda item: item[1], reverse = True)
        top_scores = ', '.join(f'{key}={score:.4f}' for key, score in ordered_scores[:5])
        if mode_name.startswith('policy') and hasattr(self, '_last_policy_occlusion'):
            info = self._last_policy_occlusion
            self.frame.append_log(
                f"{mode_name}-occ: ai={ai_index} "
                f"{info['top1_move']}-{info['top2_move']} margin={info['base_margin']:.4f} {top_scores}"
            )
        else:
            self.frame.append_log(f'{mode_name}-occ: ai={ai_index} {top_scores}')
        self.frame.show_analysis_scores(
            f'{mode_name} occlusion scores (ai={ai_index})',
            ordered_scores[: min(30, len(ordered_scores))],
        )

    def _transformer_attention_vector(self, ai, AttnIn, AttnOut, AttnCentral):
        """Return an input-feature vector that highlights Transformer attention by piece."""
        x = self.frame.cube.makedata().reshape(-1,1)
        vector = np.zeros((self.frame.cube.ips,), dtype = 'f')
        self._last_attention_scores = []
        self._last_attention_relation = None
        self._last_attention_error = ''
        if AttnOut:
            self._last_attention_mode = 'out'
        elif AttnIn:
            self._last_attention_mode = 'in'
        else:
            self._last_attention_mode = 'central'
        if not self._supports_original_attention_analysis(ai):
            self._last_attention_error = 'This AI does not expose Original Transformer attention.'
            return vector
        try:
            ai.predict(x, policy = True, value = True, loss = False, retain_cache = True)
            attention_layer = self._original_attention_layer(ai)
            attention = np.asarray(attention_layer.M[0], dtype = 'f')
            blocks = self._piece_feature_blocks()
            piece_names, relation = self._attention_piece_relation(ai, x, attention, blocks)
        except Exception as error:
            self._last_attention_error = str(error)
            return vector

        relation = self._zero_relation_diagonal(relation)
        if relation.size == 0:
            self._last_attention_error = 'Attention relation is empty.'
            return vector

        if AttnOut:
            scores = np.sum(relation, axis = 1)
            mode_name = 'out'
        elif AttnIn:
            scores = np.sum(relation, axis = 0)
            mode_name = 'in'
        else:
            scores = np.sum(relation, axis = 0) + np.sum(relation, axis = 1)
            mode_name = 'central'

        self._last_attention_scores = [(piece_names[index], float(scores[index])) for index in range(len(scores))]
        self._last_attention_relation = (piece_names, relation)
        self._last_attention_mode = mode_name
        return self._piece_scores_to_active_feature_vector(scores, blocks, x)

    def _attention_piece_relation(self, ai, x, attention, blocks):
        """Convert token-token attention into a piece-piece relation matrix."""
        if getattr(ai,'use_piece_tokens',False):
            piece_count = min(len(blocks), attention.shape[0])
            piece_names = [name for name, _ in blocks[:piece_count]]
            return piece_names, attention[:piece_count,:piece_count]
        piece_names, part_token, _ = self._attention_piece_projection(ai, x, blocks)
        relation = part_token @ attention @ part_token.T
        return piece_names, relation

    def _piece_scores_to_active_feature_vector(self, scores, blocks, x):
        """Map one score per piece to the currently active feature of that piece."""
        vector = np.zeros((self.frame.cube.ips,), dtype = 'f')
        flat_x = x.reshape(-1)
        for index, score in enumerate(scores[:len(blocks)]):
            mask = blocks[index][1]
            active_indices = np.where(mask & (flat_x > 0))[0]
            if active_indices.size == 0:
                active_indices = np.where(mask)[0][:1]
            vector[active_indices] = float(score)
        return vector

    def _show_attention_scores(self, ai_index):
        """Show attention piece scores and the strongest piece-to-piece relations."""
        title = f'transformer attention {self._last_attention_mode} (ai={ai_index})'
        if self._last_attention_error:
            self.frame.append_log(f'attention viewer: ai={ai_index} {self._last_attention_error}')
            self.frame.show_analysis_text(title, self._last_attention_error)
            return
        if not self._last_attention_scores:
            return

        ordered_scores = sorted(self._last_attention_scores, key = lambda item: item[1], reverse = True)
        top_scores = ', '.join(f'{key}={score:.4f}' for key, score in ordered_scores[:5])
        self.frame.append_log(f'transformer-attn-{self._last_attention_mode}: ai={ai_index} {top_scores}')

        lines = [
            title,
            '-' * len(title),
            '',
            '[Piece Scores]',
        ]
        names = [name for name, _ in self._last_attention_scores]
        scores = np.asarray([score for _, score in self._last_attention_scores], dtype = 'f')
        lines += self._format_ranked_scores(names, scores, top_n = 30)
        if self._last_attention_relation is not None:
            relation_names, relation = self._last_attention_relation
            lines += [
                '',
                '[Strong Relations: query -> key]',
            ]
            lines += self._format_relation_scores(relation_names, relation, top_n = 30)
        self.frame.show_analysis_text(title, '\n'.join(lines))

    def _transformer_embedding_vector(self, ai, EmbNorm, EmbPC1):
        """Return a feature vector that highlights Aff1 piece embeddings."""
        x = self.frame.cube.makedata().reshape(-1,1)
        vector = np.zeros((self.frame.cube.ips,), dtype = 'f')
        self._last_embedding_scores = []
        self._last_embedding_error = ''
        self._last_embedding_mode = 'norm' if EmbNorm else 'pc1'
        if not self._supports_original_attention_analysis(ai):
            self._last_embedding_error = 'This AI does not expose Original Transformer embeddings.'
            return vector
        try:
            ai.predict(x, policy = True, value = True, loss = False, retain_cache = True)
            blocks = self._piece_feature_blocks()
            piece_names, embeddings = self._piece_embedding_matrix(ai, x, blocks)
            if EmbNorm:
                scores = np.linalg.norm(embeddings, axis = 1)
            else:
                centered = embeddings - np.mean(embeddings, axis = 0, keepdims = True)
                _, _, scores = self._embedding_svd_summary(centered)
        except Exception as error:
            self._last_embedding_error = str(error)
            return vector
        self._last_embedding_scores = [(piece_names[index], float(scores[index])) for index in range(len(scores))]
        return self._piece_scores_to_active_feature_vector(scores, blocks, x)

    def _show_embedding_scores(self, ai_index):
        """Show the current embedding viewer scores and full embedding diagnostics."""
        title = f'transformer embedding {self._last_embedding_mode} (ai={ai_index})'
        if self._last_embedding_error:
            self.frame.append_log(f'embedding viewer: ai={ai_index} {self._last_embedding_error}')
            self.frame.show_analysis_text(title, self._last_embedding_error)
            return
        if self._last_embedding_scores:
            ordered_scores = sorted(self._last_embedding_scores, key = lambda item: item[1], reverse = True)
            top_scores = ', '.join(f'{key}={score:.4f}' for key, score in ordered_scores[:5])
            self.frame.append_log(f'transformer-emb-{self._last_embedding_mode}: ai={ai_index} {top_scores}')
        self.show_transformer_embedding_analysis(ai_index)

    def _piece_embedding_matrix(self, ai, x, blocks):
        """Return one Aff1 embedding vector per piece."""
        if getattr(ai,'use_piece_tokens',False):
            attention_layer = self._original_attention_layer(ai)
            embeddings = np.asarray(attention_layer.tokens[:,:,0], dtype = 'f')
            piece_count = min(len(blocks), embeddings.shape[0])
            piece_names = [name for name, _ in blocks[:piece_count]]
            return piece_names, embeddings[:piece_count]

        embedding_dim = ai.params['W1'].shape[0]
        embeddings = np.zeros((len(blocks), embedding_dim), dtype = 'f')
        for piece_index, (_, mask) in enumerate(blocks):
            if np.any(mask):
                embeddings[piece_index] = ai.params['W1'][:,mask] @ x[mask,0]
        piece_names = [name for name, _ in blocks]
        return piece_names, embeddings

    def _embedding_cosine_matrix(self, embeddings):
        """Return pairwise cosine similarity between piece embeddings."""
        norms = np.linalg.norm(embeddings, axis = 1, keepdims = True)
        normalized = np.zeros_like(embeddings, dtype = 'f')
        np.divide(embeddings, norms, out = normalized, where = norms > 1.0e-8)
        return normalized @ normalized.T

    def _embedding_svd_summary(self, centered_embeddings):
        """Return singular values, explained ratios and PC1 scores for embeddings."""
        if centered_embeddings.size == 0:
            return np.zeros((0,), dtype = 'f'), np.zeros((0,), dtype = 'f'), np.zeros((0,), dtype = 'f')
        try:
            _, singular_values, vh = np.linalg.svd(centered_embeddings, full_matrices = False)
        except np.linalg.LinAlgError:
            return np.zeros((0,), dtype = 'f'), np.zeros((0,), dtype = 'f'), np.zeros((centered_embeddings.shape[0],), dtype = 'f')
        total = float(np.sum(singular_values ** 2))
        explained = np.zeros_like(singular_values, dtype = 'f')
        if total > 1.0e-8:
            explained = (singular_values ** 2 / total).astype('f')
        if vh.shape[0] == 0:
            pc1_scores = np.zeros((centered_embeddings.shape[0],), dtype = 'f')
        else:
            pc1_scores = (centered_embeddings @ vh[0]).astype('f')
        return singular_values.astype('f'), explained, pc1_scores

    def _format_embedding_pair_scores(self, names, cosine, largest = True, top_n = 20):
        """Format strongest or most opposite embedding pairs."""
        if cosine.size == 0:
            return ['  (none)']
        pair_scores = []
        for source in range(cosine.shape[0]):
            for target in range(source + 1, cosine.shape[1]):
                pair_scores.append((float(cosine[source,target]), source, target))
        if not pair_scores:
            return ['  (none)']
        pair_scores.sort(key = lambda item: item[0], reverse = largest)
        lines = []
        for rank, (score, source, target) in enumerate(pair_scores[:min(top_n, len(pair_scores))], 1):
            lines.append(f'  {rank:02d}. {names[source]} <-> {names[target]}  {score: .6f}')
        return lines

    def _format_embedding_type_similarity(self, names, cosine):
        """Summarize cosine similarity grouped by piece label prefix."""
        groups = {}
        for index, name in enumerate(names):
            piece_type = name.split('-', 1)[0]
            groups.setdefault(piece_type, []).append(index)
        lines = []
        for source_type, source_indices in groups.items():
            for target_type, target_indices in groups.items():
                if source_type > target_type:
                    continue
                values = []
                for source in source_indices:
                    for target in target_indices:
                        if source >= target:
                            continue
                        values.append(float(cosine[source,target]))
                if values:
                    lines.append(f'  {source_type:<10} {target_type:<10} mean={float(np.mean(values)): .6f} n={len(values)}')
        return lines if lines else ['  (none)']

    def _current_attention_relation(self, ai, x, blocks):
        """Return the current piece-piece attention relation if available."""
        try:
            attention_layer = self._original_attention_layer(ai)
            attention = np.asarray(attention_layer.M[0], dtype = 'f')
            piece_names, relation = self._attention_piece_relation(ai, x, attention, blocks)
            return piece_names, self._zero_relation_diagonal(relation)
        except Exception:
            return None, None

    def _relation_correlation(self, relation, cosine):
        """Pearson correlation between off-diagonal attention relation and embedding cosine."""
        if relation.shape != cosine.shape or relation.size == 0:
            return 0.0
        mask = ~np.eye(relation.shape[0], dtype = bool)
        x = relation[mask].reshape(-1)
        y = cosine[mask].reshape(-1)
        if x.size == 0 or float(np.std(x)) < 1.0e-8 or float(np.std(y)) < 1.0e-8:
            return 0.0
        return float(np.corrcoef(x, y)[0,1])

    def _format_attention_embedding_pairs(self, names, relation, cosine, top_n = 20):
        """Format strong attention edges with their embedding cosine."""
        if relation.size == 0:
            return ['  (none)']
        flat = relation.reshape(-1)
        positive_indices = np.where(flat > 0)[0]
        if positive_indices.size == 0:
            return ['  (none)']
        selected = positive_indices[np.argsort(flat[positive_indices])[-min(top_n, positive_indices.size):][::-1]]
        width = relation.shape[1]
        lines = []
        for rank, flat_index in enumerate(selected, 1):
            source = int(flat_index // width)
            target = int(flat_index % width)
            lines.append(
                f'  {rank:02d}. {names[source]} -> {names[target]}  '
                f'attn={float(relation[source,target]): .6f} cos={float(cosine[source,target]): .6f}'
            )
        return lines

    def _policy_occlusion_vector(self, ai, blocks):
        """Return a feature vector weighted by top1-vs-top2 policy margin drop under occlusion."""
        x = self.frame.cube.makedata().reshape(-1,1)
        logits = ai.predict(x, policy = True, value = False).reshape(-1)
        ordered = np.argsort(logits)
        top1 = int(ordered[-1])
        top2 = int(ordered[-2]) if len(ordered) >= 2 else top1
        base_margin = float(logits[top1] - logits[top2])
        vector = np.zeros((self.frame.cube.ips,), dtype = 'f')
        self._last_occlusion_scores = []
        self._last_policy_occlusion = {
            'top1_index': top1,
            'top2_index': top2,
            'top1_move': self.frame.display_move_sequence((self.frame.move_keys[top1],))[0],
            'top2_move': self.frame.display_move_sequence((self.frame.move_keys[top2],))[0],
            'base_margin': base_margin,
        }
        for key, mask in blocks:
            occluded_x = x.copy()
            occluded_x[mask,0] = self.frame.cube.perfect_data[mask]
            occluded_logits = ai.predict(occluded_x, policy = True, value = False).reshape(-1)
            occluded_margin = float(occluded_logits[top1] - occluded_logits[top2])
            score = base_margin - occluded_margin
            vector[mask] += score
            self._last_occlusion_scores.append((key, score))
        return vector

    def _group_feature_blocks(self):
        """Return boolean feature masks for each solve group."""
        blocks = []
        for key, group_vector in self.frame.cube.group_val.items():
            blocks.append((key, group_vector.reshape(-1) > 0))
        return blocks

    def _piece_feature_blocks(self):
        """Return boolean feature masks for each piece-sized input block."""
        if self._is_megaminx_viewer():
            return self._megaminx_piece_feature_blocks()
        if self._has_generic_piece_feature_layout():
            return self._generic_piece_feature_blocks()
        return self._rubiks_piece_feature_blocks()

    def _rubiks_piece_feature_blocks(self):
        """Build Rubiks piece-level feature masks from makedata layout."""
        blocks = []
        offset = 0
        for piece in self.frame.cube.center_index:
            mask = np.zeros((self.frame.cube.ips,), dtype = bool)
            mask[offset:offset + 6] = True
            blocks.append((self._piece_block_label('Center', piece), mask))
            offset += 6
        for piece in self.frame.cube.edge_index:
            mask = np.zeros((self.frame.cube.ips,), dtype = bool)
            mask[offset:offset + 24] = True
            blocks.append((self._piece_block_label('Edge', piece), mask))
            offset += 24
        for piece in self.frame.cube.corner_index:
            mask = np.zeros((self.frame.cube.ips,), dtype = bool)
            mask[offset:offset + 24] = True
            blocks.append((self._piece_block_label('Corner', piece), mask))
            offset += 24
        return blocks

    def _megaminx_piece_feature_blocks(self):
        """Build Megaminx piece-level feature masks from makedata layout."""
        blocks = []
        offset = 0
        for piece in self.frame.cube.corner_index:
            mask = np.zeros((self.frame.cube.ips,), dtype = bool)
            mask[offset:offset + 60] = True
            blocks.append((self._piece_block_label('Corner', piece), mask))
            offset += 60
        for piece in self.frame.cube.edge_index:
            mask = np.zeros((self.frame.cube.ips,), dtype = bool)
            mask[offset:offset + 60] = True
            blocks.append((self._piece_block_label('Edge', piece), mask))
            offset += 60
        return blocks

    def _generic_piece_feature_blocks(self):
        """Build piece-level feature masks from puzzle-provided feature offsets."""
        blocks = []
        for group_name, pieces in self.frame.cube.group_pieces.items():
            for piece in pieces:
                offset, feature_size = self.frame.cube.piece_feature_offsets[piece]
                mask = np.zeros((self.frame.cube.ips,), dtype = bool)
                mask[offset:offset + feature_size] = True
                blocks.append((self._piece_block_label(group_name, piece), mask))
        return blocks

    def _piece_block_label(self, piece_type, piece):
        """Format a piece label using sticker indices so the target piece is identifiable."""
        if hasattr(self.frame.cube, 'piece_display_name'):
            return self.frame.cube.piece_display_name(piece_type, piece)
        return f"{piece_type}-{self._piece_indices_text(piece)}"

    def _piece_indices_text(self, piece):
        """Convert a piece tuple like (12, 34, 56) to a compact label."""
        return '-'.join(f'{index:02d}' for index in piece)

    def _is_megaminx_viewer(self):
        """Return whether the current cube exposes Megaminx-style feature ordering."""
        return hasattr(self.frame.cube, 'corner_key') and hasattr(self.frame.cube, 'edge_key') and self.frame.puzzle_type == 'megaminx'

    def _is_pyraminx_viewer(self):
        """Return whether the current cube exposes Pyraminx piece-feature ordering."""
        return (
            self.frame.puzzle_type in ('pyraminx', 'master_pyraminx')
            and hasattr(self.frame.cube, 'feature_index_to_piece_color')
        )

    def _has_generic_piece_feature_layout(self):
        """Return whether the current puzzle can map feature indices back to pieces."""
        return (
            hasattr(self.frame.cube, 'feature_index_to_piece_color')
            and hasattr(self.frame.cube, 'piece_feature_offsets')
            and hasattr(self.frame.cube, 'group_pieces')
        )

    def _megaminx_viewer_states(self, vector, N):
        """Map Megaminx feature indices to a viewer state using the Megaminx makedata layout."""
        state_size = len(self.frame.cube.state)
        positive_state = np.zeros(state_size, dtype = str)
        negative_state = np.zeros(state_size, dtype = str)
        positive_indices, negative_indices = self._viewer_ordered_indices(vector, N)
        self._fill_megaminx_viewer_state(positive_state, positive_indices)
        self._fill_megaminx_viewer_state(negative_state, negative_indices)
        return positive_state, negative_state

    def _fill_megaminx_viewer_state(self, state, ordered_indices):
        """Write selected Megaminx feature indices into a state array."""
        corner_limit = len(self.frame.cube.corner_index) * 60
        for vector_index in ordered_indices:
            if vector_index < corner_limit:
                self._write_megaminx_corner_to_state(state, vector_index)
            elif vector_index < self.frame.cube.ips:
                self._write_megaminx_edge_to_state(state, vector_index - corner_limit)

    def _write_megaminx_corner_to_state(self, state, vector_index):
        """Write one Megaminx corner feature to the viewer state."""
        position = self.frame.cube.corner_index[vector_index // 60]
        color = self.frame.cube.corner_colors[vector_index % 60]
        state[position[0]] = color[0]
        state[position[1]] = color[1]
        state[position[2]] = color[2]

    def _write_megaminx_edge_to_state(self, state, vector_index):
        """Write one Megaminx edge feature to the viewer state."""
        position = self.frame.cube.edge_index[vector_index // 60]
        color = self.frame.cube.edge_colors[vector_index % 60]
        state[position[0]] = color[0]
        state[position[1]] = color[1]

    def _pyraminx_viewer_states(self, vector, N):
        """Map Pyraminx piece-feature indices to a viewer state."""
        state_size = len(self.frame.cube.state)
        if self.frame.puzzle_type in ('group', 'symmetric_group', 'linear_group'):
            positive_state = np.full(state_size, '', dtype = object)
            negative_state = np.full(state_size, '', dtype = object)
        else:
            positive_state = np.zeros(state_size, dtype = str)
            negative_state = np.zeros(state_size, dtype = str)
        positive_indices, negative_indices = self._viewer_ordered_indices(vector, N)
        self._fill_pyraminx_viewer_state(positive_state, positive_indices)
        self._fill_pyraminx_viewer_state(negative_state, negative_indices)
        return positive_state, negative_state

    def _fill_pyraminx_viewer_state(self, state, ordered_indices):
        """Write selected Pyraminx feature indices into a state array."""
        for vector_index in ordered_indices:
            if vector_index not in self.frame.cube.feature_index_to_piece_color:
                continue
            piece, color = self.frame.cube.feature_index_to_piece_color[vector_index]
            for sticker_index, sticker_color in zip(piece, color):
                state[sticker_index] = sticker_color

    def _supports_vector_viewer(self):
        """Return whether the current puzzle exposes the Rubiks-style feature metadata this viewer expects."""
        required_attrs = ('center_index', 'edge_index', 'corner_index', 'edge_colors', 'corner_colors')
        return all(hasattr(self.frame.cube, attr) for attr in required_attrs)

    def _viewer_states(self, vector, N):
        """ベクトルの上位N個と下位N個を、それぞれStateViewer用の状態に変換する。"""
        if self._is_megaminx_viewer():
            return self._megaminx_viewer_states(vector, N)
        if self._is_pyraminx_viewer() or self.frame.puzzle_type == 'skewb' or self._has_generic_piece_feature_layout():
            return self._pyraminx_viewer_states(vector, N)

        state_size = 6 * self.frame.cube.surface_num
        positive_state = np.zeros(state_size,dtype = str)
        negative_state = np.zeros(state_size,dtype = str)
        if not self._supports_vector_viewer():
            return positive_state,negative_state

        positive_indices, negative_indices = self._viewer_ordered_indices(vector, N)
        self._fill_viewer_state(positive_state,positive_indices)
        self._fill_viewer_state(negative_state,negative_indices)
        return positive_state,negative_state

    def _viewer_ordered_indices(self, vector, N):
        """Positiveは大きい値、Negativeは小さい値を選ぶ。"""
        flat_vector = np.asarray(vector).reshape(-1)
        if flat_vector.size == 0:
            return np.asarray([], dtype = int), np.asarray([], dtype = int)
        count = max(0,min(int(N),flat_vector.size))
        if count == 0:
            return np.asarray([], dtype = int), np.asarray([], dtype = int)
        ordered_indices = np.argsort(flat_vector)
        positive_indices = ordered_indices[-count:][::-1]
        negative_indices = ordered_indices[:count]
        return positive_indices, negative_indices

    def _fill_viewer_state(self, state, ordered_indices):
        """選択された特徴index群をStateViewer用の色配列へ反映する。"""
        for vector_index in ordered_indices:
            self._write_vector_index_to_state(state,vector_index)

    def _write_vector_index_to_state(self, state, vector_index):
        """特徴indexがcenter/edge/cornerのどれに属するか判定して状態へ書き込む。"""
        center_limit = 36 * (self.frame.cube.size - 2) ** 2
        edge_limit = center_limit + len(self.frame.cube.edge_index) * 24
        if vector_index < center_limit:
            self._write_center_to_state(state,vector_index)
        elif vector_index < edge_limit:
            self._write_edge_to_state(state,vector_index,center_limit)
        else:
            self._write_corner_to_state(state,vector_index,edge_limit)

    def _write_center_to_state(self, state, vector_index):
        """center特徴のindexを該当ステッカー色として状態へ書き込む。"""
        position = self.frame.cube.center_index[vector_index // 6]
        color = self.frame.cube.colors[vector_index % 6]
        state[position[0]] = color[0]

    def _write_edge_to_state(self, state, vector_index, center_limit):
        """edge特徴のindexを2色のステッカー状態として書き込む。"""
        position = self.frame.cube.edge_index[(vector_index - center_limit) // 24]
        color = self.frame.cube.edge_colors[(vector_index - center_limit) % 24]
        state[position[0]] = color[0]
        state[position[1]] = color[1]

    def _write_corner_to_state(self, state, vector_index, edge_limit):
        """corner特徴のindexを3色のステッカー状態として書き込む。"""
        position = self.frame.cube.corner_index[(vector_index - edge_limit) // 24]
        color = self.frame.cube.corner_colors[(vector_index - edge_limit) % 24]
        state[position[0]] = color[0]
        state[position[1]] = color[1]
        state[position[2]] = color[2]
