from __future__ import annotations

from typing import List, Any, Optional, Tuple
from collections import Counter

from .myutils import euclidean_distance, manhattan_distance


class MyKNNClassifier:
    """K-Nearest Neighbors classifier (from scratch).

    Assumptions:
    - X is a list of list of numeric features.
    - y is a list of discrete class labels.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        distance_metric: str = "euclidean",
        weighted: bool = False
    ) -> None:
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.weighted = weighted

        self.X_train: List[List[float]] = []
        self.y_train: List[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: List[List[float]], y: List[Any]) -> "MyKNNClassifier":
        """Store the training data."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        self.X_train = X
        self.y_train = y
        return self

    def kneighbors(
        self,
        X: List[List[float]],
        n_neighbors: Optional[int] = None,
        return_distance: bool = True
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Compute the k nearest neighbors for each instance in X.

        Returns:
            distances: list of list of distances
            indices: list of list of indices into X_train
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        all_distances = []
        all_indices = []

        for x in X:
            dists = []
            for idx, x_train in enumerate(self.X_train):
                d = self._distance(x, x_train)
                dists.append((d, idx))
            # sort by distance
            dists.sort(key=lambda pair: pair[0])
            k_dists = dists[:n_neighbors]
            distances_row = [d for d, _ in k_dists]
            indices_row = [idx for _, idx in k_dists]
            all_distances.append(distances_row)
            all_indices.append(indices_row)

        if return_distance:
            return all_distances, all_indices
        else:
            return [], all_indices  # for compatibility

    def predict(self, X: List[List[float]]) -> List[Any]:
        """Predict class labels for instances in X using majority vote."""
        _, neighbors_idx = self.kneighbors(X, n_neighbors=self.n_neighbors)

        predictions = []
        for inds in neighbors_idx:
            labels = [self.y_train[i] for i in inds]

            if not self.weighted:
                # simple majority vote
                counts = Counter(labels)
            else:
                # distance-weighted vote: closer neighbors get more weight
                # re-compute distances for this row
                # (we could optimize by reusing, but this is clearer)
                dists = [self._distance(X[predictions.__len__()], self.X_train[i])
                         for i in inds]
                label_weights = {}
                for label, d in zip(labels, dists):
                    # avoid division by zero
                    w = 1.0 / (d + 1e-9)
                    label_weights[label] = label_weights.get(label, 0.0) + w
                counts = label_weights

            # pick label with highest count / weight
            max_vote = max(counts.values())
            winners = [label for label, c in counts.items() if c == max_vote]
            winners.sort(key=str)
            predictions.append(winners[0])

        return predictions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _distance(self, x1: List[float], x2: List[float]) -> float:
        """Dispatch to the proper distance metric."""
        if self.distance_metric == "euclidean":
            return euclidean_distance(x1, x2)
        elif self.distance_metric == "manhattan":
            return manhattan_distance(x1, x2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
