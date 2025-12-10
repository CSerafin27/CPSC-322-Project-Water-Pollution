from __future__ import annotations

import random
from dataclasses import dataclass
from collections import Counter
from typing import List, Any, Optional, Tuple

from .myutils import entropy


@dataclass
class _DTNode:
    """Internal node representation for MyDecisionTreeClassifier."""
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_DTNode"] = None
    right: Optional["_DTNode"] = None
    label: Optional[Any] = None  # set for leaves

    def is_leaf(self) -> bool:
        return self.label is not None


class MyDecisionTreeClassifier:
    """Decision Tree classifier (ID3/C4.5-like using entropy).

    Assumptions:
    - X is a list of list of numeric features.
    - y is a list of discrete class labels.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = None
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.root_: Optional[_DTNode] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: List[List[float]], y: List[Any]) -> "MyDecisionTreeClassifier":
        """Build a decision tree from training data."""
        if self.random_state is not None:
            random.seed(self.random_state)

        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset.")

        self.root_ = self._build_tree(
            X, y, depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split
        )
        return self

    def predict(self, X: List[List[float]]) -> List[Any]:
        """Predict class labels for X."""
        if self.root_ is None:
            raise ValueError("Decision tree has not been fitted yet.")

        return [self._predict_one(self.root_, x) for x in X]

    # ------------------------------------------------------------------
    # Internal recursive tree building
    # ------------------------------------------------------------------
    def _build_tree(
        self,
        X: List[List[float]],
        y: List[Any],
        depth: int,
        max_depth: Optional[int],
        min_samples_split: int
    ) -> _DTNode:
        # stopping conditions
        if len(set(y)) == 1:
            # pure node
            return _DTNode(label=y[0])

        if max_depth is not None and depth >= max_depth:
            return _DTNode(label=self._majority_class(y))

        if len(X) < min_samples_split:
            return _DTNode(label=self._majority_class(y))

        n_features = len(X[0])
        feature_indices = list(range(n_features))

        best_feature, best_threshold, best_gain = self._best_split(X, y, feature_indices)

        # if no useful split, make leaf
        if best_gain <= 0.0 or best_feature is None or best_threshold is None:
            return _DTNode(label=self._majority_class(y))

        # split dataset
        X_left, y_left, X_right, y_right = self._split_dataset(X, y, best_feature, best_threshold)

        if not X_left or not X_right:
            return _DTNode(label=self._majority_class(y))

        left_child = self._build_tree(
            X_left, y_left, depth + 1, max_depth, min_samples_split
        )
        right_child = self._build_tree(
            X_right, y_right, depth + 1, max_depth, min_samples_split
        )

        return _DTNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            label=None
        )

    def _best_split(
        self,
        X: List[List[float]],
        y: List[Any],
        feature_indices: List[int]
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find best (feature, threshold) by information gain."""
        base_entropy = entropy(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_samples = len(X)

        for j in feature_indices:
            values = sorted(set(x[j] for x in X))
            if len(values) <= 1:
                continue

            thresholds = [(values[k] + values[k + 1]) / 2.0
                          for k in range(len(values) - 1)]

            for threshold in thresholds:
                left_indices = [i for i in range(n_samples) if X[i][j] <= threshold]
                right_indices = [i for i in range(n_samples) if X[i][j] > threshold]

                if not left_indices or not right_indices:
                    continue

                y_left = [y[i] for i in left_indices]
                y_right = [y[i] for i in right_indices]

                n_left = len(y_left)
                n_right = len(y_right)

                ent_left = entropy(y_left)
                ent_right = entropy(y_right)

                weighted_entropy = (
                    (n_left / n_samples) * ent_left
                    + (n_right / n_samples) * ent_right
                )
                info_gain = base_entropy - weighted_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = j
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _split_dataset(
        self,
        X: List[List[float]],
        y: List[Any],
        feature_index: int,
        threshold: float
    ) -> Tuple[List[List[float]], List[Any], List[List[float]], List[Any]]:
        X_left, y_left, X_right, y_right = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[feature_index] <= threshold:
                X_left.append(xi)
                y_left.append(yi)
            else:
                X_right.append(xi)
                y_right.append(yi)
        return X_left, y_left, X_right, y_right

    def _predict_one(self, node: _DTNode, x: List[float]) -> Any:
        while not node.is_leaf():
            if node.feature_index is None or node.threshold is None:
                # safety: should not happen, but fall back to majority
                return node.label
            if x[node.feature_index] <= node.threshold:
                if node.left is None:
                    return node.label
                node = node.left
            else:
                if node.right is None:
                    return node.label
                node = node.right
        return node.label

    def _majority_class(self, y: List[Any]) -> Any:
        counts = Counter(y)
        max_count = max(counts.values())
        winners = [label for label, c in counts.items() if c == max_count]
        winners.sort(key=str)
        return winners[0]
