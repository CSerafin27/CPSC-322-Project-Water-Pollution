from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .myutils import (
    stratified_train_test_split,
    bootstrap_sample,
    entropy,
    accuracy_score,
)


@dataclass
class _TreeNode:
    """Internal representation of a decision tree node."""
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    label: Optional[Any] = None  # only set for leaf nodes

    def is_leaf(self) -> bool:
        return self.label is not None


class MyRandomForestClassifier:
    """Random Forest classifier implemented from scratch.

    Assumptions:
    - X is a list of list of numeric features (floats/ints).
    - y is a list of discrete class labels.
    - You should perform any needed encoding / scaling before calling fit().
    """

    def __init__(
        self,
        N: int = 20,           # number of trees to generate
        M: int = 7,            # number of best trees to keep
        F: int = 2,            # number of random attributes per split
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = None
    ) -> None:
        self.N = N
        self.M = M
        self.F = F
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.trees: List[_TreeNode] = []
        self.X_test_: List[List[float]] = []
        self.y_test_: List[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: List[List[float]], y: List[Any]) -> "MyRandomForestClassifier":
        """Fit the random forest model.

        1) Stratified 1/3 test split (held out for final evaluation).
        2) Over the remaining 2/3 (remainder set), generate N trees via
           bootstrap sampling and random feature selection.
        3) Evaluate each tree on its validation (out-of-bag) data.
        4) Keep the top M trees by validation accuracy.
        """
        if self.random_state is not None:
            random.seed(self.random_state)

        # 1) stratified test split
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_size=1 / 3, random_state=self.random_state
        )
        self.X_test_ = X_test
        self.y_test_ = y_test

        # 2) generate N trees
        candidates: List[Tuple[_TreeNode, float]] = []  # (tree, val_accuracy)

        for i in range(self.N):
            # bootstrap over the remainder set (X_train, y_train)
            X_boot, X_val, y_boot, y_val = bootstrap_sample(
                X_train, y_train, random_state=None
            )

            # choose random subset of features to consider at each split
            n_features = len(X_boot[0])
            F = min(self.F, n_features)

            # build tree
            tree = self._build_tree(
                X_boot,
                y_boot,
                depth=0,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features_to_sample=F
            )

            # evaluate on validation (out-of-bag) data
            if X_val:
                y_val_pred = [self._predict_one(tree, x) for x in X_val]
                val_acc = accuracy_score(y_val, y_val_pred)
            else:
                # if, for some corner case, there is no OOB data, skip evaluation
                val_acc = 0.0

            candidates.append((tree, val_acc))

        # 3) select top M trees
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        self.trees = [tree for tree, _ in candidates[: self.M]]

        return self

    def predict(self, X: List[List[float]]) -> List[Any]:
        """Predict class labels for X using majority vote."""
        if not self.trees:
            raise ValueError("Random forest has not been fitted yet.")

        predictions = []
        for x in X:
            # get all tree predictions
            votes = [self._predict_one(tree, x) for tree in self.trees]
            # majority vote
            counts = Counter(votes)
            # if tie, choose winner consistently (e.g., sorted label)
            max_count = max(counts.values())
            winners = [label for label, c in counts.items() if c == max_count]
            winners.sort(key=str)
            predictions.append(winners[0])
        return predictions

    # ------------------------------------------------------------------
    # Internal decision tree implementation
    # ------------------------------------------------------------------
    def _build_tree(
        self,
        X: List[List[float]],
        y: List[Any],
        depth: int,
        max_depth: Optional[int],
        min_samples_split: int,
        n_features_to_sample: int
    ) -> _TreeNode:
        """Recursively build a decision tree using entropy and
        random feature selection at each node.
        """
        # stopping conditions
        if len(set(y)) == 1:
            return _TreeNode(label=y[0])

        if max_depth is not None and depth >= max_depth:
            return _TreeNode(label=self._majority_class(y))

        if len(X) < min_samples_split:
            return _TreeNode(label=self._majority_class(y))

        n_features = len(X[0])

        # choose random subset of features
        feature_indices = list(range(n_features))
        random.shuffle(feature_indices)
        feature_indices = feature_indices[:n_features_to_sample]

        # find best split among these features
        best_feature, best_threshold, best_gain = self._best_split(
            X, y, feature_indices
        )

        # if no useful split, make a leaf
        if best_gain <= 0 or best_feature is None or best_threshold is None:
            return _TreeNode(label=self._majority_class(y))

        # partition data based on best split
        X_left, y_left, X_right, y_right = self._split_dataset(
            X, y, best_feature, best_threshold
        )

        if not X_left or not X_right:
            return _TreeNode(label=self._majority_class(y))

        # recursively build children
        left_child = self._build_tree(
            X_left,
            y_left,
            depth + 1,
            max_depth,
            min_samples_split,
            n_features_to_sample
        )
        right_child = self._build_tree(
            X_right,
            y_right,
            depth + 1,
            max_depth,
            min_samples_split,
            n_features_to_sample
        )

        return _TreeNode(
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
        """Find the best (feature, threshold) split among the given features."""
        base_entropy = entropy(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_samples = len(X)

        for j in feature_indices:
            # get all distinct values of feature j
            values = sorted(set(x[j] for x in X))
            if len(values) <= 1:
                continue

            # candidate thresholds = midpoints between consecutive values
            thresholds = [
                (values[k] + values[k + 1]) / 2.0
                for k in range(len(values) - 1)
            ]

            for threshold in thresholds:
                # split
                left_indices = [i for i in range(n_samples) if X[i][j] <= threshold]
                right_indices = [i for i in range(n_samples) if X[i][j] > threshold]

                if not left_indices or not right_indices:
                    continue

                y_left = [y[i] for i in left_indices]
                y_right = [y[i] for i in right_indices]

                # compute information gain
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
        """Split X, y into left/right subsets using feature_index <= threshold."""
        X_left, y_left, X_right, y_right = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[feature_index] <= threshold:
                X_left.append(xi)
                y_left.append(yi)
            else:
                X_right.append(xi)
                y_right.append(yi)
        return X_left, y_left, X_right, y_right

    def _predict_one(self, tree: _TreeNode, x: List[float]) -> Any:
        """Traverse a tree to classify a single instance x."""
        node = tree
        while not node.is_leaf():
            # safety: if something is missing, return majority label of test set
            if node.feature_index is None or node.threshold is None:
                return self._majority_class(self.y_test_) if self.y_test_ else None

            if x[node.feature_index] <= node.threshold:
                if node.left is None:
                    return self._majority_class(self.y_test_) if self.y_test_ else None
                node = node.left
            else:
                if node.right is None:
                    return self._majority_class(self.y_test_) if self.y_test_ else None
                node = node.right

        return node.label

    def _majority_class(self, y: List[Any]) -> Any:
        """Return the most common class label in y."""
        counts = Counter(y)
        max_count = max(counts.values())
        winners = [label for label, c in counts.items() if c == max_count]
        winners.sort(key=str)
        return winners[0]
