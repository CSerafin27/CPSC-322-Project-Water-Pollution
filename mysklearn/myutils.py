import math
import random
from collections import Counter
from typing import List, Tuple, Any, Optional


def stratified_train_test_split(
    X: List[List[float]],
    y: List[Any],
    test_size: float = 1/3,
    random_state: Optional[int] = None
) -> Tuple[List[List[float]], List[List[float]], List[Any], List[Any]]:
    """Stratified train/test split.
    Keeps the class distribution approximately the same in train and test.
    Returns: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        random.seed(random_state)

    # group indices by class
    label_to_indices = {}
    for idx, label in enumerate(y):
        label_to_indices.setdefault(label, []).append(idx)

    train_indices = []
    test_indices = []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        n_total = len(indices)
        n_test = max(1, int(round(n_total * test_size)))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    # shuffle final index lists
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def bootstrap_sample(
    X: List[List[float]],
    y: List[Any],
    random_state: Optional[int] = None
) -> Tuple[List[List[float]], List[List[float]], List[Any], List[Any]]:
    """Generate a bootstrap sample and a corresponding out-of-bag set.
    Returns: X_boot, X_oob, y_boot, y_oob
    """
    if random_state is not None:
        random.seed(random_state)

    n = len(X)
    indices = list(range(n))
    boot_indices = [random.choice(indices) for _ in range(n)]
    oob_indices = sorted(set(indices) - set(boot_indices))

    X_boot = [X[i] for i in boot_indices]
    y_boot = [y[i] for i in boot_indices]
    X_oob = [X[i] for i in oob_indices]
    y_oob = [y[i] for i in oob_indices]

    return X_boot, X_oob, y_boot, y_oob


def entropy(y: List[Any]) -> float:
    """Compute entropy of a label vector."""
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    ent = 0.0
    for count in counts.values():
        p = count / n
        ent -= p * math.log2(p)
    return ent


def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """Simple accuracy metric."""
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true) if y_true else 0.0


def confusion_matrix(
    y_true: List[Any],
    y_pred: List[Any],
    labels: Optional[List[Any]] = None
) -> List[List[int]]:
    """Compute confusion matrix as a 2D list (labels x labels)."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    label_to_index = {label: i for i, label in enumerate(labels)}

    matrix = [[0 for _ in labels] for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        i = label_to_index[yt]
        j = label_to_index[yp]
        matrix[i][j] += 1
    return matrix


# ----------------------------------------------------------------------
# Distance helpers for KNN and others
# ----------------------------------------------------------------------
def euclidean_distance(x1: List[float], x2: List[float]) -> float:
    """Compute Euclidean distance between two numeric vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))


def manhattan_distance(x1: List[float], x2: List[float]) -> float:
    """Compute Manhattan (L1) distance between two numeric vectors."""
    return sum(abs(a - b) for a, b in zip(x1, x2))
