# TODO: your reusable general-purpose functions here
import numpy as np
from tabulate import tabulate
from mysklearn import myevaluation, myclassifiers

def evaluate_random_forest_cv(X, y, feature_names, positive_label,
                              n_splits=10, n_trees=100,
                              max_features=None,
                              stratify=True, random_state=0, shuffle=True):
    """Run k-fold CV for a custom random forest on (X, y).

    Returns dict with: feature_names, accuracy, error_rate,
    precision, recall, f1, labels, confusion_matrix.
    """
    # choose folds
    if stratify:
        folds = myevaluation.stratified_kfold_split(
            X, y, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )
    else:
        folds = myevaluation.kfold_split(
            X, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]

        rf = myclassifiers.MyRandomForestClassifier(
            n_trees=n_trees,
            max_features=max_features,
            random_state=random_state
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    acc = myevaluation.accuracy_score(all_y_true, all_y_pred)
    err = 1.0 - acc

    precision = myevaluation.binary_precision_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )
    recall = myevaluation.binary_recall_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )
    f1 = myevaluation.binary_f1_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )

    unique_labels = list(dict.fromkeys(all_y_true))
    labels = [positive_label] + [lab for lab in unique_labels if lab != positive_label]
    cm = myevaluation.confusion_matrix(all_y_true, all_y_pred, labels)

    return {
        "feature_names": feature_names,
        "accuracy": acc,
        "error_rate": err,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels,
        "confusion_matrix": cm
    }

def evaluate_knn_cv(X, y, feature_names, positive_label,
                    n_splits=10, n_neighbors=10,
                    stratify=True, random_state=0, shuffle=True):
    """Evaluate kNN with stratified k-fold CV.

    Returns dict with: feature_names, accuracy, error_rate,
    precision, recall, f1, labels, confusion_matrix.
    """
    # choose fold generator
    if stratify:
        folds = myevaluation.stratified_kfold_split(
            X, y, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )
    else:
        folds = myevaluation.kfold_split(
            X, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]

        # normalize using train stats
        X_train_norm, min_vals, max_vals = min_max_normalize(X_train)
        X_test_norm, _, _ = min_max_normalize(X_test, min_vals, max_vals)

        knn = myclassifiers.MyKNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_norm, y_train)
        y_pred = knn.predict(X_test_norm)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    acc = myevaluation.accuracy_score(all_y_true, all_y_pred)
    err = 1.0 - acc

    precision = myevaluation.binary_precision_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )
    recall = myevaluation.binary_recall_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )
    f1 = myevaluation.binary_f1_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )

    unique_labels = list(dict.fromkeys(all_y_true))
    labels = [positive_label] + [lab for lab in unique_labels if lab != positive_label]
    cm = myevaluation.confusion_matrix(all_y_true, all_y_pred, labels)

    return {
        "feature_names": feature_names,
        "accuracy": acc,
        "error_rate": err,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels,
        "confusion_matrix": cm
    }

def value_counts(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts

def score_to_risk_label(score, low_thresh, med_thresh):
    if score <= low_thresh:
        return "low"
    elif score <= med_thresh:
        return "medium"
    else:
        return "high"

def make_risk_score(row, header):
    """Compute a crude risk score from disease incidence and WASH coverage."""
    def get_val(col_name):
        idx = header.index(col_name)
        try:
            return float(row[idx])
        except Exception:
            return 0.0

    diarr = get_val("Diarrheal Cases per 100,000 people")
    chol  = get_val("Cholera Cases per 100,000 people")
    typh  = get_val("Typhoid Cases per 100,000 people")

    access_clean = get_val("Access to Clean Water (% of Population)")
    sanitation   = get_val("Sanitation Coverage (% of Population)")

    disease_score = diarr + chol + typh
    vulnerability = (100.0 - access_clean) + (100.0 - sanitation)

    return disease_score + vulnerability

def make_pollution_label(row, header, nitrate_hi, lead_hi, bact_hi, turb_hi):
    def gv(name):
        idx = header.index(name)
        try:
            return float(row[idx])
        except Exception:
            return None

    nitrate = gv("Nitrate Level (mg/L)")
    lead    = gv("Lead Concentration (µg/L)")
    bact    = gv("Bacteria Count (CFU/mL)")
    turb    = gv("Turbidity (NTU)")
    ph      = gv("pH Level")

    unsafe = 0
    if nitrate is not None and nitrate > nitrate_hi:
        unsafe += 1
    if lead is not None and lead > lead_hi:
        unsafe += 1
    if bact is not None and bact > bact_hi:
        unsafe += 1
    if turb is not None and turb > turb_hi:
        unsafe += 1
    if ph is not None and (ph < 6.0 or ph > 9.0):
        unsafe += 1

    return "yes" if unsafe >= 2 else "no"


def rf_choose_feature_indices(n_features, max_features, rng):
    """Return a sorted list of random feature indices for one tree."""
    if max_features is None:
        f = max(1, int(np.sqrt(n_features)))
    else:
        f = min(max_features, n_features)
    all_idx = np.arange(n_features)
    rng.shuffle(all_idx)
    return [int(all_idx[i]) for i in range(f)]

def rf_majority_vote_from_array(col_values):
    counts = {}
    n = col_values.shape[0]
    for i in range(n):
        lab = col_values[i]
        counts[lab] = counts.get(lab, 0) + 1
    best_label = None
    best_count = -1
    for lab in sorted(counts.keys()):
        c = counts[lab]
        if c > best_count:
            best_count = c
            best_label = lab
    return best_label

def evaluate_decision_tree_cv(X, y, feature_names, positive_label,
                              n_splits=10, stratify=True,
                              random_state=0, shuffle=True):
    """Run k-fold CV for a decision tree on (X, y).

    Args:
        X (list of list of obj): feature matrix (already subsetted).
        y (list of obj): labels.
        feature_names (list of str): names of the features in X (same order as columns).
        positive_label (obj): label to treat as the positive class for precision/recall/F1.
        n_splits (int): number of folds.
        stratify (bool): whether to use stratified k-fold.
        random_state (int): seed for reproducibility.
        shuffle (bool): whether to shuffle before splitting.

    Returns:
        dict with keys:
            "feature_names", "accuracy", "error_rate",
            "precision", "recall", "f1",
            "labels", "confusion_matrix"
    """
    # choose fold generator
    if stratify:
        folds = myevaluation.stratified_kfold_split(
            X, y, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )
    else:
        folds = myevaluation.kfold_split(
            X, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]

        dt = myclassifiers.MyDecisionTreeClassifier()
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # accuracy and error
    acc = myevaluation.accuracy_score(all_y_true, all_y_pred)
    err = 1.0 - acc

    # precision, recall, F1 for positive_label
    precision = myevaluation.binary_precision_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )
    recall = myevaluation.binary_recall_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )
    f1 = myevaluation.binary_f1_score(
        all_y_true, all_y_pred,
        labels=None, pos_label=positive_label
    )

    # confusion matrix; order labels so positive_label is first
    unique_labels = list(dict.fromkeys(all_y_true))  # preserve order
    labels = [positive_label] + [lab for lab in unique_labels if lab != positive_label]
    cm = myevaluation.confusion_matrix(all_y_true, all_y_pred, labels)

    return {
        "feature_names": feature_names,
        "accuracy": acc,
        "error_rate": err,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels,
        "confusion_matrix": cm
    }

def compute_attribute_domains(table, attr_indexes):
    domains = {i: set() for i in attr_indexes}
    for row in table:
        for i in attr_indexes:
            domains[i].add(row[i])
    return {i: sorted(list(vs)) for i, vs in domains.items()}

def all_same_class(labels):
    return len(set(labels)) == 1

def majority_class(labels):
    # count frequencies manually
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1

    # find max count
    max_count = max(counts.values())

    # get all labels with that max count
    candidates = [lab for lab, c in counts.items() if c == max_count]

    # tie-breaker: deterministic by sorted label
    candidates.sort()
    return candidates[0]

def count_majority(labels):
    maj = majority_class(labels)
    count = 0
    for lab in labels:
        if lab == maj:
            count += 1
    return count

def count_true(labels):
    return sum(1 for y in labels if y == "True" or y == "yes")

def entropy(labels):
    n = len(labels)
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * np.log2(p)
    return h

def select_attribute(table, available_attrs):
    base_labels = [row[-1] for row in table]
    base_entropy = entropy(base_labels)
    best_attr = None
    best_gain = None

    for a in available_attrs:
        subsets = {}
        for row in table:
            v = row[a]
            subsets.setdefault(v, []).append(row)
        remainder = 0.0
        n = len(table)
        for subset in subsets.values():
            w = len(subset) / n
            subset_labels = [r[-1] for r in subset]
            remainder += w * entropy(subset_labels)
        gain = base_entropy - remainder
        if best_gain is None or gain > best_gain:
            best_gain = gain
            best_attr = a

    return best_attr

def tdidt(table, available_attrs, attr_domains, attribute_names, parent_total=None):
    labels = [row[-1] for row in table]
    current_total = len(labels)

    # base case 1: all same class
    if all_same_class(labels):
        total_for_leaf = parent_total if parent_total is not None else current_total
        return ["Leaf", labels[0], count_majority(labels), total_for_leaf]

    # base case 2: no attributes left
    if not available_attrs:
        maj = majority_class(labels)
        total_for_leaf = parent_total if parent_total is not None else current_total
        return ["Leaf", maj, count_majority(labels), total_for_leaf]

    # choose best attribute by information gain
    best_attr = select_attribute(table, available_attrs)
    node = ["Attribute", attribute_names[best_attr]]

    # for each value in sorted domain
    for v in attr_domains[best_attr]:
        subset = [row for row in table if row[best_attr] == v]
        branch = ["Value", v]
        if not subset:
            # no rows → majority label of parent
            maj = majority_class(labels)
            branch.append(["Leaf", maj, count_majority(labels), current_total])
        else:
            new_attrs = [a for a in available_attrs if a != best_attr]
            subtree = tdidt(subset, new_attrs, attr_domains, attribute_names,
                            parent_total=current_total)
            branch.append(subtree)
        node.append(branch)

    return node


def predict_one(x, tree):
    node = tree
    while node[0] == "Attribute":
        att_name = node[1]          # 'att0', 'att1', ...
        att_index = int(att_name[3:])
        val = x[att_index]

        next_node = None
        for i in range(2, len(node)):
            value_node = node[i]    # ["Value", v, subtree]
            if value_node[1] == val:
                next_node = value_node[2]
                break
        if next_node is None:
            # unseen value: fall back to first leaf we can find
            for i in range(2, len(node)):
                cand = node[i][2]
                if cand[0] == "Leaf":
                    return cand[1]
            return None

        node = next_node
    return node[1]

def print_rules(tree, attribute_names, class_name):
    def recurse(node, conditions):
        if node[0] == "Leaf":
            label = node[1]
            if conditions:
                cond_str = " AND ".join(conditions)
                print(f"IF {cond_str} THEN {class_name} = {label}")
            else:
                print(f"{class_name} = {label}")
            return
        att_name = node[1]              # 'att0'
        att_index = int(att_name[3:])
        att_print = attribute_names[att_index]

        for i in range(2, len(node)):
            value_node = node[i]
            v = value_node[1]
            subtree = value_node[2]
            new_conds = conditions + [f"{att_print} == {repr(v)}"]
            recurse(subtree, new_conds)
    recurse(tree, [])

def cross_val_predict_nb_dummy(X, y, header, n_splits=10, stratify = False, random_state=None, shuffle=True):
    # Get folds of indices
    if stratify:
        folds = myevaluation.stratified_kfold_split(
            X, y, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )
    else:
        folds = myevaluation.kfold_split(
            X, n_splits=n_splits,
            random_state=random_state, shuffle=shuffle
        )

    nb_accs = []
    dummy_accs = []

    all_y_true_nb = []
    all_y_pred_nb = []
    all_y_true_dummy = []
    all_y_pred_dummy = []

    for train_idx, test_idx in folds:
        # Build train/test from indices
        X_train = [X[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test  = [y[i] for i in test_idx]

        # evaluate NB and Dummy on this fold
        y_pred_nb, y_pred_dummy, acc_nb, acc_dummy = evaluate_nb_and_dummy(
            X_train, X_test, y_train, y_test, header)

        nb_accs.append(acc_nb)
        dummy_accs.append(acc_dummy)

        # collect labels for confusion matrices
        all_y_true_nb.extend(y_test)
        all_y_pred_nb.extend(y_pred_nb)

        all_y_true_dummy.extend(y_test)
        all_y_pred_dummy.extend(y_pred_dummy)

    avg_nb_acc = sum(nb_accs) / len(nb_accs) if nb_accs else 0.0
    avg_dummy_acc = sum(dummy_accs) / len(dummy_accs) if dummy_accs else 0.0

    return avg_nb_acc, avg_dummy_acc, \
           all_y_true_nb, all_y_pred_nb, \
           all_y_true_dummy, all_y_pred_dummy

def evaluate_nb_and_dummy(X_train, X_test, y_train, y_test, header = None):
    # Naive Bayes
    nb = myclassifiers.MyNaiveBayesClassifier()
    if header is not None:
        nb.header = header
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = myevaluation.accuracy_score(y_test, y_pred_nb)

    # Dummy
    dummy_clf = myclassifiers.MyDummyClassifier()
    dummy_clf.fit(None, y_train)
    y_pred_dummy = dummy_clf.predict([None] * len(y_test))
    acc_dummy = myevaluation.accuracy_score(y_test, y_pred_dummy)
    
    return y_pred_nb, y_pred_dummy, acc_nb, acc_dummy

def print_confusion_table(header, cm, labels, title):
    print(title)

    # header: MPG Ranking + label columns + Total + Recognition (%)
    i = 0
    while i < len(labels):
        header.append(labels[i])
        i += 1
    header.append("Total")
    header.append("Recognition (%)")

    rows = []
    i = 0
    while i < len(labels):
        true_label = labels[i]
        row_counts = cm[i]

        # total for this true class
        total = 0
        j = 0
        while j < len(row_counts):
            total += row_counts[j]
            j += 1

        # recognition = diagonal / total * 100
        if total > 0:
            correct = row_counts[i]
            recog = correct / float(total) * 100.0
        else:
            recog = 0.0

        # build row: true label + counts + total + recognition
        row = [true_label]
        j = 0
        while j < len(row_counts):
            row.append(row_counts[j])
            j += 1
        row.append(total)
        row.append(f"{recog:.0f}")

        rows.append(row)
        i += 1

    print(tabulate(rows, headers=header, tablefmt="plain"))
    print()


def bootstrap_method(X, y, k=10, n_samples=None, random_state=None):
    knn_accs = []
    dummy_accs = []

    for run in range (k):
        # make each bootstrap sample different but reproducible
        rs = None if random_state is None else (random_state + run)
        
        X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(X, y, n_samples, random_state = rs)

        acc_knn, acc_dummy, _, _ = evaluate_knn_and_dummy(X_train, X_test, y_train, y_test)

        knn_accs.append(acc_knn)
        dummy_accs.append(acc_dummy)

    avg_knn_acc = sum(knn_accs) / len(knn_accs)
    avg_dummy_acc = sum(dummy_accs) / len(dummy_accs)

    return avg_knn_acc, avg_dummy_acc

def cross_val_predict(X, y, n_splits=10, stratify = False, random_state=None, shuffle=True):
    # Get folds of indicies
    if stratify:
        folds = myevaluation.stratified_kfold_split(X, y, n_splits = n_splits, random_state = random_state, shuffle = shuffle)
    else:
        folds = myevaluation.kfold_split(X, n_splits = n_splits, random_state = random_state, shuffle = shuffle)

    knn_accs = []
    dummy_accs = []

    all_y_true_knn = []
    all_y_pred_knn = []
    all_y_true_dummy = []
    all_y_pred_dummy = []

    for train_idx, test_idx in folds:
        # Build train/test from indices
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        acc_knn, acc_dummy, y_pred_knn, y_pred_dummy = evaluate_knn_and_dummy(X_train, X_test, y_train, y_test)

        knn_accs.append(acc_knn)
        dummy_accs.append(acc_dummy)
        
        # collect labels for confusion matrices
        # same y_test for both classifiers
        all_y_true_knn.extend(y_test)
        all_y_pred_knn.extend(y_pred_knn)

        all_y_true_dummy.extend(y_test)
        all_y_pred_dummy.extend(y_pred_dummy)

    avg_knn_acc = sum(knn_accs) / len(knn_accs)
    avg_dummy_acc = sum(dummy_accs) / len(dummy_accs)

    return avg_knn_acc, avg_dummy_acc, all_y_true_knn, all_y_pred_knn, all_y_true_dummy, all_y_pred_dummy      


def random_subsample(X, y, k=10, test_size=0.33, random_state=None):
    knn_accs = []
    dummy_accs = []

    for run in range(k):
        rs = None if random_state is None else (random_state + run)

        # Split indices via train_test_split
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=rs, shuffle=True)

        acc_knn, acc_dummy, _, _ = evaluate_knn_and_dummy(X_train, X_test, y_train, y_test)

        knn_accs.append(acc_knn)
        dummy_accs.append(acc_dummy)

    avg_knn_acc = sum(knn_accs) / len(knn_accs)
    avg_dummy_acc = sum(dummy_accs) / len(dummy_accs)

    return avg_knn_acc, avg_dummy_acc

def evaluate_knn_and_dummy(X_train, X_test, y_train, y_test, n_neighbors=10):
    # Normalize using train stats
    X_train_norm, min_vals, max_vals = min_max_normalize(X_train)
    X_test_norm, _, _ = min_max_normalize(X_test, min_vals, max_vals)

    # kNN
    knn = myclassifiers.MyKNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train_norm, y_train)
    y_pred_knn = knn.predict(X_test_norm)

    # Dummy
    dummy_clf = myclassifiers.MyDummyClassifier()
    dummy_clf.fit(None, y_train)
    y_pred_dummy = dummy_clf.predict([None] * len(y_test))

    # Accuracies
    acc_knn = myevaluation.accuracy_score(y_test, y_pred_knn)
    acc_dummy = myevaluation.accuracy_score(y_test, y_pred_dummy)
    
    return acc_knn, acc_dummy, y_pred_knn, y_pred_dummy

def simple_bootstrap_indices(n, n_samples = None, random_state = None):
    if n_samples is None:
        n_samples = n
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size = n_samples, replace = True)

def get_oob_indices(n, sample_indices):
    all_indices = set(range(n))
    sample_set = set(sample_indices)
    oob_indices = list(all_indices - sample_set)
    return oob_indices

def split_indices_evenly(idxs, n_splits):
    splits = []
    n = len(idxs)
    base = n // n_splits
    rem = n % n_splits
    start = 0
    for k in range(n_splits):
        size = base + (1 if k < rem else 0)
        end = start + size
        splits.append(idxs[start:end])
        start = end
    return splits

def my_discretizer(y):
    return "high" if y >= 100 else "low"

def compute_euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions.")
    squared_sum = 0
    for i in range (len(point1)):
        diff = point1[i] - point2[i]
        squared_sum += pow(diff, 2)
    return pow(squared_sum, 0.5)

def mpg_to_doe_rating(mpg):
    try:
        mpg_value = float(mpg)
    except Exception:
        return None
    if mpg_value >= 45:
        return 10
    elif 37 <= mpg_value < 45:
        return 9
    elif 31 <= mpg_value < 37:
        return 8
    elif 27 <= mpg_value < 31:
        return 7
    elif 24 <= mpg_value < 27:
        return 6
    elif 20 <= mpg_value < 24:
        return 5
    elif 17 <= mpg_value < 20:
        return 4
    elif 15 <= mpg_value < 17:
        return 3
    elif mpg_value == 14:
        return 2
    elif mpg_value < 14:
        return 1
    else:
        return None  

def add_column(table, column_name, new_column_data):
    """
    Add a new column to the table.
    Parameters:
        table: The table object with 'column_names' and 'data'.
        column_name: Name of the new column (str).
        new_column_data: List of new column values (same length as the number of rows).
    """
    if column_name in table.column_names:
        print(f"Column '{column_name}' already exists. Skipping addition.")
        return
    if len(table.data) != len(new_column_data):
        raise ValueError("Length of new_column_data must match number of rows in table.")
    table.column_names.append(column_name)
    i = 0
    while i < len(table.data):
        table.data[i].append(new_column_data[i])
        i += 1

def remove_column(table, column_name):
    col_index = table.column_names.index(column_name)
    table.column_names.pop(col_index)
    i = 0
    while i < len(table.data):
        table.data[i].pop(col_index)
        i+=1

def evaluate_linear_classifier(test_table, y_pred_ranks, actual_ranks):
    results = []
    correct = 0
    for i in range(len(y_pred_ranks)):
        features = test_table.data[i]
        pred = y_pred_ranks[i]
        actual = actual_ranks[i]
        results.append([features, pred, actual])
        if pred == actual:
            correct += 1
    accuracy = correct / len(y_pred_ranks)
    return results, accuracy

def min_max_normalize(X, min_vals=None, max_vals=None):
    if min_vals is None or max_vals is None:
        min_vals = [min([row[j] for row in X]) for j in range(len(X[0]))]
        max_vals = [max([row[j] for row in X]) for j in range(len(X[0]))]
    X_norm = []
    for row in X:
        norm_row = []
        for j in range(len(row)):
            val = row[j]
            denom = max_vals[j] - min_vals[j]
            if denom != 0:
                val = (val - min_vals[j]) / denom
            else:
                val = 0.0
            norm_row.append(val)
        X_norm.append(norm_row)
    return X_norm, min_vals, max_vals

def project_table(table, keep_cols):
    keep_idx = [table.column_names.index(c) for c in keep_cols]
    new = MyPyTable()
    new.column_names = keep_cols
    new.data = [
        [row[i] for i in keep_idx]
        for row in table.data
    ]
    return new