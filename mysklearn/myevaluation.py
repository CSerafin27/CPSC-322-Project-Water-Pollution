import numpy as np
from mysklearn import myutils

# TODO: copy your myevaluation.py solution from PA5 here
def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    n_samples = len(X)

    # Validate length
    if len(y) != n_samples:
        raise ValueError("X and y must have the same length")

    # Determine the number of test samples
    if isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size as float must be in (0, 1)")
        raw = n_samples * test_size
        n_test = int(raw)
        if n_test < raw:
            n_test += 1
    else:
        n_test = int(test_size)
        if n_test <= 0 or n_test >= n_samples:
            raise ValueError("test_size as int must be in (0, n_samples)")
    
    n_train = n_samples - n_test

    # Generate indicies
    indices = np.arange(n_samples)

    # Shuffle if needed
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
                # when shuffled: take first part as test, rest as train
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        # when not shuffled: keep original order, train first, test last
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    # Select sample
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    # Shuffle if needed
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
    
    # Determine fold_size
    fold_sizes = [n_samples//n_splits] * n_splits
    for i in range (n_samples % n_splits):
        fold_sizes[i] += 1 # Distribute remainder 

    # Split indicies into folds
    current = 0
    folds = []
    for fold_size in fold_sizes:
        # Get indicies for the test_fold
        test_idx = list(indices[current:current+fold_size])
        #Get indices for the training folds: everything before and after the current test fold
        train_idx = list(indices[:current]) + list(indices[current + fold_size:])
        # Add a tuple to the next fold
        folds.append((train_idx, test_idx))
        # Move to the next fold
        current += fold_size
    return folds 

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    n_samples = len(X)
    if n_samples != len(y):
        raise ValueError("X and y must have the same length")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    indices = np.arange(n_samples)
    
    # Shuffle if needed 
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
    
    # find unique class labels
    unique_labels = []
    for label in y:
        if label not in unique_labels:
            unique_labels.append(label)
    
    # build per-class index lists
    label_indices = []
    for label in unique_labels:
        cur_idxs = []
        for idx in indices:
            if y[idx] == label:
                cur_idxs.append(idx)
        label_indices.append(cur_idxs)

    # for each class, split its indices into n_split buckets
    per_label_folds = []
    for class_idxs in label_indices:
        if shuffle:
            rng.shuffle(class_idxs)
        splits = myutils.split_indices_evenly(class_idxs, n_splits)
        per_label_folds.append(splits)
    
    folds = []
    for fold_idx in range(n_splits):
        # test indices for this fold
        test_idx = []
        for class_splits in per_label_folds:
            test_idx.extend(class_splits[fold_idx])

        # train indices = all indices not in test_idx
        train_idx = [i for i in indices if i not in test_idx]

        folds.append((train_idx, test_idx))        
    return folds # TODO: (BONUS) fix this

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    n = len(X)
    
    # default n_samples: size of dataset (standard bootstrap)
    if n_samples is None:
        n_samples = n
    
    # get bootstrap indices and OOB indices
    sample_indices = myutils.simple_bootstrap_indices(n, n_samples, random_state)
    oob_indices = myutils.get_oob_indices(n, sample_indices)

    # X samples
    X_sample = [X[i] for i in sample_indices]
    X_out_of_bag = [X[i] for i in oob_indices]
    
    # y samples (if provided)
    if y is not None:
        y_sample = [y[i] for i in sample_indices]
        y_out_of_bag = [y[i] for i in oob_indices]
    else:
        y_sample = None
        y_out_of_bag = None
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag # TODO: fix this

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    n_classes = len(labels)

    # map each label to its index 
    label_to_index = {}
    for idx in range(n_classes):
        label_to_index[labels[idx]] = idx

    # initialize matrix of zeros (list of lists)
    matrix = []
    for _ in range(n_classes):
        row = []
        for _ in range(n_classes):
            row.append(0)
        matrix.append(row)

    # fill counts: rows = true, cols = predicted 
    n_samples = len(y_true)
    for k in range(n_samples):
        yt = y_true[k]
        yp = y_pred[k]
        if yt in label_to_index and yp in label_to_index:
            i = label_to_index[yt]   # true label index (row)
            j = label_to_index[yp]   # predicted label index (column)
            matrix[i][j] += 1    
    return matrix # TODO: fix this

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    n_samples = len(y_true)
    correct = 0

    for i in range(n_samples):
        if y_true[i] == y_pred[i]:
            correct += 1

    if normalize:
        # fraction of correctly classified samples
        return correct / float(n_samples) if n_samples > 0 else 0.0
    else:
        # number of correctly classified samples
        return correct
    return 0.0 # TODO: fix this

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
        # derive labels from y_true if not given
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    # choose which label is "positive"
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0

    # count true positives and false positives for pos_label
    for i in range(len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                tp += 1
            else:
                fp += 1

    denom = tp + fp
    if denom == 0:
        return 0.0

    return tp / denom # TODO: fix this

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    # choose positive label
    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fn = 0

    for i in range(len(y_true)):
        yt = y_true[i]
        yp = y_pred[i]

        if yt == pos_label:
            if yp == pos_label:
                tp += 1      # true positive
            else:
                fn += 1      # false negative

    denom = tp + fn
    if denom == 0:
        return 0.0

    return tp / denom # TODO: fix this

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
        # derive labels if needed
    if labels is None:
        labels = []
        for y in y_true:
            if y not in labels:
                labels.append(y)

    # choose positive label
    if pos_label is None:
        pos_label = labels[0]

    tp = fp = fn = 0

    for i in range(len(y_true)):
        yt = y_true[i]
        yp = y_pred[i]

        if yp == pos_label:
            if yt == pos_label:
                tp += 1
            else:
                fp += 1
        else:
            if yt == pos_label:
                fn += 1

    # precision and recall
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0.0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

    