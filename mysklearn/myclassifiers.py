from mysklearn import myutils, myevaluation
import numpy as np

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for test_instance in X_test:
            dists =  [myutils.compute_euclidean_distance(test_instance, train_instance) for train_instance in self.X_train]
            k_smallest_dists = []
            k_smallest_indices = []
            
            for n in range(self.n_neighbors):
                min_val = float('inf')
                min_index = -1
        
                for i in range(len(dists)):
                    if i not in k_smallest_indices and dists[i] < min_val:
                        min_val = dists[i]
                        min_index = i

                if min_index != -1:
                    k_smallest_dists.append(min_val)
                    k_smallest_indices.append(min_index)
            distances.append(k_smallest_dists)
            neighbor_indices.append(k_smallest_indices)

        return distances, neighbor_indices 

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []
        for i in range(len(neighbor_indices)):
            label_counts = {}
            for neighbor_index in neighbor_indices[i]:
                label = self.y_train[neighbor_index]
                if label not in label_counts:
                    label_counts[label] = 1
                else:
                    label_counts[label] += 1
            max_count = -1
            majority_label = None
            for label in label_counts:
                if label_counts[label] > max_count:
                    max_count = label_counts[label]
                    majority_label = label
            y_predicted.append(majority_label)
        return y_predicted

class MyRandomForestClassifier:
    def __init__(self, n_trees=20, max_trees=None, max_features=None, random_state=None):
        self.n_trees = n_trees          # N
        self.max_trees = max_trees      # M
        self.max_features = max_features  # F
        self.random_state = random_state

        self.rng_ = np.random.RandomState(random_state)
        self.forest_ = []  # list of (tree, feature_indices)

    def fit(self, X_train, y_train):
        if len(X_train) == 0:
            raise ValueError("X_train is empty")
        n_features = len(X_train[0])

        if self.max_trees is None or self.max_trees > self.n_trees:
            self.max_trees = self.n_trees

        candidates = []
        accuracies = []

        for _ in range(self.n_trees):
            seed = int(self.rng_.randint(0, 2**31 - 1))
            X_boot, X_oob, y_boot, y_oob = myevaluation.bootstrap_sample(
                X_train, y_train, random_state=seed
            )

            feat_idx = myutils.rf_choose_feature_indices(
                n_features, self.max_features, self.rng_
            )

            X_boot_sub = [[row[j] for j in feat_idx] for row in X_boot]
            tree = MyDecisionTreeClassifier()
            tree.fit(X_boot_sub, y_boot)

            if X_oob and y_oob:
                X_oob_sub = [[row[j] for j in feat_idx] for row in X_oob]
                y_oob_pred = tree.predict(X_oob_sub)
                acc = myevaluation.accuracy_score(y_oob, y_oob_pred)
            else:
                acc = 0.0

            candidates.append((tree, feat_idx))
            accuracies.append(acc)

        accuracies = np.array(accuracies)
        best_idx = np.argsort(-accuracies)[: self.max_trees]
        self.forest_ = [candidates[int(i)] for i in best_idx]
        return self

    def predict(self, X_test):
        if not self.forest_:
            raise ValueError("MyRandomForestClassifier has not been fitted.")

        n_samples = len(X_test)
        all_preds = []

        for tree, feat_idx in self.forest_:
            X_sub = [[row[j] for j in feat_idx] for row in X_test]
            preds = tree.predict(X_sub)   # list of labels
            all_preds.append(preds)

        # convert to 2D NumPy array of shape (M, n_samples)
        all_preds = np.array(all_preds, dtype=object)

        y_pred = []
        for i in range(n_samples):
            col_vals = all_preds[:, i]  # 1D array (M,)
            y_pred.append(myutils.rf_majority_vote_from_array(col_vals))

        return y_pred

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        n_features = len(X_train[0])
        self.attribute_names_ = [f"att{i}" for i in range(n_features)]
        available_attrs = list(range(n_features))

        # table = X with label appended as last column
        table = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        attr_domains = myutils.compute_attribute_domains(table, available_attrs)

        self.tree = myutils.tdidt(table, available_attrs, attr_domains, self.attribute_names_, parent_total=len(table))
        return self

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [myutils.predict_one(x, self.tree) for x in X_test]

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = self.attribute_names_
        myutils.print_rules(self.tree, attribute_names, class_name)

   