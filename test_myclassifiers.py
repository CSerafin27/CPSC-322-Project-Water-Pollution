import numpy as np
from scipy import stats
from mysklearn import myutils, myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn.myclassifiers import MyRandomForestClassifier

def test_decision_tree_classifier_fit():
    # Test Case 1
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"], # False
        ["Senior", "Java", "no", "yes"], # False
        ["Mid", "Python", "no", "no"], # True
        ["Junior", "Python", "no", "no"], # True 
        ["Junior", "R", "yes", "no"], # True
        ["Junior", "R", "yes", "yes"], # False
        ["Mid", "R", "yes", "yes"], # True
        ["Senior", "Python", "no", "no"], # False
        ["Senior", "R", "yes", "no"], # True
        ["Junior", "Python", "yes", "no"], # True 
        ["Senior", "Python", "yes", "yes"], # True
        ["Mid", "Python", "no", "yes"], # True
        ["Mid", "Java", "yes", "no"], # True
        ["Junior", "Python", "no", "yes"] # False
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
    ["Attribute", "att0",
        ["Value", "Junior",
            ["Attribute", "att3",
                ["Value", "no",
                    ["Leaf", "True", 3, 5]
                ],
                ["Value", "yes",
                    ["Leaf", "False", 2, 5]
                ]
            ]
        ],
        ["Value", "Mid",
            ["Leaf", "True", 4, 14]
        ],
        ["Value", "Senior",
            ["Attribute", "att2",
                ["Value", "no",
                    ["Leaf", "False", 3, 5]
                ],
                ["Value", "yes",
                    ["Leaf", "True", 2, 5]
                ]
            ]
        ]
    ]
    dt = MyDecisionTreeClassifier()
    dt.fit(X_train_interview, y_train_interview)
    assert dt.tree == tree_interview
    print(f"Interview tree: {dt.tree}")
    
    # Test Case 2
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"], # no
        [1, 3, "excellent"], # no
        [2, 3, "fair"], # yes
        [2, 2, "fair"], # yes
        [2, 1, "fair"], # yes
        [2, 1, "excellent"], # no
        [2, 1, "excellent"], # yes
        [1, 2, "fair"], # no
        [1, 1, "fair"], # yes
        [2, 2, "fair"], # yes
        [1, 2, "excellent"], # yes
        [2, 2, "excellent"], # yes
        [2, 3, "fair"], # yes 
        [2, 2, "excellent"], # no
        [2, 3, "fair"] # yes
    ]
    
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    tree_iphone = \
    ['Attribute', 'att0', 
        ['Value', 1, 
            ['Attribute', 'att1', 
                ['Value', 1, 
                    ['Leaf', 'yes', 1, 5]], 
                ['Value', 2, 
                    ['Attribute', 'att2', 
                        ['Value', 'excellent', 
                            ['Leaf', 'yes', 1, 2]], 
                        ['Value', 'fair', 
                            ['Leaf', 'no', 1, 2]]]], 
                ['Value', 3, 
                    ['Leaf', 'no', 2, 5]]]], 
        ['Value', 2, 
            ['Attribute', 'att2', 
                ['Value', 'excellent', 
                    ['Attribute', 'att1', 
                        ['Value', 1,
                            ['Leaf', 'no', 1, 4]], 
                        ['Value', 2, 
                            ['Leaf', 'no', 1, 4]], 
                        ['Value', 3, 
                            ['Leaf', 'no', 2, 4]]]], 
                ['Value', 'fair', 
                    ['Leaf', 'yes', 6, 10]
                ]
            ]
        ]
    ]
    dt = MyDecisionTreeClassifier()
    dt.fit(X_train_iphone, y_train_iphone)
    assert dt.tree == tree_iphone

def test_decision_tree_classifier_predict():
    # Test case 1: Interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"], # False
        ["Senior", "Java", "no", "yes"], # False
        ["Mid", "Python", "no", "no"], # True
        ["Junior", "Python", "no", "no"], # True
        ["Junior", "R", "yes", "no"], # True
        ["Junior", "R", "yes", "yes"], # False
        ["Mid", "R", "yes", "yes"], # True
        ["Senior", "Python", "no", "no"], # False
        ["Senior", "R", "yes", "no"], # True
        ["Junior", "Python", "yes", "no"], # True
        ["Senior", "Python", "yes", "yes"], # True
        ["Mid", "Python", "no", "yes"], # True
        ["Mid", "Java", "yes", "no"], # True
        ["Junior", "Python", "no", "yes"] # False
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
    ["Attribute", "att0",
        ["Value", "Junior",
            ["Attribute", "att3",
                ["Value", "no",
                    ["Leaf", "True", 3, 5]
                ],
                ["Value", "yes",
                    ["Leaf", "False", 2, 5]
                ]
            ]
        ],
        ["Value", "Mid",
            ["Leaf", "True", 4, 14]
        ],
        ["Value", "Senior",
            ["Attribute", "att2",
                ["Value", "no",
                    ["Leaf", "False", 3, 5]
                ],
                ["Value", "yes",
                    ["Leaf", "True", 2, 5]
                ]
            ]
        ]
    ]

    dt = MyDecisionTreeClassifier()
    dt.fit(X_train_interview, y_train_interview)
    ## First unseen instance: ["Junior", "Java", "yes", "no"]
    X = [["Junior", "Java", "yes", "no"]]
    y_expected = ["True"]
    assert y_expected == dt.predict(X)

    ## Second unseen instance: ["Junior", "Java", "yes", "no"]
    X = [["Junior", "Java", "yes", "yes"]]
    y_expected = ["False"]
    assert y_expected == dt.predict(X)

    # Test case 2: Iphone dataset
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"], # no
        [1, 3, "excellent"], # no
        [2, 3, "fair"], # yes
        [2, 2, "fair"], # yes
        [2, 1, "fair"], # yes
        [2, 1, "excellent"], # no
        [2, 1, "excellent"], # yes
        [1, 2, "fair"], # no
        [1, 1, "fair"], # yes
        [2, 2, "fair"], # yes
        [1, 2, "excellent"], # yes
        [2, 2, "excellent"], # yes
        [2, 3, "fair"], # yes 
        [2, 2, "excellent"], # no
        [2, 3, "fair"] # yes
    ]
    
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]
    tree_iphone = \
    ['Attribute', 'att0', 
        ['Value', 1, 
            ['Attribute', 'att1', 
                ['Value', 1, 
                    ['Leaf', 'yes', 1, 5]], 
                ['Value', 2, 
                    ['Attribute', 'att2', 
                        ['Value', 'excellent', 
                            ['Leaf', 'yes', 1, 2]], 
                        ['Value', 'fair', 
                            ['Leaf', 'no', 1, 2]]]], 
                ['Value', 3, 
                    ['Leaf', 'no', 2, 5]]]], 
        ['Value', 2, 
            ['Attribute', 'att2', 
                ['Value', 'excellent', 
                    ['Attribute', 'att1', 
                        ['Value', 1,
                            ['Leaf', 'no', 1, 4]], 
                        ['Value', 2, 
                            ['Leaf', 'no', 1, 4]], 
                        ['Value', 3, 
                            ['Leaf', 'no', 2, 4]]]], 
                ['Value', 'fair', 
                    ['Leaf', 'yes', 6, 10]
                ]
            ]
        ]
    ]
    dt = MyDecisionTreeClassifier()
    dt.fit(X_train_iphone, y_train_iphone)
    ## First unseen instance: [2, 2 "fair"]
    X = [[2, 2, "fair"]]
    y_expected = ["yes"]
    assert dt.predict(X) == y_expected
    ## Second unseen instance: [1, 1, "excellent"]     
    X = [[1, 1, "excellent"]]
    y_expected = ["yes"]
    assert dt.predict(X) == y_expected

def test_kneighbors_classifier_kneighbors():
    # example #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test1 = [[0,0]]

    # Desk Check: distances from [0,0]
    # [1,1]: sqrt(2) = pow(2, 0.5) = 1.414
    # [1,0]: 1
    # [0.33,0]: 0.33
    # [0,0]: 0
    expected_distances = [0.0, 0.33, 1]
    expected_neighbor_indices = [3, 2, 1]
    myKnn = MyKNeighborsClassifier()
    myKnn.fit(X_train_class_example1, y_train_class_example1)
    distances, neighbor_indices = myKnn.kneighbors(X_test1)
    assert all(np.isclose(distances[0], expected_distances))
    assert neighbor_indices[0] == expected_neighbor_indices

    # example #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test2 = [[2, 1]]
    # Desk Check: distances from [2, 1]
    # [3, 2]: sqrt(2) = pow(2, 0.5) = 1.414
    # [6, 6]: sqrt(41) = pow(41, 0.5) 
    # [4, 1]: 2
    # [4, 4]: sqrt(13) = pow(13, 0.5)
    # [1, 2]: sqrt(2) = pow(2, 0.5) = 1.414
    # [2, 0]: 1
    # [0, 3]: sqrt(8) = pow(8, 0.5) 
    # [1, 6]: sqrt(26)
    expected_distances = [1.0, pow(2, 0.5), pow(2, 0.5)]
    expected_neighbor_indices = [5,0,4]
    myKnn = MyKNeighborsClassifier()
    myKnn.fit(X_train_class_example2, y_train_class_example2)
    distances, neighbor_indices = myKnn.kneighbors(X_test2)
    assert all(np.isclose(distances[0], expected_distances))
    assert neighbor_indices[0] == expected_neighbor_indices

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    X_test3 = [[0, 0]]
    # Desk Check: distances from [0,0] (approx rounding 2 decimals)
    # [0.8, 6.3] = 6.35
    # [1.4, 8.1] = 8.22
    # [2.1, 7.4] = 7.69
    # [2.6, 14.3] = 14.53
    # [6.8, 12.6] = 14.32
    # [8.8, 9.8] = 13.17
    # [9.2, 11.6] = 14.81
    # [10.8, 9.6] = 14.45
    # [11.8, 9.9] = 15.40
    # [12.4, 6.5]= 14.00
    # [12.8, 1.1] = 12.85
    # [14.0, 19.9] = 24.33
    # [14.2, 18.5] = 23.32
    # [15.6, 17.4] = 23.37
    # [15.8, 12.2] = 19.96
    # [16.6, 6.7] = 17.90
    # [17.4, 4.5] = 17.97
    # [18.2, 6.9] = 19.46
    # [19.0, 3.4] = 19.30
    # [19.6, 11.1] = 22.52
    expected_distances = [6.35, 7.69, 8.22]
    expected_neighbor_indices = [0, 2, 1]
    myKnn = MyKNeighborsClassifier()
    myKnn.fit(X_train_bramer_example, y_train_bramer_example)
    distances, neighbor_indices = myKnn.kneighbors(X_test3)
    rounded_distances = [round(num, 2) for num in distances[0]]
    assert rounded_distances == expected_distances
    assert neighbor_indices[0] == expected_neighbor_indices # TODO: fix this

def test_kneighbors_classifier_predict():
   # example #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test1 = [[0,0]]
    # Desk Check: knn distances from [0,0]
    # [1,0]: 1 -> bad
    # [0.33,0]: 0.33 -> good
    # [0,0]: 0 -> good
    expected_class = ['good']
    myClf = MyKNeighborsClassifier()
    myClf.fit(X_train_class_example1, y_train_class_example1)
    y_pred = myClf.predict(X_test1)
    assert y_pred == expected_class

    # example #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test2 = [[2, 1]]
    # Desk Check: knn distances from [2, 1]
    # [3, 2]: sqrt(2) = pow(2, 0.5) = 1.414 -> no
    # [1, 2]: sqrt(2) = pow(2, 0.5) = 1.414 -> yes
    # [2, 0]: 1 -> no
    expected_class = ['no']
    myClf = MyKNeighborsClassifier()
    myClf.fit(X_train_class_example2, y_train_class_example2)
    y_pred = myClf.predict(X_test2)
    assert y_pred == expected_class 

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    X_test3 = [[0, 0]]
    # Desk Check: knn distances from [0,0] (approx rounding 2 decimals)
    # [0.8, 6.3] = 6.35 -> -
    # [1.4, 8.1] = 8.22 -> -
    # [2.1, 7.4] = 7.69 -> - 
    expected_class = ['-']
    myClf = MyKNeighborsClassifier()
    myClf.fit(X_train_bramer_example,y_train_bramer_example)
    y_pred = myClf.predict(X_test3)
    assert y_pred == expected_class # TODO: fix this

def test_random_forest_fit():
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y = ["A", "A", "B", "B"]

    n_trees = 5
    max_trees = 3
    clf = MyRandomForestClassifier(n_trees=n_trees,
                                   max_trees=max_trees,
                                   max_features=1,
                                   random_state=0)

    clf.fit(X, y)
    assert len(clf.forest_) == max_trees
    for tree, feat_idx in clf.forest_:
        assert hasattr(tree, "predict")
        assert isinstance(feat_idx, list)
        assert len(feat_idx) >= 1
        for j in feat_idx:
            assert 0 <= j < len(X[0])

def test_random_forest_predict(): 
    X = [
        [0, 0],  # A
        [0, 1],  # B
        [1, 0],  # B
        [1, 1],  # B
        [0, 0],  # A
        [0, 1],  # B
        [1, 0],  # B
        [1, 1]   # B
    ]
    y = ["A", "B", "B", "B", "A", "B", "B", "B"]

    X_train = X[:6]
    y_train = y[:6]
    X_test  = X[6:]
    y_test  = y[6:]

    clf = MyRandomForestClassifier(n_trees=10,
                                   max_trees=5,
                                   max_features=1,
                                   random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert len(y_pred) == len(y_test)
    acc = myevaluation.accuracy_score(y_test, y_pred)
    assert acc >= 0.5