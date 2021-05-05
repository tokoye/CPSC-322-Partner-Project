"""
Charles Walker 
CPSC 322
Section 02
PA6
"""
import numpy as np
import scipy.stats as stats 

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import mysklearn.myutils as myutils

# note: order is actual/received student value, expected/solution

def test_decision_tree_classifier_fit():
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    tree = MyDecisionTreeClassifier()
    tree.fit(X_train, y_train)
    interview_tree = \
    ["Attribute", "attr0", 
        ["Value", "Senior", 
            ["Attribute", "attr2",
                ["Value", "no", 
                    ["Leaf", "False", 3, 5]
                ],
                ["Value", "yes", 
                    ["Leaf", "True", 2, 5]
                ]
            ]
        ],
        ["Value", "Mid", 
            ["Leaf", "True", 4, 14]
        ],
        ["Value", "Junior", 
            ["Attribute", "attr3", 
                ["Value", "no", 
                    ["Leaf", "True", 3, 5]
                ],
                ["Value", "yes", 
                    ["Leaf", "False", 2, 5]
                ]
            ]
        ]
    ]
    assert tree.tree == interview_tree
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

    degrees_result = ['Attribute', 'attr2', 
                        ['Value', 'A', ['Attribute', 'attr3', 
                            ['Value', 'B', ['Leaf', 'B', 7, 9]], 
                        ['Value', 'A', ['Attribute', 'attr0', 
                            ['Value', 'A', ['Leaf', 'A', 1, 2]], 
                        ['Value', 'B', ['Leaf', 'B', 1, 2]]]]]],
                        ['Value', 'B', ['Attribute', 'attr1', 
                            ['Value', 'B', ['Leaf', 'A', 11, 17]], 
                            ['Value', 'A', ['Leaf', 'B',6, 17]]]]
                    ]

    X_train = []
    y_train = []
    for row in degrees_table:
        X_train.append(row[0:4])
        y_train.append(row[4])

    tree1 = MyDecisionTreeClassifier()
    tree1.fit(X_train, y_train)
    
    assert tree1.tree == degrees_result
def test_decision_tree_classifier_predict():
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    tree = MyDecisionTreeClassifier()
    tree.fit(X_train, y_train)
    X_test = [["Senior", "Java", "no", "no"], ["Senior", "Java", "no", "yes"], ["Senior", "Java", "yes", "no"]]
    pred = tree.predict(X_test)
    assert  pred == ["False", "False", "True"] # TODO: fix this

    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

    X_train = []
    y_train = []
    for row in degrees_table:
        X_train.append(row[0:4])
        y_train.append(row[4])

    tree1 = MyDecisionTreeClassifier()
    tree1.fit(X_train, y_train)

    test_vals = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]

    assert tree1.predict(test_vals) == ['A', 'A', 'A']