#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Christopher C. Strelioff <chris.strelioff@gmail.com>
#
# Distributed under terms of the MIT license.

"""analyze_dt.py -- probe a decision tree found with scikit-learn."""
from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def get_code(tree, feature_names, target_names, spacer_base="    "):
    """Produce psuedo-code for decision tree.
    
    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
   
    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse (left, right, threshold, features, left[node],
                            depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse (left, right, threshold, features, right[node],
                             depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + " ( " + \
                      str(target_count) + " examples )")
    
    recurse(left, right, threshold, features, 0, 0)


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.
    
    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names)
    
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing new Target column.

    Returns
    -------
    df -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


def get_iris_data():
    """Get the iris data, from local csv or pandas repo."""
    if os.path.exists("Cars.csv"):
        print("-- Cars.csv found locally")
        df = pd.read_csv("Cars.csv")
    else:
        print("-- trying to download from github")
        fn = "https://raw.githubusercontent.com/pydata/pandas/master/pandas/tests/data/iris.csv"
        try:
            df = pd.read_csv(fn)
        except:
            exit("-- Unable to download Cars.csv")

        with open("Cars.csv", 'w') as f:
            print("-- writing to local Cars.csv file")
            df.to_csv(f)

    return df

if __name__ == '__main__':
    print("\n-- get data:")
    df = get_iris_data()

    print("\n-- df.head():")
    print(df.head(), end="\n\n")
    print(df.tail(), end="\n\n")
    print("* Rating types:", df["Rating"].unique(), sep="\n")
    features = ["CarPrice", "MaintCost", "NumDoors", "NumPersons", "TrunkSize", "Safety"]
    df, targets = encode_target(df, "CarPrice")
    df, targets = encode_target(df, "MaintCost")
    df, targets = encode_target(df, "NumDoors")
    df, targets = encode_target(df, "NumPersons")
    df, targets = encode_target(df, "TrunkSize")
    df, targets = encode_target(df, "Safety")
    df, targets = encode_target(df, "Rating")
    print(df.head(), end="\n\n")
    print(df.tail(), end="\n\n")
    y = df["Rating"]
    X = df[features]

    dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(X, y)

    print("\n-- get_code:")
    get_code(dt, features, targets)

    #print("\n-- look back at original data using pandas")
    #print("-- df[df['Safety'] <= 2.45]]['Rating'].unique(): ",
    #      df[df['Safety'] <= 2.45]['Rating'].unique(), end="\n\n")

    visualize_tree(dt, features)
