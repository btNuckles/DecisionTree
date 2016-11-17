# decision trees: scikit-learn + pandas

This script provides an example of learning a decision tree with
scikit-learn.  Pandas is used to read data and custom functions are employed
to investigate the decision tree after it is learned.  Grab the code and try
it out.

A blog post about this code is available
[here](http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html),
check it out!

## Requirements

* python -- developed with 2.7.6
* sckit-learn -- using version 0.16.1
* pandas -- using version 0.16.1
* numpy -- using version 1.9.2

and to create the graphic of the tree you must have graphviz/dot installed.

## Usage

### 1. Run script from command line

This provides an example of using the available functions-- look at lines
122-143 to see what is done.

```bash
$ python analyze_dt.py
```

This:

* Fetches the data using pandas, or grabs the local copy.
* Outputs the *head* of the pandas data frame.
* Fits the decision tree and outputs the *pseudo code* for the decision tree.
* Uses pandas to show that the first branch at *PetalLength <= 2.45* is easily
  verified.

The resulting output is:

```
-- get data:
-- iris.csv found locally

-- df.head():
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa


-- get_code:
if ( PetalLength <= 2.45000004768 ) {
    return Iris-setosa ( 50 examples )
}
else {
    if ( PetalWidth <= 1.75 ) {
        if ( PetalLength <= 4.94999980927 ) {
            if ( PetalWidth <= 1.65000009537 ) {
                return Iris-versicolor ( 47 examples )
            }
            else {
                return Iris-virginica ( 1 examples )
            }
        }
        else {
            return Iris-versicolor ( 2 examples )
            return Iris-virginica ( 4 examples )
        }
    }
    else {
        if ( PetalLength <= 4.85000038147 ) {
            return Iris-versicolor ( 1 examples )
            return Iris-virginica ( 2 examples )
        }
        else {
            return Iris-virginica ( 43 examples )
        }
    }
}

-- look back at original data using pandas
-- df[df['PetalLength'] <= 2.45]]['Name'].unique():  ['Iris-setosa']
```

### 2. Use interactively with (i)python

This code can also be used interactively by importing the available functions.
I do this by importing `analyze_dt as adt` and using a *function* like so
`adt.function()`. Follow along:

```python
>>> import analyze_dt as adt
>>> df = adt.get_iris_data()
-- iris.csv found locally
>>> df.head()
   SepalLength  SepalWidth  PetalLength  PetalWidth         Name
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa
>>> df.columns
Index([u'SepalLength', u'SepalWidth', u'PetalLength', u'PetalWidth', u'Name'], dtype='object')
>>> features = list(df.columns[:4])
>>> features
['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
>>> df, targets = adt.encode_target(df, "Name")
>>> y = df["Target"]
>>> X = df[features]
>>> dt = adt.DecisionTreeClassifier(min_samples_split=20, random_state=99)
>>> dt.fit(X,y)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            random_state=99, splitter='best')
>>> adt.get_code(dt, features, targets)
if ( PetalLength <= 2.45000004768 ) {
    return Iris-setosa ( 50 examples )
}
else {
    if ( PetalWidth <= 1.75 ) {
        if ( PetalLength <= 4.94999980927 ) {
            if ( PetalWidth <= 1.65000009537 ) {
                return Iris-versicolor ( 47 examples )
            }
            else {
                return Iris-virginica ( 1 examples )
            }
        }
        else {
            return Iris-versicolor ( 2 examples )
            return Iris-virginica ( 4 examples )
        }
    }
    else {
        if ( PetalLength <= 4.85000038147 ) {
            return Iris-versicolor ( 1 examples )
            return Iris-virginica ( 2 examples )
        }
        else {
            return Iris-virginica ( 43 examples )
        }
    }
}
>>> df[df['PetalLength'] <= 2.45]['Name'].unique()
array(['Iris-setosa'], dtype=object)
>>> adt.visualize_tree(dt, features)
```
