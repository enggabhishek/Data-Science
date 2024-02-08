# Machine-Learning
This repository includes all the machine learning projects written in jupyter notebook using python language.
There are 46 jupyter notebook files which are mentioned as follows:
### 1. Feature_Scaling.ipynb:
    - Description: Using Pandas library creating dataframe from 'wine_data.csv' file.
    - Functions Used: pandas.read_csv, pandas.DataFrame, preprocessing.scale(wine), min_max_scaler_object.fit_transform(wine)

#### 2. CrossValidation.ipynb:
    - Description: Cross-validation is a method for assessing the performance of a statistical model in which the data is split into two or more subsets, or folds. In linear regression, cross-validation can be used to evaluate how well the model can generalize to new, unseen data.
    - Functions Used:
        * cross_val_score(clf, iris.data, iris.target, cv = KFold(3, True, 0 ))
        * iris = datasets.load_iris()
        * xtrain , xtest , ytrain , ytest = train_test_split(iris.data , iris.target , test_size = 0.2)
           
### 3. KNN_Scratch.ipynb:
    - Description:
      * KNN, or k-nearest neighbors, is a popular algorithm in data science used for classification and regression tasks. The KNN algorithm works by calculating the distance between the new data point and all other data points in the dataset. The distance is typically calculated using Euclidean distance, although other distance metrics can be used as well. The k nearest neighbors are then identified, where k is a pre-specified number.
      * The choice of the value of k is an important hyperparameter in the KNN algorithm. A small value of k will make the algorithm more sensitive to noise and outliers, while a large value of k will make the algorithm less sensitive to noise but may result in less accurate predictions.
    - Functions Used:
        * clf = KNeighborsClassifier(n_neighbors=7)
        * clf.fit(X_train, Y_train)
        * clf.score(X_test, Y_test)
        * accuracy_score(Y_test, y_pred)

