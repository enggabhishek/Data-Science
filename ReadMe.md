# Machine-Learning
This repository includes all the machine learning projects written in jupyter notebook using python language.
There are 46 jupyter notebook files which are mentioned as follows:
1. Feature_Scaling.ipynb:
    - Description: Using Pandas library creating dataframe from 'wine_data.csv' file.
    - Functions Used: pandas.read_csv, pandas.DataFrame, preprocessing.scale(wine), min_max_scaler_object.fit_transform(wine)

2. CrossValidation.ipynb:
    - Description: Cross-validation is a method for assessing the performance of a statistical model in which the data is split into two or more subsets, or folds. In linear regression, cross-validation can be used to evaluate how well the model can generalize to new, unseen data.
    - Functions Used:
        1. cross_val_score(clf, iris.data, iris.target, cv = KFold(3, True, 0 ))
        2. iris = datasets.load_iris()
        3. xtrain , xtest , ytrain , ytest = train_test_split(iris.data , iris.target , test_size = 0.2)
           
3. CrossValidation.ipynb:
    - Description: Cross-validation is a method for assessing the performance of a statistical model in which the data is split into two or more subsets, or folds. In linear regression, cross-validation can be used to evaluate how well the model can generalize to new, unseen data.
    - Functions Used:
        * cross_val_score(clf, iris.data, iris.target, cv = KFold(3, True, 0 ))
        * iris = datasets.load_iris()
        * xtrain , xtest , ytrain , ytest = train_test_split(iris.data , iris.target , test_size = 0.2)

