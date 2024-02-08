# Data Analytics and Machine-Learning
This repository includes all the Data Analytics and Machine learning projects written in jupyter notebook using python language.
There are 46 jupyter notebook files which are mentioned as follows:
### 1. Feature_Scaling.ipynb:
   #### - Description: 
   * Using Pandas library creating dataframe from 'wine_data.csv' file.
   #### - Functions Used: 
   * pandas.read_csv
   * pandas.DataFrame
   * preprocessing.scale(wine)
   * min_max_scaler_object.fit_transform(wine)

### 2. CrossValidation.ipynb:
   #### - Description: 
   * Cross-validation is a method for assessing the performance of a statistical model in which the data is split into two or more subsets, or folds. In linear regression, cross-validation can be used to evaluate how well the model can generalize to new, unseen data.
   #### - Functions Used:
   * cross_val_score(clf, iris.data, iris.target, cv = KFold(3, True, 0 ))
   * iris = datasets.load_iris()
   * xtrain , xtest , ytrain , ytest = train_test_split(iris.data , iris.target , test_size = 0.2)
           
### 3. KNN_Scratch.ipynb:
   #### - Description:
   * KNN, or k-nearest neighbors, is a popular algorithm in data science used for classification and regression tasks. The KNN algorithm works by calculating the distance between the new data point and all other data points in the dataset. The distance is typically calculated using Euclidean distance, although other distance metrics can be used as well. The k nearest neighbors are then identified, where k is a pre-specified number.
   * The choice of the value of k is an important hyperparameter in the KNN algorithm. A small value of k will make the algorithm more sensitive to noise and outliers, while a large value of k will make the algorithm less sensitive to noise but may result in less accurate predictions.
   #### - Functions Used:
   * clf = KNeighborsClassifier(n_neighbors=7)
   * clf.fit(X_train, Y_train)
   * clf.score(X_test, Y_test)
   * accuracy_score(Y_test, y_pred)
     
### 4. SVMDummyData.ipynb:
   #### - Description:
   * Support Vector Machine (SVM) is a popular machine learning algorithm used for classification and regression tasks. It works by creating a hyperplane or a set of hyperplanes that separates the different classes of data. SVM is a powerful algorithm for classification tasks in high-dimensional spaces and is particularly useful when there are clear margins of separation between the different classes.
   #### - Functions Used:
   * svcLinear =  SVC(kernel='linear', C=100000).fit(X, y)
   * svcLinear.coef_
   * svcLinear.intercept_
   * plt.scatter(X_x1, X_x2, c = y)

### 5. SVM_Iris.ipynb:
   #### - Description:
   * Support Vector Machine (SVM) is a popular machine learning algorithm used for classification and regression tasks. It works by creating a hyperplane or a set of hyperplanes that separates the different classes of data. SVM is a powerful algorithm for classification tasks in high-dimensional spaces and is particularly useful when there are clear margins of separation between the different classes.
   #### - Functions Used:
   * xx, yy = makegrid(x[:, 0], x[:, 1])
   * predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
   * plt.scatter(xx, yy, c = predictions)
     
### 6. SVRBoston.ipynb:
   #### - Description:
   * In machine learning, kernel is a hyperparameter of the Support Vector Machine (SVM) algorithm that determines the type of kernel function used to transform the input data into a higher-dimensional space. kernel="rbf" is one of the options for the kernel parameter in the SVC class of scikit-learn, where "rbf" stands for Radial Basis Function.
   * The Radial Basis Function kernel is commonly used in SVM for classification tasks. It maps the input data into a higher-dimensional space where it is easier to find a hyperplane that can separate the classes. The RBF kernel is defined by a single parameter, gamma, which controls the width of the kernel and the influence of individual training examples. A higher value of gamma will lead to a tighter and more complex decision boundary that can potentially overfit the data, while a lower value will result in a smoother decision boundary.
   #### - Functions Used:
   * clf = svm.SVR(kernel="rbf")
   * clf.score(x_test, y_test)


### 7. GridSearchCV.ipynb:
   #### - Description:
   * The basic idea behind GridSearchCV is to define a grid of hyperparameters to evaluate, and then perform an exhaustive search over that grid to find the optimal hyperparameters. This is achieved by cross-validating the model with each set of hyperparameters to evaluate its performance, and then selecting the set of hyperparameters that gives the best performance.
   #### - Functions Used:
   * abc = GridSearchCV(clf, grid)
   * abc.fit(x_train, y_train)
   * abc.best_estimator_

### 8. KNN.ipynb:
   #### - Description:
   * KNN, or k-nearest neighbors, is a popular algorithm in data science used for classification and regression tasks. The KNN algorithm works by calculating the distance between the new data point and all other data points in the dataset. The distance is typically calculated using Euclidean distance, although other distance metrics can be used as well. The k nearest neighbors are then identified, where k is a pre-specified number.
   #### - Functions Used:
   * clf = KNeighborsClassifier()
   * clf.fit(X_train, Y_train)
   * clf.score(X_test, Y_test)

### 9. PCA_3D.ipynb:
   #### - Description:
   * Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform a high-dimensional dataset into a lower-dimensional one by identifying and capturing the most important patterns and relationships in the data.
   * The steps involved in performing PCA are as follows:
        <ol type="a">
           <li>Standardize the data by subtracting the mean and dividing by the standard deviation for each feature.</li>
           <li>Compute the covariance matrix of the standardized data.</li>
           <li>Compute the eigenvectors and eigenvalues of the covariance matrix.</li>
           <li>Sort the eigenvectors in descending order of their corresponding eigenvalues to obtain the principal components.</li>
           <li>Project the data onto the principal components to obtain the lower-dimensional representation of the data.</li>
        </ol>
   * PCA has a wide range of applications in various fields such as data science, image processing, and finance. It can be used for tasks such as data compression, data visualization, and feature extraction.
   #### - Functions Used:
   * pca = PCA(n_components = 2)
   * transformed_data = pca.fit_transform(all_data)
   * pca = PCA(n_components = 1)
   * X_reduced = pca.fit_transform(X)
   * X_approx = pca.inverse_transform(X_reduced)

### 10. PCA_2D.ipynb:
   #### - Description:
   * Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform a high-dimensional dataset into a lower-dimensional one by identifying and capturing the most important patterns and relationships in the data.
   * The steps involved in performing PCA are as follows:
        <ol type="a">
           <li>Standardize the data by subtracting the mean and dividing by the standard deviation for each feature.</li>
           <li>Compute the covariance matrix of the standardized data.</li>
           <li>Compute the eigenvectors and eigenvalues of the covariance matrix.</li>
           <li>Sort the eigenvectors in descending order of their corresponding eigenvalues to obtain the principal components.</li>
           <li>Project the data onto the principal components to obtain the lower-dimensional representation of the data.</li>
        </ol>
   * PCA has a wide range of applications in various fields such as data science, image processing, and finance. It can be used for tasks such as data compression, data visualization, and feature extraction.
   #### - Functions Used:
   * pca = PCA(n_components = 2)
   * transformed_data = pca.fit_transform(all_data)
   * pca = PCA(n_components = 1)
   * X_reduced = pca.fit_transform(X)
   * X_approx = pca.inverse_transform(X_reduced)

### 11. NaiveBayes.ipynb:
   #### - Description:
   * Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that the presence of a particular feature in a class is independent of the presence of any other feature in the same class. This assumption is known as the "naive" assumption and hence the name "Naive Bayes".
   * There are three main types of Naive Bayes algorithms:
        <ol type="a">
           <li>Gaussian Naive Bayes: assumes that the input features are normally distributed and calculates the conditional probability of each feature given each class using the Gaussian probability density function.</li>
           <li>Multinomial Naive Bayes: used for discrete data such as text data and assumes that the input features have a multinomial distribution.</li>
           <li>Bernoulli Naive Bayes: also used for discrete data but assumes that each input feature is binary (present or absent) and calculates the conditional probability using the Bernoulli distribution.</li>
        </ol>
   #### - Functions Used:
   * dictionary = fit(X_train,Y_train)
   * Y_pred = predict(dictionary,X_test)
   * classification_report(Y_test,Y_pred)
   * confusion_matrix(Y_test,Y_pred)

### 12. Movie_Reviews_SKLearn_via_NLTK.ipynb:
   #### - Description:
   * The Natural Language Toolkit (NLTK) is a popular open-source library for natural language processing (NLP) in Python. NLTK provides a wide range of tools and resources for working with human language data, including data pre-processing, text classification, tokenization, stemming, tagging, parsing, and semantic analysis.
   * Some of the key features of NLTK include:
        <ol type="a">
           <li>Tokenization: breaking up text into individual words or sentences.</li>
           <li>Stemming: reducing words to their root form.</li>
           <li>Part-of-speech (POS) tagging: labeling words with their grammatical categories.</li>
           <li>Named Entity Recognition (NER): identifying and classifying named entities such as people, organizations, and locations.</li>
           <li>Sentiment analysis: determining the sentiment or polarity of a text.</li>
           <li>Parsing: analyzing the grammatical structure of a sentence.</li>
           <li>Machine learning algorithms: NLTK includes various machine learning algorithms for text classification and clustering.</li>
        </ol>
   #### - Functions Used:
   * movie_reviews.words(movie_reviews.fileids()[5])
   * lemmatizer = WordNetLemmatizer()
   * stops = set(stopwords.words('english'))
   * stops.update(punctuations)
   * classfier = NaiveBayesClassifier.train(training_data)
   * nltk.classify.accuracy(classfier, testing_data)
   * classifier_sklearn = SklearnClassifier(svc)

### 13. Movie_Reviews_SKLearn_CountVectorizer.ipynb:
   #### - Description:
   * NLTK also provides access to various corpora and lexicons, including the Brown Corpus, Gutenberg Corpus, WordNet, and the Penn Treebank Corpus. These resources can be used for tasks such as training machine learning models, evaluating performance, and creating language models.
   #### - Functions Used:
   * train_set = {"the sky sky is blue", "the sun is bright"}
   * count_vec = CountVectorizer(max_features = 3)
   * a = count_vec.fit_transform(train_set)

### 14. NeuralNetworkImplementationWithoutHiddenLayer.ipynb:
   #### - Description:
   * A neural network without a hidden layer is also known as a single-layer neural network or a perceptron. It consists of only one layer of neurons, which directly connects the input to the output.
   * A neural network without a hidden layer is typically used for simple classification tasks or linear regression problems, where the relationship between the input and output variables can be expressed as a linear equation. It can be trained using supervised learning algorithms such as gradient descent or backpropagation.
   #### - Functions Used:
   * numpy.random.random((2, 1))
   * numpy.dot(output0, weights)

