# Data Analytics
This repository includes all the Data Analytics related projects written in jupyter notebook using python language.
There are 43 jupyter notebook files which are mentioned as follows:
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

### 15. ForwardPropagationWithoutHiddenLayer.ipynb:
   #### - Description:
   * Forward propagation without hidden layer is a process in which a neural network with only one input layer and one output layer calculates the output for a given input. It is a simple process that involves multiplying the input by the weights and adding a bias term, and then passing this value through an activation function to get the output.
   * The process of forward propagation can be summarized in the following steps:
        <ol type="a">
           <li>Initialize the weights and bias: Each input is associated with a weight and a bias term, which are initially set to random values.</li>
           <li>Multiply input by weights: Multiply the input value with the corresponding weight associated with that input.</li>
           <li>Add bias term: Add the bias term to the weighted sum obtained in step 2.</li>
           <li>Apply activation function: Pass the value obtained in step 3 through an activation function to obtain the output value.</li>
           <li>Output: The output value obtained in step 4 is the result of the forward propagation. Repeat for all inputs: Repeat steps 2-5 for all the inputs in the input layer to get the output for each input..</li>
        </ol>
   #### - Functions Used:
   * numpy.random.random((2, 1))
   * numpy.dot(output0, weights)
### 16. ForwardPropagation_One Hidden layer.ipynb:
   #### - Description:
   * Forward propagation without hidden layer is a process in which a neural network with only one input layer and one output layer calculates the output for a given input. It is a simple process that involves multiplying the input by the weights and adding a bias term, and then passing this value through an activation function to get the output.
   * The process of forward propagation can be summarized in the following steps:
        <ol type="a">
           <li>Initialize the weights and bias: Each input is associated with a weight and a bias term, which are initially set to random values.</li>
           <li>Multiply input by weights: Multiply the input value with the corresponding weight associated with that input.</li>
           <li>Add bias term: Add the bias term to the weighted sum obtained in step 2.</li>
           <li>Apply activation function: Pass the value obtained in step 3 through an activation function to obtain the output value.</li>
           <li>Output: The output value obtained in step 4 is the result of the forward propagation. Repeat for all inputs: Repeat steps 2-5 for all the inputs in the input layer to get the output for each input..</li>
        </ol>
   #### - Functions Used:
   * numpy.random.random((2, 1))
   * numpy.dot(output0, weights)

### 17. MNIST_Tensorflow_Predictions.ipynb:
   #### - Description:
   * The MNIST dataset is a popular dataset in machine learning and computer vision. It consists of 70,000 images of handwritten digits (0-9) that are 28x28 pixels in size.
   #### - Functions Used:
   * input_data.read_data_sets("MNIST_data/", one_hot=True)
   * in_layer1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
   * out_layer1 = tf.nn.relu(in_layer1)
   * x = tf.placeholder("float", [None, n_input])
   * y =tf.placeholder(tf.int32, [None, n_classes])
   * cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
   * optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
   * optimize = optimizer.minimize(cost)
   * predictions = tf.argmax(pred, 1)
   * correct_labels = tf.argmax(y, 1)
### 18. MNIST_Tensorflow_Optimizer.ipynb:
   #### - Description:
   * The MNIST dataset is a popular dataset in machine learning and computer vision. It consists of 70,000 images of handwritten digits (0-9) that are 28x28 pixels in size.
   #### - Functions Used:
   * input_data.read_data_sets("MNIST_data/", one_hot=True)
   * in_layer1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
   * out_layer1 = tf.nn.relu(in_layer1)
   * x = tf.placeholder("float", [None, n_input])
   * y =tf.placeholder(tf.int32, [None, n_classes])
   * cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
   * optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
   * optimize = optimizer.minimize(cost)
   * predictions = tf.argmax(pred, 1)
   * correct_labels = tf.argmax(y, 1)

### 19. CNN_MNIST.ipynb:
   #### - Description:
   * To train a CNN on the MNIST dataset, we typically use a network architecture that consists of several convolutional layers, followed by a few fully connected layers. The convolutional layers are responsible for extracting features from the images, while the fully connected layers are responsible for performing the final classification.
   #### - Functions Used:
   * tf.Variable(tf.random_normal([conv1_k, conv1_k, input_channels, n_conv1]))
   * tf.nn.relu(hidden_output_before_activation)
   * tf.nn.dropout(hidden_output_before_dropout, keep_prob)
     
### 20. RNN_Airlines.ipynb:
   #### - Description:
   * To use an RNN for predicting airline passenger traffic, we can first preprocess the data by normalizing it and splitting it into training, validation, and test sets. We can then define the RNN architecture, which typically consists of several layers of recurrent units, such as LSTM or GRU units. These units are designed to capture the temporal dependencies in the data and can "remember" information from previous time steps.
   #### - Functions Used:
   * scaler = MinMaxScaler(feature_range =(0,1))
   * train = scaler.fit_transform(train)
   * test = scaler.transform(test)
   * combinedPredicted = np.concatenate((trainPredict, testPredict))
   * combinedTrue = np.concatenate((trainTrue, testTrue))

### 21. KerasIntro.ipynb:
   #### - Description:
   * Keras is an open-source deep learning framework written in Python. It was developed with a focus on enabling fast experimentation with deep neural networks, while also being user-friendly and easy to understand. Keras provides a high-level interface for building and training neural networks, and it can run on top of several low-level deep learning frameworks, such as TensorFlow, Theano, and Microsoft Cognitive Toolkit.
   #### - Functions Used:
   * model = Sequential()
   * layer1 = Dense(units=32, activation = 'relu', input_dim = 30)
   * model.add(layer1)
   * sc = StandardScaler()
   * x_train = sc.fit_transform(x_train)
   * x_test = sc.transform(x_test)
   * score = model.evaluate(x_test, y_test)

### 22. APICall.ipynb:
   #### - Description:
   * An API call in Python is a request made to a web API using the Python programming language. This request can be used to retrieve data, post data, or perform other actions on a remote server.
   #### - Functions Used:
   * res=req.get('https://dog.ceo/api/breeds/list/all')
   * json_data=res.text
   * python_data=json.loads(json_data)
   * response = requests.get("https://api.github.com/repos/google/clusterfuzz"
   * auth = (user_name, password)
   * headers  = {"Accept": "application/vnd.github.mercy-preview+json"})
   * r_python = response.json()
     
### 23. BeautifulSoup.ipynb:
   #### - Description:
   * BeautifulSoup is a Python library used for web scraping and parsing HTML and XML documents. It allows you to extract data from HTML and XML files and to manipulate the parsed data in Python.
   #### - Functions Used:
   * data=BeautifulSoup(html,'html.parser')
   * next_page=data.find(class_='next')

### 24. Catplot.ipynb:
   #### - Description:
   * Creating categorical based graphs..
   #### - Functions Used:
   * seaborn.catplot(x = 'sex', y = 'total_bill', data = df)
     
### 25. Cifar10.ipynb:
   #### - Description:
   * CIFAR-10 is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is commonly used for computer vision tasks such as object recognition and image classification. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
   * The CIFAR-10 dataset is split into a training set of 50,000 images and a test set of 10,000 images. The training set is further divided into five batches of 10,000 images each. The test set contains exactly 1,000 randomly-selected images from each class.
   #### - Functions Used:
   * fig=plt.figure(figsize=(16, 16))
   * pca=PCA(whiten=True)
   * clf1=RandomForestClassifier(n_estimators=429, n_jobs=-1, max_depth=2000, max_leaf_nodes=2350)
   * clf2=LogisticRegression(n_jobs=-1, multi_class="auto")

### 26. ClassificationMetrics.ipynb:
   #### - Description:
   * Showing classification report:
     <ol type="a">
           <li>Precision</li>
           <li>Recall</li>
           <li>F1-score</li>
           <li>Support.</li>
        </ol>
   #### - Functions Used:
   * classification_report(y_test, y_test_pred)

### 27. ClassificationMetrics.ipynb:
   #### - Description:
   * Showing classification report:
     <ol type="a">
           <li>Precision</li>
           <li>Recall</li>
           <li>F1-score</li>
           <li>Support.</li>
        </ol>
   #### - Functions Used:
   * classification_report(y_test, y_test_pred)

### 28. GraphAssignment.ipynb:
   #### - Description:
   * Showing graphs based on different amazon job sales.
   #### - Functions Used:
   * classification_report(y_test, y_test_pred)
### 29. IMDB.ipynb:
   #### - Description:
   * SQLite3 is a lightweight, serverless, embedded relational database management system (RDBMS) that is included as part of the Python Standard Library. It allows you to create, access, and modify databases using SQL (Structured Query Language), a standard language for managing RDBMS.
   * SQLite3 is widely used in embedded systems, mobile applications, and desktop software because it is fast, efficient, and easy to use. It is a self-contained, zero-configuration database that does not require a separate server process or installation.
   #### - Functions Used:
   * db=sqlite3.connect('IMDB.sqlite')
   * cur=db.cursor()
   * data=pd.read_sql_query("Select * from earning",db)
   * cur.execute('Select rating from IMDB where Movie_id=?',(movie_id,))
   * rows=cur.fetchone()

### 30. Instabot_1.ipynb:
   #### - Description:
   * Web scraping refers to the process of extracting data from websites using automated scripts or programs. Python is a popular language for web scraping due to its rich ecosystem of libraries and tools. One of the most commonly used tools for web scraping in Python is Selenium.
   #### - Functions Used:
   * driver = webdriver.Chrome('C:/Users/Aman Bhalla/Downloads/chromedriver_win32/chromedriver')
   * driver.get('https://www.instagram.com/')
   * driver.implicitly_wait(10)
   * user = driver.find_element_by_name('username')
   * user.send_keys('SAMPLE USERNAME')
   * login = driver.find_element_by_xpath('//div[starts-with(@class,Igw0E)]/button')
   * lis = driver.find_elements_by_class_name('Ap253')

### 31. Instabot_2.ipynb:
   #### - Description:
   * Web scraping refers to the process of extracting data from websites using automated scripts or programs. Python is a popular language for web scraping due to its rich ecosystem of libraries and tools. One of the most commonly used tools for web scraping in Python is Selenium.
   #### - Functions Used:
   * driver = webdriver.Chrome('C:/Users/Aman Bhalla/Downloads/chromedriver_win32/chromedriver')
   * driver.get('https://www.instagram.com/')
   * driver.implicitly_wait(10)
   * user = driver.find_element_by_name('username')
   * user.send_keys('SAMPLE USERNAME')
   * login = driver.find_element_by_xpath('//div[starts-with(@class,Igw0E)]/button')
   * lis = driver.find_elements_by_class_name('Ap253')

### 32. Scatter_masked.ipynb:
   #### - Description:
   * A scatter plot with masked points can be created using the scatter function in combination with a mask array. A mask is a boolean array that indicates which data points should be displayed and which should be masked (hidden).
   #### - Functions Used:
   * area1 = numpy.ma.masked_where(r < r0, area)
   * area2 = numpy.ma.masked_where(r >= r0, area)
   * pyplot.scatter(x, y, s=area1, marker='^', c=c)
   * pyplot.scatter(x, y, s=area2, marker='o', c=c)
   * theta = np.arange(0, np.pi / 2, 0.01)
   * pyplot.plot(r0 * np.cos(theta), r0 * np.sin(theta))

### 33. RedditAPI.ipynb:
   #### - Description:
   * Getting data from Reddit account using API call.
   #### - Functions Used:
   * res=requests.get('https://www.reddit.com/api/v1/authorize',params=data)

### 34. Gradient_Descent_Boston_Dataset_Manual.ipynb:
   #### - Description:
   * Created three functions:
     <ol type="a">
           <li>step_gradient: Finding the slope and intercept of the given dataset.</li>
           <li> gd: Finding the optimal value of slope and intercept based on the training dataset using learning rate and number of iterations.</li>
           <li>run: This function is used to find the predictions based on slope and intercept retrieved from training dataset using step_gradient and gd function.</li>
        </ol>
   #### - Functions Used:
   * run()
   * gd(points, learning_rate, num_iterations)
   * step_gradient(points, learning_rate, m, c)

### 35. Gradient_Descent_Multiple_Variables_Manual.ipynb:
   #### - Description:
   * Created three functions:
     <ol type="a">
           <li>step_gradient: Finding the slope and intercept of the given dataset.</li>
           <li> gd: Finding the optimal value of slope and intercept based on the training dataset using learning rate and number of iterations.</li>
           <li>run: This function is used to find the predictions based on slope and intercept retrieved from training dataset using step_gradient and gd function.</li>
        </ol>
   #### - Functions Used:
   * run()
   * gd(points, learning_rate, num_iterations)
   * step_gradient(points, learning_rate, m, c)

### 36. LogisticRegression_Titanic.ipynb:
   #### - Description:
   * The Titanic dataset is a classic dataset used in data science and machine learning for classification tasks. The objective is to predict whether a passenger on the Titanic survived or not based on a set of features like age, sex, ticket class, etc.
   * Logistic regression is a commonly used algorithm for binary classification tasks like this. In logistic regression, we use a logistic function (also known as the sigmoid function) to transform the linear regression output into a probability value between 0 and 1. We then use a threshold value to classify the output as either 0 or 1.
     <ol type="a">
           <li>step_gradient: Finding the slope and intercept of the given dataset.</li>
           <li> gd: Finding the optimal value of slope and intercept based on the training dataset using learning rate and number of iterations.</li>
           <li>run: This function is used to find the predictions based on slope and intercept retrieved from training dataset using step_gradient and gd function.</li>
        </ol>
   #### - Functions Used:
   * data["Age"].apply(Age)
   * model = LR()
   * model.fit(x_train, y_train)
   * model.score(x_train, y_train)
   * model.predict_proba(x_test_array)
     
### 37. Decision_Tree_implementation.ipynb:
   #### - Description:
   * A decision tree is a popular machine learning algorithm used for both regression and classification tasks. It is a tree-like model where each internal node represents a test on an attribute or feature, each branch represents an outcome of the test, and each leaf node represents a class label or a numerical value. The topmost node of the tree is called the root node, and the nodes with no children are called leaf nodes.
   * The decision tree algorithm involves the following steps:
     <ol type="a">
           <li>Select the best attribute or feature to split the data based on the criterion of maximum information gain or minimum impurity.</li>
           <li>Select the best attribute or feature to split the data based on the criterion of maximum information gain or minimum impurity.</li>
           <li>Recursively apply the above steps to each subset until all the data is classified.</li>
           <li> If the goal is regression, the leaf node will represent a numerical value that is the predicted output. If the goal is classification, the leaf node will represent a class label.</li>
        </ol>
   #### - Functions Used:
   * datasets.load_iris()
   * toLabel(df, old_feature_name)
   * clf = DecisionTreeClassifier()
   * clf.fit(x_train, y_train)
   * dot_data = export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)
   * graph = pydotplus.graph_from_dot_data(dot_data)
   * graph.write_pdf("iris.pdf")

### 38. Decision_Tree_implementation.ipynb:
   #### - Description:
   * XGBoost (eXtreme Gradient Boosting) is a popular and powerful open-source gradient boosting library used for machine learning tasks.
   * The XGBoost algorithm involves the following steps:
     <ol type="a">
           <li> Initialize the model with a constant value, typically the mean or median of the target variable.</li>
           <li>SFit a decision tree to the data, using the negative gradient of the loss function as the target variable.</li>
           <li>Add the resulting tree to the model, adjusting the predictions of each instance in the training set by a learning rate.</li>
           <li> Repeat steps 2-3 for a fixed number of iterations or until a stopping criterion is met.</li>
           <li> Predict the output of a new instance by averaging the predictions of all the trees in the model.</li>
        </ol>
   #### - Functions Used:
   * GB = xgboost.XGBRegressor(objective="reg:linear", n_estimators=265, seed=123,max_depth=4)
   * GB.fit(x,y)
   * predictions=GB.predict(test)

### 39. TerroristRelatedPandas.ipynb:
   #### - Description:
   * Reading datasets and storing values in dataframe in Pandas.
   #### - Functions Used:
   * pandas.read_csv()

### 40. Diabetes_Datasets.ipynb:
   #### - Description:
   * Reading datasets and storing values in dataframe in Pandas.
   #### - Functions Used:
   * pandas.read_csv()

### 41. California_Datasets.ipynb:
   #### - Description:
   * Reading datasets and storing values in dataframe in Pandas.
   #### - Functions Used:
   * pandas.read_csv()

### 42. LinearRegression.ipynb:
   #### - Description:
   * Linear regression is a statistical method used to analyze the relationship between a dependent variable (also called the response variable) and one or more independent variables (also called the predictor variables). The goal of linear regression is to find the best-fit line that describes the relationship between the variables. The best-fit line is determined by minimizing the sum of the squared differences between the actual and predicted values of the dependent variable.
   #### - Functions Used:
   * model = GradientBoostingRegressor()
   * model.fit(x_train, y_train)
### 43. LinearRegressionWithManualMethods.ipynb:
   #### - Description:
   * In simple linear regression, there is only one independent variable, and the relationship between the independent and dependent variables can be described by a straight line. In multiple linear regression, there are multiple independent variables, and the relationship between the independent and dependent variables can be described by a plane or hyperplane.
   #### - Functions Used:
   * m,c = fit(x_train, y_train)
   * y_pred = predict(x_test, m, c)
   * score (y_test, y_pred)
