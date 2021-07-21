---
title: "Question Set 3"
date: 2021-07-13T19:40:39+05:30
draft: false
---

**1. What are the types of Machine Learning?**
  
In all the ML Interview Questions that we would be going to discuss, this is one of the most basic question.
So, basically, there are three types of Machine Learning techniques:
Supervised Learning: In this type of the Machine Learning technique, machines learn under the supervision of labeled data. There is a training dataset on which the machine is trained, and it gives the output according to its training.
Unsupervised Learning: Unlike supervised learning, it has unlabeled data. So, there is no supervision under which it works on the data. Basically, unsupervised learning tries to identify patterns in data and make clusters of similar entities. After that, when a new input data is fed into the model, it does not identify the entity; rather, it puts the entity in a cluster of similar objects.
Reinforcement Learning: Reinforcement learning includes models that learn and traverse to find the best possible move. The algorithms for reinforcement learning are constructed in a way that they try to find the best possible suite of action on the basis of the reward and punishment theory.

**2. Differentiate between classification and regression in Machine Learning.**

In Machine Learning, there are various types of prediction problems based on supervised and unsupervised learning. These are classification, regression, clustering, and association. Here, we will discuss about classification and regression.
Classification: In classification, we try to create a Machine Learning model that assists us in differentiating data into separate categories. The data is labeled and categorized based on the input parameters.
For example, imagine that we want to make predictions on the churning out customers for a particular product based on some data recorded. Either the customers will churn out or they will not. So, the labels for this would be ‘Yes’ and ‘No.’
Regression: It is the process of creating a model for distinguishing data into continuous real values, instead of using classes or discrete values. It can also identify the distribution movement depending on the historical data. It is used for predicting the occurrence of an event depending on the degree of association of variables.
For example, the prediction of weather condition depends on factors such as temperature, air pressure, solar radiation, elevation of the area, and distance from sea. The relation between these factors assists us in predicting the weather condition.

**3. What is Linear Regression?**

Linear Regression is a supervised Machine Learning algorithm. It is used to find the linear relationship between the dependent and the independent variables for predictive analysis.
The equation for Linear Regression:
Y = A + BX
where:
X is the input or the independent variable
Y is the output or the dependent variable
a is the intercept and b is the coefficient of X
Below is the best fit line that shows the data of weight (Y or the dependent variable) and height (X or the independent variable) of 21-years-old candidates scattered over the plot. This straight line shows the best linear relationship that would help in predicting the weight of candidates according to their height.

![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/What-is-Linear-Regression-2-1.png "Subtitle")


To get this best fit line, we will try to find the best values of a and b. By adjusting the values of a and b, we will try to reduce errors in the prediction of Y.

This is how linear regression helps in finding the linear relationship and predicting the output.

**4. How will you determine the Machine Learning algorithm that is suitable for your problem?**

To identify the Machine Learning algorithm for our problem, we should follow the below steps:

Step 1: Problem Classification: Classification of the problem depends on the classification of input and output:

Classifying the input: Classification of the input depends on whether we have data labeled (supervised learning) or unlabeled (unsupervised learning), or whether we have to create a model that interacts with the environment and improves itself (reinforcement learning).

Classifying the output: If we want the output of our model as a class, then we need to use some classification techniques.

If it is giving the output as a number, then we must use regression techniques and, if the output is a different cluster of inputs, then we should use clustering techniques.

Step 2: Checking the algorithms in hand: After classifying the problem, we have to look for the available algorithms that can be deployed for solving the classified problem.

Step 3: Implementing the algorithms: If there are multiple algorithms available, then we will implement each one of them, one by one. Finally, we would select the algorithm that gives the best performance.

**5. What are Bias and Variance?**

Bias is the difference between the average prediction of our model and the correct value. If the bias value is high, then the prediction of the model is not accurate. Hence, the bias value should be as low as possible to make the desired predictions.

Variance is the number that gives the difference of prediction over a training set and the anticipated value of other training sets. High variance may lead to large fluctuation in the output. Therefore, the model’s output should have low variance.

The below diagram shows the bias–variance trade off:

![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/What-are-Bias-and-Variance.png "Subtitle")


Here, the desired result is the blue circle at the center. If we get off from the blue section, then the prediction goes wrong.

**6. What is Variance Inflation Factor?**

Variance Inflation Factor (VIF) is the estimate of the volume of multicollinearity in a collection of many regression variables.

VIF = Variance of the model / Variance of the model with a single independent variable

We have to calculate this ratio for every independent variable. If VIF is high, then it shows the high collinearity of the independent variables.

**7. Explain false negative, false positive, true negative, and true positive with a simple example.**

True Positive (TP): When the Machine Learning model correctly predicts the condition, it is said to have a True Positive value.

True Negative (TN): When the Machine Learning model correctly predicts the negative condition or class, then it is said to have a True Negative value.

False Positive (FP): When the Machine Learning model incorrectly predicts a negative class or condition, then it is said to have a False Positive value.

False Negative (FN): When the Machine Learning model incorrectly predicts a positive class or condition, then it is said to have a False Negative value.

**8. What is a Confusion Matrix?**

Confusion matrix is used to explain a model’s performance and gives the summary of predictions on the classification problems. It assists in identifying the uncertainty between classes.

A confusion matrix gives the count of correct and incorrect values and also the error types.Accuracy of the model: 


![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/Accuracy-of-the-model.png "Subtitle")


For example, consider this confusion matrix. It consists of values as True Positive, True Negative, False Positive, and False Negative for a classification model. Now, the accuracy of the model can be calculated as follows:

![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/What-is-a-Confusion-Matrix.png "Subtitle")


Thus, in our example:
Accuracy = (200 + 50) / (200 + 50 + 10 + 60) = 0.78
This means that the model’s accuracy is 0.78, corresponding to its True Positive, True Negative, False Positive, and False Negative values.

**9. What do you understand by Type I and Type II errors?**

Type I Error: Type I error (False Positive) is an error where the outcome of a test shows the non-acceptance of a true condition.
For example, a cricket match is going on and, when a batsman is not out, the umpire declares that he is out. This is a false positive condition. Here, the test does not accept the true condition that the batsman is not out.
Type II Error: Type II error (False Negative) is an error where the outcome of a test shows the acceptance of a false condition.
For example, the CT scan of a person shows that he is not having a disease but, in reality, he is having it. Here, the test accepts the false condition that the person is not having the disease.

**10. When should you use classification over regression?**

Both classification and regression are associated with prediction. Classification involves the identification of values or entities that lie in a specific group. The regression method, on the other hand, entails predicting a response value from a consecutive set of outcomes.
The classification method is chosen over regression when the output of the model needs to yield the belongingness of data points in a dataset to a particular category.
For example, we have some names of bikes and cars. We would not be interested in finding how these names are correlated to bikes and cars. Rather, we would check whether each name belongs to the bike category or to the car category.

**11. Explain Logistic Regression.**

Logistic regression is the proper regression analysis used when the dependent variable is categorical or binary. Like all regression analyses, logistic regression is a technique for predictive analysis. Logistic regression is used to explain data and the relationship between one dependent binary variable and one or more independent variables. Also, it is employed to predict the probability of a categorical dependent variable.
We can use logistic regression in the following scenarios:
To predict whether a citizen is a Senior Citizen (1) or not (0)
To check whether a person is having a disease (Yes) or not (No)
There are three types of logistic regression:
Binary Logistic Regression: In this, there are only two outcomes possible.
Example: To predict whether it will rain (1) or not (0)
Multinomial Logistic Regression: In this, the output consists of three or more unordered categories.
Example: Prediction on the regional languages (Kannada, Telugu, Marathi, etc.)
Ordinal Logistic Regression: In ordinal logistic regression, the output consists of three or more ordered categories.
Example: Rating an Android application from 1 to 5 stars.

**12. Imagine, you are given a dataset consisting of variables having more than 30% missing values. Let’s say, out of 50 variables, 8 variables have missing values, which is higher than 30%. How will you deal with them?**

To deal with the missing values, we will do the following:
We will specify a different class for the missing values.
Now, we will check the distribution of values, and we would hold those missing values that are defining a pattern.
Then, we will charge these into a yet another class, while eliminating others.

**13. How do you handle the missing or corrupted data in a dataset?**

In Python Pandas, there are two methods that are very useful. We can use these two methods to locate the lost or corrupted data and discard those values:
isNull(): For detecting the missing values, we can use the isNull() method.
dropna(): For removing the columns/rows with null values, we can use the dropna() method.
Also, we can use fillna() to fill the void values with a placeholder value.

**14. Explain Principal Component Analysis (PCA).**

Firstly, this is one of the most important Machine Learning Interview Questions.
In the real world, we deal with multi-dimensional data. Thus, data visualization and computation become more challenging with the increase in dimensions. In such a scenario, we might have to reduce the dimensions to analyze and visualize the data easily. We do this by:
 Removing irrelevant dimensions
 Keeping only the most relevant dimensions
This is where we use Principal Component Analysis (PCA).
Finding a fresh collection of uncorrelated dimensions (orthogonal) and ranking them on the basis of variance are the goals of Principal Component Analysis.
The Mechanism of PCA:
Compute the covariance matrix for data objects
Compute the Eigen vectors and the Eigen values in a descending order
To get the new dimensions, select the initial N Eigen vectors
Finally, change the initial n-dimensional data objects into N-dimensions
Example: Below are the two graphs showing data points (objects) and two directions: one is ‘green’ and the other is ‘yellow.’ We got the Graph 2 by rotating the Graph 1 so that the x-axis and y-axis represent the ‘green’ and ‘yellow’ directions, respectively.

![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/Explain-Principal-Component-Analysis-PCA.png "Subtitle")


![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/Output-from-PCA.png "Subtitle")

After the rotation of the data points, we can infer that the green direction (x-axis) gives us the line that best fits the data points.
Here, we are representing 2-dimensional data. But in real-life, the data would be multi-dimensional and complex. So, after recognizing the importance of each direction, we can reduce the area of dimensional analysis by cutting off the less-significant ‘directions.’
Now, we will look into another important Machine Learning Interview Question on PCA.

**15. Why rotation is required in PCA? What will happen if you don’t rotate the components?**

Rotation is a significant step in PCA as it maximizes the separation within the variance obtained by components. Due to this, the interpretation of components becomes easier.
The motive behind doing PCA is to choose fewer components that can explain the greatest variance in a dataset. When rotation is performed, the original coordinates of the points get changed. However, there is no change in the relative position of the components.
If the components are not rotated, then we need more extended components to describe the variance.

**16. We know that one hot encoding increases the dimensionality of a dataset, but label encoding doesn’t. How?**

When we use one hot encoding, there is an increase in the dimensionality of a dataset. The reason for the increase in dimensionality is that, for every class in the categorical variables, it forms a different variable.
Example: Suppose, there is a variable ‘Color.’ It has three sub-levels as Yellow, Purple, and Orange. So, one hot encoding ‘Color’ will create three different variables as Color.Yellow, Color.Porple, and Color.Orange.
In label encoding, the sub-classes of a certain variable get the value as 0 and 1. So, we use label encoding only for binary variables.
This is the reason that one hot encoding increases the dimensionality of data and label encoding does not.

**17. How can you avoid overfitting?**

Overfitting happens when a machine has an inadequate dataset and it tries to learn from it. So, overfitting is inversely proportional to the amount of data.

For small databases, we can bypass overfitting by the cross-validation method. In this approach, we will divide the dataset into two sections. These two sections will comprise testing and training sets. To train the model, we will use the training dataset and, for testing the model for new inputs, we will use the testing dataset.
This is how we can avoid overfitting.

**18. Why do we need a validation set and a test set?**

We split the data into three different categories while creating a model:
Training set: We use the training set for building the model and adjusting the model’s variables. But, we cannot rely on the correctness of the model build on top of the training set. The model might give incorrect outputs on feeding new inputs.

Validation set: We use a validation set to look into the model’s response on top of the samples that don’t exist in the training dataset. Then, we will tune hyperparameters on the basis of the estimated benchmark of the validation data.
When we are evaluating the model’s response using the validation set, we are indirectly training the model with the validation set. This may lead to the overfitting of the model to specific data. So, this model won’t be strong enough to give the desired response to the real-world data.
Test set: The test dataset is the subset of the actual dataset, which is not yet used to train the model. The model is unaware of this dataset. So, by using the test dataset, we can compute the response of the created model on hidden data. We evaluate the model’s performance on the basis of the test dataset.
Note: We always expose the model to the test dataset after tuning the hyperparameters on top of the validation set.
As we know, the evaluation of the model on the basis of the validation set would not be enough. Thus, we use a test set for computing the efficiency of the model.

**19. What is a Decision Tree?**

A decision tree is used to explain the sequence of actions that must be performed to get the desired output. It is a hierarchical diagram that shows the actions.

![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/What-is-a-Decision-Tree.png "Subtitle")


We can create an algorithm for a decision tree on the basis of the hierarchy of actions that we have set.
In the above decision tree diagram, we have made a sequence of actions for driving a vehicle with/without a license.

**20. Explain the difference between KNN and K-means Clustering.**

K-nearest neighbors: It is a supervised Machine Learning algorithm. In KNN, we give the identified (labeled) data to the model. Then, the model matches the points based on the distance from the closest points.

![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/K-nearest-neighbors.png "Subtitle")


K-means clustering: It is an unsupervised Machine Learning algorithm. In this, we give the unidentified (unlabeled) data to the model. Then, the algorithm creates batches of points based on the average of the distances between distinct points.

![Alt text]([url](https://intellipaat.com/blog/wp-content/uploads/2019/12/K-means-clustering.png) "Subtitle")


**21. What is Dimensionality Reduction?**

In the real world, we build Machine Learning models on top of features and parameters. These features can be multi-dimensional and large in number. Sometimes, the features may be irrelevant and it becomes a difficult task to visualize them.
Here, we use dimensionality reduction to cut down the irrelevant and redundant features with the help of principal variables. These principal variables are the subgroup of the parent variables that conserve the feature of the parent variables.

**22. Both being tree-based algorithms, how is Random Forest different from Gradient Boosting Algorithm (GBM)?**

The main difference between a random forest and GBM is the use of techniques. Random forest advances predictions using a technique called ‘bagging.’ On the other hand, GBM advances predictions with the help of a technique called ‘boosting.’
Bagging: In bagging, we apply arbitrary sampling and we divide the dataset into N After that, we build a model by employing a single training algorithm. Following, we combine the final predictions by polling. Bagging helps increase the efficiency of the model by decreasing the variance to eschew overfitting.
Boosting: In boosting, the algorithm tries to review and correct the inadmissible predictions at the initial iteration. After that, the algorithm’s sequence of iterations for correction continues until we get the desired prediction. Boosting assists in reducing bias and variance, both, for making the weak learners strong.

**23. Suppose, you found that your model is suffering from high variance. Which algorithm do you think could handle this situation and why?**

Handling High Variance
For handling issues of high variance, we should use the bagging algorithm.
Bagging algorithm would split data into sub-groups with replicated sampling of random data.
Once the algorithm splits the data, we use random data to create rules using a particular training algorithm.
After that, we use polling for combining the predictions of the model.

**24. What is ROC curve and what does it represent?**

ROC stands for ‘Receiver Operating Characteristic.’ We use ROC curves to represent the trade-off between True and False positive rates, graphically.
In ROC, AUC (Area Under the Curve) gives us an idea about the accuracy of the model.

![Alt text](https://intellipaat.com/blog/wp-content/uploads/2019/12/What-is-ROC-curve-and-what-does-it-represent.png "Subtitle")


The above graph shows an ROC curve. Greater the Area Under the Curve better the performance of the model.
Next, we would be looking at Machine Learning Interview Questions on Rescaling, Binarizing, and Standardizing.

**25. What is Rescaling of data and how is it done?**

In real-world scenarios, the attributes present in data will be in a varying pattern. So, rescaling of the characteristics to a common scale gives benefit to algorithms to process the data efficiently.
We can rescale the data using Scikit-learn. The code for rescaling the data using MinMaxScaler is as follows:

```
#Rescaling data
import pandas
import scipy
import numpy

from sklearn.preprocessing import MinMaxScaler

names = ['Abhi', 'Piyush', 'Pranay', 'Sourav', 'Sid', 'Mike', 'pedi', 'Jack', 'Tim']
Dataframe = pandas.read_csv(url, names=names)
Array = dataframe.values

# Splitting the array into input and output
X = array[:,0:8]
Y = array[:,8]

Scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# Summarizing the modified data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
```

**26. What is Binarizing of data? How to Binarize?**

In most of the Machine Learning Interviews, apart from theoretical questions, interviewers focus on the implementation part. So, this ML Interview Questions in focused on the implementation of the theoretical concepts.
Converting data into binary values on the basis of threshold values is known as the binarizing of data. The values that are less than the threshold are set to 0 and the values that are greater than the threshold are set to 1. This process is useful when we have to perform feature engineering, and we can also use it for adding unique features.
We can binarize data using Scikit-learn. The code for binarizing the data using Binarizer is as follows:

```
from sklearn.preprocessing import Binarizer
import pandas
import numpy
names = ['Abhi', 'Piyush', 'Pranay', 'Sourav', 'Sid', 'Mike', 'pedi', 'Jack', 'Tim']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# Splitting the array into input and output
X = array[:,0:8]
Y = array[:,8]
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# Summarizing the modified data
numpy.set_printoptions(precision=3)
print(binaryX[0:5,:])
```

**27. How to Standardize data?**

Standardization is the method that is used for rescaling data attributes. The attributes would likely have a value of mean as 0 and the value of standard deviation as 1. The main objective of standardization is to prompt the mean and standard deviation for the attributes.
We can standardize the data using Scikit-learn. The code for standardizing the data using StandardScaler is as follows:

```
# Python code to Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
names = ['Abhi', 'Piyush', 'Pranay', 'Sourav', 'Sid', 'Mike', 'pedi', 'Jack', 'Tim']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# Separate the array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# Summarize the transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
```

**28. Executing a binary classification tree algorithm is a simple task. But, how does a tree splitting take place? How does the tree determine which variable to break at the root node and which at its child nodes?**

Gini index and Node Entropy assist the binary classification tree to take decisions. Basically, the tree algorithm determines the feasible feature that is used to distribute data into the most genuine child nodes.
According to Gini index, if we arbitrarily pick a pair of objects from a group, then they should be of identical class and the possibility for this event should be 1.
To compute the Gini index, we should do the following:
Compute Gini for sub-nodes with the formula: The sum of the square of probability for success and failure (p^2 + q^2)
Compute Gini for split by weighted Gini rate of every node of the split
Now, Entropy is the degree of indecency that is given by the following:
where a and b are the probabilities of success and failure of the node
When Entropy = 0, the node is homogenous
When Entropy is high, both groups are present at 50–50 percent in the node.
Finally, to determine the suitability of the node as a root node, the entropy should be very low.

**29. What is SVM (Support Vector Machines)?**

SVM is a Machine Learning algorithm that is majorly used for classification. It is used on top of the high dimensionality of the characteristic vector.
Below is the code for the SVM classifier:

```
# Introducing required libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# Stacking the Iris dataset
iris = datasets.load_iris()
# A -> features and B -> label
A = iris.data
B = iris.target
# Breaking A and B into train and test data
A_train, A_test, B_train, B_test = train_test_split(A, B, random_state = 0)
# Training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(A_train, B_train)
svm_predictions = svm_model_linear.predict(A_test)
# Model accuracy for A_test
accuracy = svm_model_linear.score(A_test, B_test)
# Creating a confusion matrix
cm = confusion_matrix(B_test, svm_predictions)
```

**30. Implement the KNN classification algorithm.**

We will use the Iris dataset for implementing the KNN classification algorithm.

```
# KNN classification algorithm
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
iris_dataset=load_iris()
A_train, A_test, B_train, B_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
kn = KNeighborsClassifier(n_neighbors=1) 
kn.fit(A_train, B_train)
A_new = np.array([[8, 2.5, 1, 1.2]])
prediction = kn.predict(A_new)
print("Predicted target value: {}\n".format(prediction))
print("Predicted feature name: {}\n".format
(iris_dataset["target_names"][prediction]))
print("Test score: {:.2f}".format(kn.score(A_test, B_test)))
Output:
Predicted Target Name: [0]
Predicted Feature Name: [‘ Setosa’]
Test Score: 0.92
```



