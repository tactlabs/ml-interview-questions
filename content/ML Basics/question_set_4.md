---
title: "Question Set 4"
date: 2021-07-13T19:40:39+05:30
draft: false
---

**Question Set 4**

**1. Explain the terms Artificial Intelligence (AI), Machine Learning (ML and Deep Learning?**

Artificial Intelligence (AI) is the domain of producing intelligent machines. ML refers to systems that can assimilate from experience (training data) and Deep Learning (DL) states to systems that learn from experience on large data sets. ML can be considered as a subset of AI. Deep Learning (DL) is ML but useful to large data sets. The figure below roughly encapsulates the relation between AI, ML, and DL:


https://lh4.googleusercontent.com/-ouE3yZO_ZRxZeOGwqifKZ5ntrWy6aSjRVVPM-Qoi5Gvzm8Luan8u5fEVvRH7o37q-Ibn2F8EgFeBj2CZiNvTAY5TJjvR1UVpmDkAW_wOe5Y_1mz8QDGEUCYK_c408h3yIc2TAM

In summary, DL is a subset of ML & both were the subsets of AI.

Additional Information: ASR (Automatic Speech Recognition) & NLP (Natural Language Processing) fall under AI and overlay with ML & DL as ML is often utilized for NLP and ASR tasks.

https://lh5.googleusercontent.com/JKPNQcXJUiSjgj-gIbddFTqHFmLfspBeDR7_1syUdAYotUd2VoYIKoUI_Puw2mUgnBksLOfY2gf_SpBnRYkxuH1P-HZMMRIVkd8BYRl-RJ3VedAGDjjd0TW3Q4T2px90SVgHlvo

**2. What are the different types of Learning/ Training models in ML?**

ML algorithms can be primarily classified depending on the presence/absence of target variables.
A. Supervised learning: [Target is present]
The machine learns using labelled data. The model is trained on an existing data set before it starts making decisions with the new data.
The target variable is continuous: Linear Regression, polynomial Regression, quadratic Regression.
The target variable is categorical: Logistic regression, Naive Bayes, KNN, SVM, Decision Tree, Gradient Boosting, ADA boosting, Bagging, Random forest etc.

B. Unsupervised learning: [Target is absent]
The machine is trained on unlabelled data and without any proper guidance. It automatically infers patterns and relationships in the data by creating clusters. The model learns through observations and deduced structures in the data.
Principal component Analysis, Factor analysis, Singular Value Decomposition etc.

C. Reinforcement Learning:
The model learns through a trial and error method. This kind of learning involves an agent that will interact with the environment to create actions and then discover errors or rewards of that action.

**3. What is the difference between deep learning and machine learning?**

https://d1m75rqqgidzqn.cloudfront.net/2019/10/OCT-31-ML-infographic1.jpg

https://d1m75rqqgidzqn.cloudfront.net/2019/10/OCT-31-ML-infographic2.jpg

Machine Learning involves algorithms that learn from patterns of data and then apply it to decision making. Deep Learning, on the other hand, is able to learn through processing data on its own and is quite similar to the human brain where it identifies something, analyse it, and makes a decision.
The key differences are as follow:
The manner in which data is presented to the system.
Machine learning algorithms always require structured data and deep learning networks rely on layers of artificial neural networks.

**4. What is the main key difference between supervised and unsupervised machine learning?**

Supervised learning technique needs labeled data to train the model. For example, to solve a classification problem (a supervised learning task), you need to have label data to train the model and to classify the data into your labeled groups. Unsupervised learning does not  need any labelled dataset. This is the main key difference between supervised learning and unsupervised learning.

**5. How do you select important variables while working on a data set?**

There are various means to select important variables from a data set that include the following:
Identify and discard correlated variables before finalizing on important variables
The variables could be selected based on ‘p’ values from Linear Regression
Forward, Backward, and Stepwise selection
Lasso Regression
Random Forest and plot variable chart
Top features can be selected based on information gain for the available set of features.

**6. There are many machine learning algorithms till now. If given a data set, how can one determine which algorithm to be used for that?**

Machine Learning algorithm to be used purely depends on the type of data in a given dataset. If data is linear then, we use linear regression. If data shows non-linearity then, the bagging algorithm would do better. If the data is to be analyzed/interpreted for some business purposes then we can use decision trees or SVM. If the dataset consists of images, videos, audios then, neural networks would be helpful to get the solution accurately.
So, there is no certain metric to decide which algorithm to be used for a given situation or a data set. We need to explore the data using EDA (Exploratory Data Analysis) and understand the purpose of using the dataset to come up with the best fit algorithm. So, it is important to study all the algorithms in detail.

https://lh3.googleusercontent.com/RWwiL2SBnidXsTxn8lOujq3OA4y5H4nWJ-SvvbXUrZJTdNxs6ltECSAFuEukddIADMqqi_dsXutmHdPx4QUJR8JzDN2YtOeTORuV6jl16mdx0c5w0ZbXQL5HkQ4DY85Jl1F_SxI

**7. How are covariance and correlation different from one another?**

Covariance measures how two variables are related to each other and how one would vary with respect to changes in the other variable. If the value is positive it means there is a direct relationship between the variables and one would increase or decrease with an increase or decrease in the base variable respectively, given that all other conditions remain constant.

Correlation quantifies the relationship between two random variables and has only three specific values, i.e., 1, 0, and -1.

1 denotes a positive relationship, -1 denotes a negative relationship, and 0 denotes that the two variables are independent of each other.

**8. State the differences between causality and correlation?**

Causality applies to situations where one action, say X, causes an outcome, say Y, whereas Correlation is just relating one action (X) to another action(Y) but X does not necessarily cause Y.

https://lh5.googleusercontent.com/WXyGpuV7wUYgQv5cVCq1zYfbN28xQ2yGX0b8j2dfPXjy5RFfqC-JplGNMFCefIgJ7oSe_H40GMeMrzIwQ1UlcbttrmnMGG7T9XrkLWeVVH8ZgftvsCdFor3b8EkUXVGube5j8AU


**9. We look at machine learning software almost all the time. How do we apply Machine Learning to Hardware?**

We have to build ML algorithms in System Verilog which is a Hardware development Language and then program it onto an FPGA to apply Machine Learning to hardware.

**10. Explain One-hot encoding and Label Encoding. How do they affect the dimensionality of the given dataset?**

One-hot encoding is the representation of categorical variables as binary vectors. Label Encoding is converting labels/words into numeric form. Using one-hot encoding increases the dimensionality of the data set. Label encoding doesn’t affect the dimensionality of the data set. One-hot encoding creates a new variable for each level in the variable whereas, in Label encoding, the levels of a variable get encoded as 1 and 0.

https://d1m75rqqgidzqn.cloudfront.net/wp-data/2020/06/05174859/June-3_ML-infograph-for-blog-1.png





