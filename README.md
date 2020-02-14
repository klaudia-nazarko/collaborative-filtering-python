# Collaborative Filtering Recommender System with Python

![books recommendations](img/books_header.jpg)



**Collaborative filtering** is a technique commonly used to build personalized recommendations in online products. Among companies using the collaborative filtering technology we can find some popular websites like: Amazon, Netflix, IMDB. In collaborative filtering, algorithms are used to make automatic predictions about a user's interests by compiling preferences from several users.

The main focus of this repository is to build collaborative filtering recommender systems for a **Book-Crossing dataset**. It contains data about book ratings collected in a 4-week crawl in 2004 as well as detailed information about books and users. Further details on the dataset are given in this publication:

> [Improving Recommendation Lists Through Topic Diversification](http://www2.informatik.uni-freiburg.de/~dbis/Publications/05/WWW05.html),
>
> Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen; *Proceedings of the 14th International World Wide Web Conference (WWW '05),* May 10-14, 2005, Chiba, Japan. *To appear.*



------

**Contents:**

1. [**Preprocessing of Book-Crossing dataset**](book-crossing-preprocessing.ipynb) - the script includes loading data in the correct format, filtering out incorrect rows and reducing dimensionality of the dataset.
2. [**Exploratory Data Analysis of Book-Crossing dataset**](book-crossing-eda.ipynb) - the analysis provides insights about distribution of ratings, most popular readings and characteristics of users giving the scores.
3. [**Memory-based approach to Collaborative Filtering**](collaborative-filtering-memory-based.ipynb) - memory based algorithms apply statistical techniques to the entire dataset to calculate the predictions. In this notebook two methods are compared (user-user and user-item) and the model is optimized to provide the best predictions.
4. [**Model-based approach to Collaborative Filtering**](collaborative-filtering-model-based.ipynb) - model based approach involves building machine learning algorithms to predict user's ratings. In this notebook SVD and NMF methods are compared and the model is optimized to provide the best predictions.

------

**Reference:**

1. https://surprise.readthedocs.io/en/stable/getting_started.html#getting-started
2. https://realpython.com/build-recommendation-engine-collaborative-filtering/
3. https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
4. https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b