# Player-s-Rating-Prediction-in-FIFA-2006
In this research project, we used some machine learning techniques to evaluate the overall rankings of Portuguese football players for the 2005-2006 game season. The utilized dataset was scrapped from [3] for FIFA 06 as the following game versions' data were extensively used in many other ML codes. Three different regression models (linear, decision tree, and random tree) were used for the ranking prediction. Scatterplots and 5-fold cross validation showed that the best model for this project is linear regression.

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
sns.set(style='ticks', color_codes=True)
sns.set(style='darkgrid')

soccer_data = pd.read_csv('players_06.csv')
soccer_data.head()

soccer_data.shape

soccer_data = soccer_data.dropna(axis='columns')
soccer_data = soccer_data._get_numeric_data()
soccer_data = soccer_data.drop(columns=['age', 'height', 'weight'])

data_df = soccer_data
data_df.head()
# Histograms are useful because they show the number of instances (y-axis) that have a given range of values (x-axis). Each histogram has a range of values from 0 to 100. This is a good indication that all the attributes are at the same scale, which means we don't need to do any scaling transformations of the functions.


data_df.hist(bins=50, figsize=(30,15))
plt.show()

#Here, we split the data into into training and testing sets. We set test_size = 0.2 so that 20% of the data is kept in the test case. The remaining 80% of the data is stored in the training set. There should be more data in the training set because this is the set from which the model learns. The test set will be used to test the model.

# 5. Split the Data into Training and Testing Sets
train_set, test_set = train_test_split(data_df, test_size = 0.2, random_state=42)
print("Length of training data:", len(train_set))
print("Length of testing data:", len(test_set))
print("Length of total data:", len(data_df))

#We can calculate the standard correlation coefficient between each pair of attributes using the corr() method to see how well each attribute correlates with the overall value. The correlation coefficient ranges from -1 to 1. When the correlation coefficient is close to 1, it means that there is a strong positive correlation; for example, overall tends to increase when ball control increases. When the correlation coefficient is close to -1, it means that there is a strong negative correlation, in other words, the opposite of a strong positive correlation.

# 6. Look for Correlations
corr_matrix = train_set.corr()
corr_matrix['overall'].sort_values()

#We can use the pandas scatter_matrix () function to visualize the correlation between attributes. This function maps each numeric attribute to all other numeric attributes.

attributes = ['overall', 'ball_control', 'passing', 'stamina', 'penalties']
pd.plotting.scatter_matrix(train_set[attributes], figsize=(15,12))
plt.show()

#I have split the training and testing set into separate feature sets and goals. The DataFrame y_train and y_test include target values (target value is overall), whereas X_train and X_test include all other attributes that correlate with target value.

y_train = train_set['overall']
X_train = train_set.drop('overall', axis=1)
y_test = test_set['overall']
X_test = test_set.drop('overall', axis=1)

#Firstly, we trained a Linear Regression model and measured the RMSE of the regression model on the entire training set.

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_predictions = lin_reg.predict(X_train)
lin_rmse = np.sqrt(mean_squared_error(y_train, y_predictions))
print(lin_rmse)

scores = cross_val_score(lin_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5)
lin_reg_scores = np.sqrt(-scores)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

#Secondly, Decision Tree Regressor was trained and measured the RMSE of the regression model.

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

y_predictions = tree_reg.predict(X_train)
tree_rmse = np.sqrt(mean_squared_error(y_train, y_predictions))
print(tree_rmse)

scores = cross_val_score(tree_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5)
tree_scores = np.sqrt(-scores)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

#Thirdly, we trained a Random Dorest Regressor and the RMSE of the regression model was measured.

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

y_predictions = forest_reg.predict(X_train)
forest_rmse = np.sqrt(mean_squared_error(y_train, y_predictions))
print(forest_rmse)

scores = cross_val_score(forest_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5)
forest_scores = np.sqrt(-scores)
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

#Comparing the standard deviation of these three models, the result of the Linear Regression model is the best one.

from sklearn.model_selection import GridSearchCV
param_grid = [
{'normalize': [True, False], 'fit_intercept': [True, False]}
]

grid_search = GridSearchCV(lin_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)

#Now, we have selected our model. We would like to see how the forecasts differ from the actual target. To do this, simply select seven instances from the test set and use the predict() method in the final model:

final_model = grid_search.best_estimator_
data = X_test.iloc[:7]
label = y_test.iloc[:7]
print("Predictions:", final_model.predict(data))
print("Labels:", list(label))

#This project showed us that we can build a machine learning model to predict the overall ranking of a player in FIFA. Three different regression models were compared with each other and the best one was Linear Regression model with the smallest standard deviation stdv = 1.18.
grid_search.best_params_
