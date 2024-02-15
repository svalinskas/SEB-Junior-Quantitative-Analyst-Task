# SEB Junior Quantitative Analyst Task

Here are the three tasks, each of which may be completed independently of the others:

1) Data Manipulation Task:
  - Select a random subsample of the dataset;
  - Filter the desired rows using simple and more complex conditions;
  - Drop unnecessary variables, rename some variables;
  - Calculate summarizing statistics (for the whole sample and by categorical variables as well);
  - Create new variables using simple transformation and custom functions;
  - Order the dataset by several variables.

2) Data Visualization Task:
  - In order to understand the data, please visualize it. You are free to select the scope, types of plots, etc.

3) Modelling Task:
  - Perform a logistic regression to obtain the predicted probability that a customer has subscribed for a term deposit.
    Use continuous variables and dummy variables created for categorical columns. Not necessarily all variables provided in data sample should be used.
    Evaluate the model's goodness of fit and predictive ability. If needed, the dataset could be split into training and test sets.

The tasks will be completed using Python 3.12.2 and its selected libraries. Let's start with Task 1.

## Data Manipulation Task

First, for data manipulation, let's import pandas. Then, let's load the full .csv file from a local directory into the dataframe. We have to keep in mind that the delimiter in the .csv file is not the usual comma, but a semicolon:
```python
import pandas as pd
df = pd.read_csv('/Users/sarunas/Desktop/bank-full.csv', delimiter = ';')
