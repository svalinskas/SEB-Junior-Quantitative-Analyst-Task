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
```
To see if the file was loaded correctly, we can check its first few rows and shape:
```python
print(df.head())
   age           job  marital  education  ... pdays  previous poutcome   y
0   58    management  married   tertiary  ...    -1         0  unknown  no
1   44    technician   single  secondary  ...    -1         0  unknown  no
2   33  entrepreneur  married  secondary  ...    -1         0  unknown  no
3   47   blue-collar  married    unknown  ...    -1         0  unknown  no
4   33       unknown   single    unknown  ...    -1         0  unknown  no

[5 rows x 17 columns]

print(df.shape)
(45211, 17)
```
As we can see, the dataframe has 17 columns, as intended.

To select a random subsample of the dataset, we can use pandas' sample() method and specify what fraction of the dataset we want. For variety, let's randomly select only from among rows that satisfy a condition:
```python
subsample = df[df['balance'] > df['balance].median()].sample(frac=0.05)
```
The condition within square brackets only selects rows with above-median 'balance' values, 10% of which are then randomly sampled.

Next, let's filter the desired rows using a simple condition:
```python
clients_with_loan = df[df['loan'] == 'yes']
```
For more complex filtering, for example, clients with a balance, we can use the query method:
```python
queried_data = df.query['balance > 1000 & loan == 'no'
```
We can drop some unnecessary columns and rename others:
```python
df = df.drop(columns='day', 'month']
df = df.rename(columns={'balance':'account_balance', 'y':'subscribed'}) 
