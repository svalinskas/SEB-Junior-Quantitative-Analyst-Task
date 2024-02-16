# SEB Junior Quantitative Analyst Task

Here are three tasks, each of which may be completed independently of the others:

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
For more complex filtering, for example, clients with a balance of at least 1000, no personal loan, and older than 40, we can use the query method to filter with one string:
```python
queried_sample = df.query['balance>=1000 & loan=='no' & age>40')
```
If we deem some columns as unnecessary, we can drop some them, and also rename others:
```python
df_cleaned = (df
              .drop(columns=['contact', 'day', 'month'])
              .rename(columns={'balance':'account_balance', 'y':'subscribed'})
              )
```
We can then use the describe method to retrieve some summarizing statistics for the whole subsample:
```python
sample_statistics = df_cleaned.describe()
print(sample_statistics)
                age  account_balance  ...         pdays      previous
count  45211.000000     45211.000000  ...  45211.000000  45211.000000
mean      40.936210      1362.272058  ...     40.197828      0.580323
std       10.618762      3044.765829  ...    100.128746      2.303441
min       18.000000     -8019.000000  ...     -1.000000      0.000000
25%       33.000000        72.000000  ...     -1.000000      0.000000
50%       39.000000       448.000000  ...     -1.000000      0.000000
75%       48.000000      1428.000000  ...     -1.000000      0.000000
max       95.000000    102127.000000  ...    871.000000    275.000000
```
We can also aggregate by some groups and retrieve some more specific statistics:
```python
grouped_statistics = df_cleaned.groupby('job').agg({
                     'account_balance': ['mean', 'median', 'std'],
                     'age': ['min', 'max'],
                     'duration': ['mean', 'sum']
                     })
print(grouped_statistics)
              account_balance                     age        duration         
                         mean median          std min max        mean      sum
job                                                                           
admin.            1135.838909  396.0  2641.962686  20  75  246.896732  1276703
blue-collar       1078.826654  388.0  2240.523208  20  75  262.901562  2558558
entrepreneur      1521.470074  352.0  4153.442626  21  84  256.309348   381132
housemaid         1392.395161  406.0  2984.692098  22  83  245.825000   304823
management        1763.616832  572.0  3822.965605  21  81  253.995771  2402292
retired           1984.215106  787.0  4397.044177  24  95  287.361307   650586
self-employed     1647.970868  526.0  3684.259573  22  76  268.157061   423420
services           997.088108  339.5  2164.493505  20  69  259.318729  1077210
student           1388.060768  502.0  2441.703526  18  48  246.656716   231364
technician        1252.632092  421.0  2548.544019  21  71  252.904962  1921319
unemployed        1521.745971  529.0  3144.666754  21  66  288.543361   375972
unknown           1772.357639  677.0  2970.288559  25  82  237.611111    68432
```
