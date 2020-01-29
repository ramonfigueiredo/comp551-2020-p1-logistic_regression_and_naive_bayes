# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],    # The column numbers to be transformed (here is [3] but can be [0, 1, 2, 3, ..])
    remainder='passthrough'                         				 # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
'''
Linear regression is going to take care of feature scaling for us.
Therefore we don't need to do feature scaling.
'''

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
'''
Backward Elimination

Step 1: Select a significance level to stay in the model (e.g. SL = 0.05)

Step 2: Fit the full model with all possible predictors

Step 3: Consider the predictor with the highest P-value. If P > SL, go to STEP 4, otherwise go to the END (Your model is ready)

Step 4: Remove the predictor

Step 5: Fit model without this variable
'''
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
### Columns in 50_Startups.csv = R&D Spend,Administration,Marketing Spend,State,Profit
## 0 = X0 = const = 1 (np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1))
## 1 = X1 = First dummy variable for state
## 2 = X2 = Second dummy variable for state
## 3 = X3 = R&D Spend
## 4 = X4 = Administration Spend
## 5 = X5 = Marketing Spend
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# OLS = Ordinary Least Square
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Step 2
print(regressor_OLS.summary())
'''													|
													V
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607 <==
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================
'''
# Step 3, Step 4, Step 5
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Step 2
print(regressor_OLS.summary())
'''													|
													V
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138 <==
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
'''
# Step 3, Step 4, Step 5
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Step 2
print(regressor_OLS.summary())
'''													|
													V
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076 <==
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
'''
# Step 3, Step 4, Step 5
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Step 2
print(regressor_OLS.summary())
'''													|
													V
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061 <==
==============================================================================
'''
# Step 3, Step 4, Step 5
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Step 2
print(regressor_OLS.summary())
'''													|
													V
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
'''

'''
We conclude that R&D Spend (index = 3) is a very powerful predictor of the profit (y).

So finally if we follow the backward elimination algorithm then the conclusion is that 
the optimal group of independent variables that can predict a profit with the highest 
statistical significance the strongest impact is composed by only one independent variable, 
and this variable is the R&D spend.
'''