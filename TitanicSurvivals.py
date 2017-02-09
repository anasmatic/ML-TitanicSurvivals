import sys
import pandas
titanic_train = pandas.read_csv("train.csv");
#print titanic_train.head()
#print titanic_train.describe()
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

### set Sex => 1=(male) , 0=(female)
titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 1
titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 0

### set Embarked => S for Southampton = 0
###                 C for Cherbourg = 1
###                 Q for Queenstown = 2
###                 NaN will be treated as S
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")
print(titanic_train["Embarked"].unique())
titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2

#print titanic_train.describe()
#print pandas.DataFrame(titanic_train)
print(titanic_train["Embarked"].unique())

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic_train[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic_train["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic_train[predictors].iloc[test,:])
    predictions.append(test_predictions)

import numpy as np

# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0


accuracy_arr = (predictions==titanic_train["Survived"])
accuracy = (accuracy_arr==True).sum() / float(len(accuracy_arr))
print "accuracy"
print accuracy
