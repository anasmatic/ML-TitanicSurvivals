import sys
import pandas
import numpy as np

titanic_train = pandas.read_csv("train.csv");
#print titanic_train.head()
#print titanic_train.describe()
#fix missing Age
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())
#titanic_train["Age"].median() = 28 which i suspect to be the cause of accuracy low rate
#enumerate Age into groups
"""
titanic_train.loc[np.logical_and(titanic_train["Age"] >= 0 , titanic_train["Age"] <16),"Age"] = 0
titanic_train.loc[np.logical_and(titanic_train["Age"] >= 16 , titanic_train["Age"] <50),"Age"] = 1
titanic_train.loc[np.logical_and(titanic_train["Age"] >= 50 , titanic_train["Age"] <65),"Age"] = 2
titanic_train.loc[titanic_train["Age"] >= 65 ,"Age"] = 3
"""
titanic_train.loc[np.logical_and(titanic_train["Age"] >= 0 , titanic_train["Age"] <16),"Age"] = 0
titanic_train.loc[np.logical_and(titanic_train["Age"] >= 16 , titanic_train["Age"] <50),"Age"] = 1
titanic_train.loc[np.logical_and(titanic_train["Age"] >= 50 , titanic_train["Age"] <65),"Age"] = 0
titanic_train.loc[titanic_train["Age"] >= 65 ,"Age"] = 1
### set Sex => 1=(male) , 0=(female)
titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 1
titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 0

### set Embarked => S for Southampton = 0
###                 C for Cherbourg = 1
###                 Q for Queenstown = 2
###                 NaN will be treated as S
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")
#print(titanic_train["Embarked"].unique())
titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2

#print titanic_train.describe()
#print pandas.DataFrame(titanic_train)
#print(titanic_train["Embarked"].unique())
#print titanic_train.describe()
#print titanic_train.loc[titanic_train["Fare"]==512.3292]
#print titanic_train.loc[titanic_train["Fare"]==0]
"""
a7a = zip(titanic_train["Age"],titanic_train["Survived"])
age0 , age1 , age2 , age3 = 0,0,0,0
surv0 , surv1 , surv2 , surv3 = 0,0,0,0
for i in a7a:
    if i[0] == 0 :
        age0 = age0+1
        if i[1] == 1:
            surv0 = surv0+1
    elif i[0] == 1 :
        age1 = age1+1
        if i[1] == 1:
            surv1 = surv1+1
    elif i[0] == 2 :
        age2 = age2+1
        if i[1] == 1:
            surv2 = surv2+1
    elif i[0] == 3 :
        age3 = age3+1
        if i[1] == 1:
            surv3 = surv3+1
print "age0 =",surv0,"of", age0 ,"=",(surv0/float(age0))
print "age1 =",surv1,"of", age1 ,"=",(surv1/float(age1))
print "age2 =",surv2,"of", age2 ,"=",(surv2/float(age2))
print "age3 =",surv3,"of", age3 ,"=",(surv3/float(age3))
sys.exit()
"""
a7a = zip(titanic_train["Name"],titanic_train["Survived"])
for i in sorted(a7a):
    print i
sys.exit()

from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# The columns we'll use to predict the target
features = ["Pclass", "Sex", "Age"]#, "Embarked","Fare","SibSp","Parch"

X = (titanic_train[features])
# The target we're using to train the algorithm.
y = titanic_train["Survived"]
# Initi alize our algorithm class
clf = DecisionTreeClassifier(criterion = "entropy", random_state=0)
cv_score = cross_val_score(clf, X, y, cv=5)
clf.fit(X,y)
print "cv= 5 mean:" , np.mean(cv_score)
print "scr on training:" , clf.score(X,y)
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.

titanic_test = pandas.read_csv("test.csv")
titanic_result = pandas.read_csv("gender_submission.csv")
#fix missing Age
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_train["Age"].median())
#titanic_train["Age"].median() = 28 which i suspect to be the cause of accuracy low rate
#enumerate Age into groups
titanic_test.loc[np.logical_and(titanic_test["Age"] >= 0 , titanic_test["Age"] <16),"Age"] = 0
titanic_test.loc[np.logical_and(titanic_test["Age"] >= 16 , titanic_test["Age"] <50),"Age"] = 1
titanic_test.loc[np.logical_and(titanic_test["Age"] >= 50 , titanic_test["Age"] <65),"Age"] = 0
titanic_test.loc[titanic_test["Age"] >= 65 ,"Age"] = 1

### set Sex => 1=(male) , 0=(female)
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0

### set Embarked => S for Southampton = 0
###                 C for Cherbourg = 1
###                 Q for Queenstown = 2
###                 NaN will be treated as S
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
#print(titanic_train["Embarked"].unique())
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_train["Fare"].median())

X_test = (titanic_test[features])
y_test = titanic_result["Survived"]
pred = clf.predict(X_test)
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": pred
    })
submission.to_csv("kaggle.csv", index=False)
print "score= ",clf.score(X_test,y_test)
#accuracy_arr = (predictions==titanic_train["Survived"])
#accuracy = (accuracy_arr==True).sum() / float(len(accuracy_arr))
#print "accuracy"
#print accuracy
import pydotplus
from IPython.display import Image  
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=features,  
#                         class_names=["PassengerId","Survived"],
#                         filled=True, rounded=True,
                         impurity=True,
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("titanic.pdf") 
Image(graph.create_png())  
