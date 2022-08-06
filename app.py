# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

print(train_data.columns)
print(train_data.dropna(axis=0))

y = train_data.Survived

passenger_features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']

# X is standard/conventional
X = train_data[passenger_features]
print(X.head())

# Here is an example of defining a decision tree model with scikit-learn and fitting it with the features and target variable.
passenger_model = HistGradientBoostingClassifier()

passenger_model.fit(X, y)


print("Making predictions for the following 5 passengers:")
print(X.head())
print("The predictions are")
print(passenger_model.predict(X.head()))

X_test = pd.get_dummies(test_data[passenger_features])

predictions = passenger_model.predict(X_test)
output = pd.DataFrame(
    {'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission2.csv', index=False)
print("Your submission was successfully saved!")

# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)

# print("% of women who survived:", rate_women)


# men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men)/len(men)

# print("% of men who survived:", rate_men)


# y = train_data["Survived"]

# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train_data[features])
# X_test = pd.get_dummies(test_data[features])

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame(
#     {'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")
