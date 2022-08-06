## Titanic Machine Learning Competition

##### V1 Score: 0.77511
following tutorial

##### V2 Score: 0.64114
implemented sklearn.ensembleHistGradientBoostingClassifier(), using ['Pclass', 'Age', 'Fare'] as features and performing a single fit.

##### V3 Score: 0.62918
adds SibSp and Parch as parameters

removed SibSp, Score: 0.63875

add SibSp, remove Parch - Score: 0.64114

add Parch, revert to tutorial model: 

attempting fix for NaN values on RandomForestClassifier here: https://stackoverflow.com/questions/30317119/classifiers-in-scikit-learn-that-handle-nan-null

current exercise: https://www.kaggle.com/code/jtreading/exercise-your-first-machine-learning-model/edit
