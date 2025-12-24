import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score,accuracy_score
import matplotlib.pyplot as plt
import shap


warnings.filterwarnings('ignore')


df = pd.read_csv('data.csv')
# print(df.head())

# print(df.info())

df.drop(['Unnamed: 32','id'],axis=1,inplace=True)

# print(df.isnull().sum())

x = df.drop('diagnosis',axis=1)
y = df['diagnosis']

y = y.map({'B':0,'M':1})

# print(y)

# print(y.value_counts())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=26)

# models = {
#     "LogisticRegression" : LogisticRegression(),
#     "RandomForest" : RandomForestClassifier(),
#     "SVM" : SVC(),
#     "AdaBoost" : AdaBoostClassifier()
# }
parameters_log = {
    "Logistic_Regression__penalty" :  ['l1','l2','elasticnet',None],
    "Logistic_Regression__solver" : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
}
log_regression = LogisticRegression()

Pipeline_logistic = Pipeline([
    ("Standard_scaler",StandardScaler()),
    ("Logistic_Regression",LogisticRegression())
])
logistic_tuned = RandomizedSearchCV(estimator=Pipeline_logistic,cv=3,n_iter=100,verbose=3,n_jobs=-1,param_distributions=parameters_log,scoring='recall')

logistic_tuned.fit(x_train,y_train)

logistic_predicted_y = logistic_tuned.predict(x_test)

parameters_forest = {
    "n_estimators" : [100,200,300],
    "max_depth" : [None,5,10,20],
    "criterion" : ['gini','entropy','log_loss'],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features" : ['sqrt','log2']
}

Random_Forest = RandomForestClassifier()

RandomForest_tuned = RandomizedSearchCV(estimator=Random_Forest,n_iter=100,cv=3,verbose=3,n_jobs=-1,scoring='recall',param_distributions=parameters_forest)

RandomForest_tuned.fit(x_train,y_train)

randomforest_predicted_y = RandomForest_tuned.predict(x_test)

parameters_svm = {
    'SVC__C': [0.01, 0.1, 1, 10, 100],
    'SVC__kernel' : ['linear','poly','rbf','sigmoid'],
    'SVC__gamma' : ['scale','auto'],
    'SVC__class_weight': [None, 'balanced']
}

Pipeline_svm = Pipeline([
    ('Standard_scaler',StandardScaler()),
    ("SVC",SVC()) ]
)

svc_tuned = RandomizedSearchCV(estimator=Pipeline_svm,cv=3,n_iter=100,scoring='recall',n_jobs=-1,verbose=3,param_distributions=parameters_svm)

svc_tuned.fit(x_train,y_train)

svc_predicted_y = svc_tuned.predict(x_test)

AdaBoost_params = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
}

adaboost = AdaBoostClassifier()

adaboost_tuned = RandomizedSearchCV(estimator=adaboost,n_iter=20,cv=3,param_distributions=AdaBoost_params,verbose=3,n_jobs=-1,scoring='recall')

adaboost_tuned.fit(x_train,y_train)

adaboost_predicted_y = adaboost_tuned.predict(x_test)

results = {
    "Model" : ['Logistic Regression','Random Forest','SVM','AdaBoost'],
    "Recall" : [recall_score(y_test, logistic_predicted_y),
        recall_score(y_test, randomforest_predicted_y),
        recall_score(y_test, svc_predicted_y),
        recall_score(y_test, adaboost_predicted_y)],
    "Accuracy" : [
        accuracy_score(y_test, logistic_predicted_y),
        accuracy_score(y_test, randomforest_predicted_y),
        accuracy_score(y_test, svc_predicted_y),
        accuracy_score(y_test, adaboost_predicted_y)
    ]
}

result_df = pd.DataFrame(results)


print("----------------------LOGISTIC REGRESSION----------------------")
print(f"Confusion Matrix :\n{confusion_matrix(y_test,logistic_predicted_y)}")
print(f"Recall Score :\n{recall_score(y_test,logistic_predicted_y)}")
print(f"Accuracy :\n{accuracy_score(y_test,logistic_predicted_y)}")

Log_best_params =logistic_tuned.best_params_

print("----------------------BEST PARAMETERS----------------------")
print(Log_best_params,"\n\n")

print("*"*100)

print("----------------------RANDOM FOREST----------------------")
print(f"Confusion Matrix :\n{confusion_matrix(y_test,randomforest_predicted_y)}")
print(f"Recall Score :\n{recall_score(y_test,randomforest_predicted_y)}")
print(f"Accuracy :\n{accuracy_score(y_test,randomforest_predicted_y)}")

randomforest_best_params = RandomForest_tuned.best_params_
RandomForest_best_score = RandomForest_tuned.best_score_

print("----------------------BEST PARAMETERS----------------------")
print(randomforest_best_params,"\n\n")

print("*"*100)

print("----------------------SVC----------------------")
print(f"Confusion Matrix :\n{confusion_matrix(y_test,svc_predicted_y)}")
print(f"Recall Score :\n{recall_score(y_test,svc_predicted_y)}")
print(f"Accuracy :\n{accuracy_score(y_test,svc_predicted_y)}")


print("----------------------BEST PARAMETERS----------------------")
svm_best_params = svc_tuned.best_params_
print(svm_best_params,"\n\n")

print("*"*100)

print("----------------------AdaBoost----------------------")
print(f"Confusion Matrix :\n{confusion_matrix(y_test,adaboost_predicted_y)}")
print(f"Recall Score :\n{recall_score(y_test,adaboost_predicted_y)}")
print(f"Accuracy :\n{accuracy_score(y_test,adaboost_predicted_y)}")

adaboost_best_params = adaboost_tuned.best_params_
print("----------------------BEST PARAMETERS----------------------")
adaboost_best_params = adaboost_tuned.best_params_
adaboost_best_score = adaboost_tuned.best_score_
print(adaboost_best_params,"\n\n")

print("*"*100)

print(result_df)


plt.figure()
plt.bar(result_df['Model'],result_df['Recall'])
plt.xlabel('Model')
plt.ylabel('Recall Score')
plt.ylim(0, 1)
plt.title('Model Comparision - Recall')
plt.show()


plt.figure()
plt.bar(result_df['Model'],result_df['Accuracy'])
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
plt.title('Model Comparision - Accuracy')
plt.show()

best_rf = RandomForest_tuned.best_estimator_

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(x_test)

shap.summary_plot(
    shap_values,
    x_test,
    feature_names=x_test.columns
)

shap.summary_plot(
    shap_values,
    x_test,
    plot_type="bar",
    feature_names=x_test.columns
)
