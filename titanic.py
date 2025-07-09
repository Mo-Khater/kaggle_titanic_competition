import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
def load_data():
    df=pd.read_csv('data/train.csv')
    return df

def preprocess_data(X):
    le= LabelEncoder()
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    X = X.drop(cols_to_drop, axis=1)
    # X = df.drop('Survived', axis=1)
    # y = df['Survived']
    X['Sex'] = le.fit_transform(X['Sex'])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    age_mean = X['Age'].mean()
    X['Age']= X['Age'].fillna(age_mean)
    # X_test['Age'] = X_test['Age'].fillna(age_mean)
    embarked_mode = X['Embarked'].mode()[0]
    X['Embarked'] = X['Embarked'].fillna(embarked_mode)
    # X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)
    X = pd.get_dummies(X, columns=['Embarked'], drop_first=True,dtype='int')
    # X_test = pd.get_dummies(X_test, columns=['Embarked'], drop_first=True,dtype='int')
    return X
    
    
def model_train(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=10,min_samples_split=5,)
    rf.fit(X_train, y_train)
    return rf

def model_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def model_prediction(X_test,model):
    preds = model.predict(X_test)
    return preds
    
# def model_evaluate(model, X_test, y_test):
#     accuracy = model.score(X_test, y_test)
#     print(f'Model Accuracy: {accuracy:.2f}')

df=load_data()
print(df.head())
X=df.drop('Survived', axis=1)
# print(df.isna().sum())
# print(df.describe())
# embarked_mode = df['Embarked'].mode()[0]
# print(df['Embarked'].value_counts())
processed_train_data = preprocess_data(X)
# print(X_train.columns)
# print(X_test.columns)
# model_tuning(X_train, y_train)

model = model_train(processed_train_data, df['Survived'])
test_data= pd.read_csv('data/test.csv')
processed_test_data = preprocess_data(test_data)
preds = model_prediction(processed_test_data, model)
passenger_ids = test_data['PassengerId']
submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds})
submission.to_csv('data/submission.csv', index=False)
# model_evaluate(model, X_test, y_test)



