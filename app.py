from flask import Flask, render_template,  request, jsonify
#from ml_model import find  # Import other necessary functions from ml_model.py
#from database import open_another_window  # Import other necessary functions from database.py
from flask_pymongo import PyMongo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from pymongo import MongoClient
app = Flask(__name__)

# Check MongoDB connection status
def check_mongo_connection():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        # Access a specific database (replace 'your_database_name' with your actual database name)
        db = client.Malnutrition
        names = db.list_collection_names()
        print("Connected Succesfully",names)
    except Exception as e:
        print(f"Failed to connect to MongoDB: {str(e)}")
        
check_mongo_connection()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("malnutrition.csv")

# Assuming the actual column names are 'Country_Short_Name', 'Year_period', etc.
print(data.head())

# Drop rows with missing values
data = data.dropna()


datacopy = data.copy()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_columns = datacopy.select_dtypes(exclude='number').columns
for column in cat_columns:
  le.fit(datacopy[column])
  datacopy[column] = le.fit_transform(datacopy[column])
  datacopy[column] = datacopy[column]+1
datacopy
print(datacopy.shape)
datacopy = datacopy.dropna()
print(datacopy.isnull().sum())
print(datacopy.shape)
from sklearn.impute import SimpleImputer
categorical_imputer = SimpleImputer(strategy = 'most_frequent')
numerical_imputer = SimpleImputer(strategy = 'mean')
for column in data.columns:
  if(datacopy[column].dtype == 'int64' or datacopy[column].dtype == 'float'):
    datacopy[column]=numerical_imputer.fit_transform(datacopy[column].values.reshape(-1,1))
  else:
    datacopy[column]=categorical_imputer.fit_transform(datacopy[column].values.reshape(-1,1))
print(datacopy.isnull().sum())
print(datacopy.shape)
data = datacopy.copy()
datacopy = data.copy()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cat_columns = datacopy.select_dtypes(exclude='number').columns
for column in cat_columns:
  le.fit(datacopy[column])
  datacopy[column] = le.fit_transform(datacopy[column])
  datacopy[column] = datacopy[column]+1
datacopy
data = datacopy.copy()
print(data)
x = data[["Country Short Name","Year period","Median Year","Start Month","End Month","Age","Sex","MUAC(IN)","WHZ","Weight(kg)","Height(in)","Head Circumference(cm)"]]
y = data['JME (Y/N)']

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
sam = RandomUnderSampler(random_state  = 0)
x_resampled_under,y_resampled_under = sam.fit_resample(x,y)

from imblearn.over_sampling import RandomOverSampler
sam = RandomOverSampler(random_state  = 0)
x_resampled_over,y_resampled_over = sam.fit_resample(x,y)

import pandas as pd
datacopy = pd.concat([x_resampled_over,y_resampled_over],axis = 1)
print(datacopy.shape)
datacopy
data = datacopy.copy()
data
x = data[["Country Short Name","Year period","Median Year","Start Month","End Month","Age","Sex","MUAC(IN)","WHZ","Weight(kg)","Height(in)","Head Circumference(cm)"]]
y = data['JME (Y/N)']
from sklearn.feature_selection import SelectKBest,chi2
bestfeatures = SelectKBest(score_func=chi2,k=5)
fit = bestfeatures.fit(x,y)
feature_names = list(x.columns)
feature_scores = pd.DataFrame({'Feature':feature_names,'Score':fit.scores_})
feature_scores = feature_scores.sort_values(by='Score',ascending=False)
feature_scores
from sklearn.feature_selection import SelectKBest,mutual_info_classif
bestfeatures = SelectKBest(score_func=chi2,k=5)
fit = bestfeatures.fit(x,y)
feature_names = list(x.columns)
feature_scores = pd.DataFrame({'Feature':feature_names,'Score':fit.scores_})
feature_scores = feature_scores.sort_values(by='Score',ascending=False)
feature_scores
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
corr = datacopy.corr()
sns.heatmap(corr,annot=True)
plt.show()
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
features = datacopy[["Country Short Name","Year period","Median Year","Start Month","End Month","Age","Sex","MUAC(IN)","WHZ","Weight(kg)","Height(in)","Head Circumference(cm)"]]
y = datacopy['JME (Y/N)']
featurescopy = features.copy()
selectObject = MinMaxScaler().fit(features)
selectedFeatutes = selectObject.transform(features)
selectedFeatutesDF = DataFrame(selectedFeatutes,columns=featurescopy.columns)
selectedFeatutesDF
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
features = datacopy[["Country Short Name","Year period","Median Year","Start Month","End Month","Age","Sex","MUAC(IN)","WHZ","Weight(kg)","Height(in)","Head Circumference(cm)"]]
y = datacopy['JME (Y/N)']
featurescopy = features.copy()
selectObject = StandardScaler().fit(features)
selectedFeatutes = selectObject.transform(features)
selectedFeatutesDF = DataFrame(selectedFeatutes,columns=featurescopy.columns)
selectedFeatutesDF
datacopy = pd.concat([selectedFeatutesDF,y],axis=1)
from sklearn.model_selection import train_test_split
from pandas import DataFrame
features = datacopy[["Country Short Name","Year period","Median Year","Start Month","End Month","Age","Sex","MUAC(IN)","WHZ","Weight(kg)","Height(in)","Head Circumference(cm)"]]
y = datacopy['JME (Y/N)']
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=400)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics

from sklearn import model_selection
models = [
        ('LogReg', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC()),
        ('MLP', MLPClassifier())
    ]

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
dfs     = []
target_names = ['1','2']
for name, model in models:
  kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
  cv_results = model_selection.cross_validate(model, X_train, Y_train.values.ravel(), cv=kfold, scoring=scoring)
  clf = model.fit(X_train, Y_train.values.ravel())
  y_pred = clf.predict(X_test)
  print("\n"+name)
  print(classification_report(Y_test, y_pred, target_names=target_names))
  confusionMatrix = confusion_matrix(Y_test, y_pred)
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=['1','2'])

  this_df = pd.DataFrame(cv_results)
  this_df['model'] = name
  dfs.append(this_df)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
param_grid = {
    'n_estimators': [25, 50],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [3],
    'max_leaf_nodes': [6,9]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=cv)
grid_search.fit(X_train, Y_train.values.ravel())
grid_search.best_estimator_



def findMal(df,df2):
  combined_df = pd.concat([df, df2], axis=1)
  doc = combined_df.to_dict(orient='records')[0]
  print(df)
  
  le = preprocessing.LabelEncoder()
  cat_columns = df.select_dtypes(exclude='number').columns
  for column in cat_columns:
    le.fit(df[column])
    df[column] = le.fit_transform(df[column])
    df[column] = df[column]+1
  datacopy
  # Transform user input using the same scaler used for training
  user_transformed = selectObject.transform(df)

  # Make prediction
  prediction = grid_search.predict(user_transformed)

  fin=str(prediction)[1:-1]
  if fin == "2":
    Ans = "Malnurist"
  else : 
    Ans = "Not Malnurist"
 
  doc['outcome'] = Ans
  print(doc)
  client = MongoClient('mongodb://localhost:27017/')
  # Access the 'Malnutrition' database
  db = client.Malnutrition

  # Access a specific collection within the database (replace 'your_collection_name' with the actual collection name)
  collection = db.Datas

  result = collection.insert_one(doc)
  print(f"Inserted document with ID: {result.inserted_id}")
  
  return Ans

# Define Flask routes and functions
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/find', methods=['POST'])
def find():
    data = {
        'CountryShortName': request.form.get('Country Short Name'),
        'YearPeriod': request.form.get('Year Period'),
        'MedianYear': request.form.get('Median Year'),
        'StartMonth': request.form.get('Start Month'),
        'EndMonth': request.form.get('End Month'),
        'Age': request.form.get('Age'),
        'Sex': request.form.get('Sex'),
        'MUAC': request.form.get('MUAC'),
        'WHZ': request.form.get('WHZ'),
        'Weight': request.form.get('Weight(k6)'),
        'Height': request.form.get('Height(in)'),
        'HeadCircumference': request.form.get('HeadCircumference(cm)')
    }
    data2 = {
       'Name': request.form.get('Name'),
        'FatherName': request.form.get('Father Name'),
        'MotherName': request.form.get('Mother Name'),
        'RationCardNo': request.form.get('Ration Card No'),
        'Pincode': request.form.get('Pincode'),
        'HealthCenterNo': request.form.get('Health center No')
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([data])
    df2 = pd.DataFrame([data2])
    res = findMal(df,df2)
    print(res)
    return jsonify(res)

@app.route('/Show')
def Show():
    client = MongoClient('mongodb://localhost:27017/')
    # Access the 'Malnutrition' database
    db = client.Malnutrition

    # Access a specific collection within the database (replace 'your_collection_name' with the actual collection name)
    collection = db.Datas

    data_from_mongo = list(collection.find({}, {'_id': 0}))

    # Convert data to a DataFrame
    df = pd.DataFrame(data_from_mongo)

    # Render the HTML page with the Bootstrap table
    return render_template('view_datas.html', data=df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)