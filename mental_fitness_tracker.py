import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import warnings
import pickle
warnings.filterwarnings("ignore")
# Load the datasets
df_dalys = pd.read_csv('daly.csv')
df_prevalence = pd.read_csv('pd.csv')
df_merged = pd.merge(df_dalys, df_prevalence, on=['Entity', 'Code', 'Year'])
# Feature Engineering
# Select relevant features for mental fitness tracking
features = [ 'Schizophrenia','Bipolar disorder','Eating disorders',' Anxiety disorders','Drug use disorders','Depressive disorders','Alcohol use disorders']
#set axis
df_merged.set_axis(['Country','Code','Year','DALY','Schizophrenia','Bipolar disorder','Eating disorders',' Anxiety disorders','Drug use disorders','Depressive disorders','Alcohol use disorders'], axis='columns', inplace='True')
mean=df_merged['DALY'].mean()
# Split the dataset into features (X) and target variable (y)
X = df_merged[features]
y = df_merged['DALY']
y = y.astype('int')
X = X.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

#print('Mean Squared Error:', mse)
Score= r2_score(y_test,y_pred)
#print('r2 Score of Model:', Score)
# Real-Time Tracking (Example: Predict mental fitness label for a new data point)
new_data = pd.DataFrame([[ 0.1, 0.2, 0.3, 0.4, 0.5,0.1,0.2]], columns=features)
prediction = model.predict(new_data)
#print('Predicted Mental Fitness Label:', prediction)
inputt=[float(x) for x in "1 1 1 1 1 1 1".split(' ')]
final=[inputt]

b = model.predict(final)


pickle.dump(model,open('modell.pkl','wb'))
modell=pickle.load(open('model.pkl','rb'))