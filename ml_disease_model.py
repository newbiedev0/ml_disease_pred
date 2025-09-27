# %%
import pandas as pd 
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
df1 = pd.read_csv('C:/Users/venka/Downloads/parkinsons.csv')

# %%
df1.shape

# %%
df1.describe()

# %%
df1.dtypes

# %%
df1.columns

# %%
df1.isna().sum()

# %%
df1["status"].value_counts()

# %%
for i in df1.select_dtypes(include=[np.number]).columns:
    plt.boxplot(df1[i])
    plt.title(i)
    plt.show()

# %%
for i in df1.select_dtypes(include=[np.number]).columns:
    sns.kdeplot(df1[i])
    plt.show()

# %%
class OutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds = {}
        self.feature_cols = None

    def fit(self, X, y=None):
        
        numeric_X = X.select_dtypes(include=np.number)
        self.feature_cols = numeric_X.columns.tolist()
        
        for col in self.feature_cols:
            Q1 = numeric_X[col].quantile(0.25)
            Q3 = numeric_X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = {
                'lower': Q1 - (self.factor * IQR),
                'upper': Q3 + (self.factor * IQR)
            }
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in self.feature_cols:
            lower_bound = self.bounds[col]['lower']
            upper_bound = self.bounds[col]['upper']
            
            
            X_copy[col] = np.where(X_copy[col] < lower_bound, lower_bound, X_copy[col])
            X_copy[col] = np.where(X_copy[col] > upper_bound, upper_bound, X_copy[col])
        
        
        return X_copy[self.feature_cols]

# %%
X = df1.drop(columns=['name', 'status'])
y = df1['status']

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# %%
gbc = GradientBoostingClassifier(
    learning_rate=0.1, 
    max_depth=3, 
    n_estimators=200, 
    subsample=0.8,
    random_state=42
)


# %%
pipeline = Pipeline([
    ('capper', OutlierCapper(factor=1.5)), 
    ('scaler', StandardScaler()),        
    ('gbc', gbc)                          
])

# %%
pipeline.fit(X_train, y_train)

# %%
MODEL_FILE_PATH = 'parkinsons_gbc_pipeline.pkl'
joblib.dump(pipeline, MODEL_FILE_PATH)

# %%
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)
y_pred

# %%
auc_score = roc_auc_score(y_test, y_pred_proba)
auc_score

# %%
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix (Rows=Actual, Cols=Predicted):\n", confusion_matrix(y_test, y_pred))

# %%
df2 = pd.read_csv('C:/Users/venka/Downloads/kidney_disease.csv')

# %%
df2

# %%
df2.dtypes

# %%
df2.shape

# %%
df2.describe()

# %%
df2.isna().sum()

# %%
df2["classification"].value_counts()

# %%
object_data=df2.select_dtypes(include=['object'])

# %%
object_data

# %%
number_data =df2.select_dtypes(include=['float64'])

# %%
number_data

# %%
df2 = df2.drop(columns=['id'])

# %%
df2

# %%
df2['classification'] = df2['classification'].str.strip().map({'ckd': 1, 'notckd': 0})

# %%
df2["classification"]

# %%

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds = {}
        self.feature_cols = None

    def fit(self, X, y=None):
        
        X_df = pd.DataFrame(X) 
        self.feature_cols = X_df.columns.tolist() 
        
        for col in self.feature_cols:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = {
                'lower': Q1 - (self.factor * IQR),
                'upper': Q3 + (self.factor * IQR)
            }
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_cols)
        
        for col in self.feature_cols:
            if col in self.bounds:
                lower_bound = self.bounds[col]['lower']
                upper_bound = self.bounds[col]['upper']
                
                
                X_df[col] = np.where(X_df[col] < lower_bound, lower_bound, X_df[col])
                X_df[col] = np.where(X_df[col] > upper_bound, upper_bound, X_df[col])
        
        
        return X_df.values

# %%
convert_to_numeric = ['pcv', 'wc', 'rc']
for col in convert_to_numeric:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')

# %%
y = df2['classification']
X = df2.drop(columns=['classification'])

# %%
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# %%
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('capper', OutlierCapper(factor=1.5)), 
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
])

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# %%
pipeline_kidneydces = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42,C=0.1))
])


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %%
pipeline_kidneydces.fit(X_train, y_train)

# %%
MODEL_FILE_PATH = 'pipeline_kidneydces.pkl'
joblib.dump(pipeline_kidneydces, MODEL_FILE_PATH)

# %%
y_pred_proba = pipeline_kidneydces.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

# %%
y_pred_proba

# %%
auc_score

# %%
y_pred = (y_pred_proba > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix (Rows=Actual, Cols=Predicted):\n", confusion_matrix(y_test, y_pred))

# %%
df3 = pd.read_csv('C:/Users/venka/Downloads/indian_liver_patient.csv')

# %%
df3

# %%
df3.shape

# %%
df3.describe()

# %%
df3.dtypes

# %%
df3.isna().sum()

# %%
for i in df3.select_dtypes(include=[np.number]).columns:
    sns.kdeplot(df3[i])
    plt.show()

# %%
for i in df3.select_dtypes(include=[np.number]).columns:
    fig = px.box(df3, x='Gender', y=i)
    fig.update_layout(title_text=f'Distribution of {i} by Gender', xaxis_title='Gender', yaxis_title=i)
    fig.show()

# %%
df3.select_dtypes(include=object).columns

# %%
df3["Dataset"]

# %%
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds = {}
        self.feature_cols = None

    def fit(self, X, y=None):
        
        X_df = pd.DataFrame(X) 
        self.feature_cols = X_df.columns.tolist() 
        
        for col in self.feature_cols:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = {
                'lower': Q1 - (self.factor * IQR),
                'upper': Q3 + (self.factor * IQR)
            }
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_cols)
        
        for col in self.feature_cols:
            if col in self.bounds:
                lower_bound = self.bounds[col]['lower']
                upper_bound = self.bounds[col]['upper']
                
                
                X_df[col] = np.where(X_df[col] < lower_bound, lower_bound, X_df[col])
                X_df[col] = np.where(X_df[col] > upper_bound, upper_bound, X_df[col])
        
        
        return X_df.values


# %%
df3['Dataset'] = df3['Dataset'].map({1: 1, 2: 0})

# %%
df3["Dataset"]

# %%
df3['Dataset'].value_counts()

# %%
X = df3.drop(columns=['Dataset'])
y= df3['Dataset']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %%
numerical_features = X.select_dtypes(include=np.number).columns.tolist() 
categorical_features = ['Gender']

# %%
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('capper', OutlierCapper(factor=1.5)), 
    ('scaler', StandardScaler())
])

# %%
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# %%

pipeline_logistic_liver = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(C=0.1, max_iter=1000, random_state=42, class_weight='balanced')) 
])

# %%
print("\nFitting XGBoost Classifier pipeline on Liver Patient data...")
pipeline_logistic_liver.fit(X_train, y_train)

# %%
MODEL_FILE_PATH = 'pipeline_logistic_liver.pkl'
joblib.dump(pipeline_logistic_liver, MODEL_FILE_PATH)

# %%
y_pred = pipeline_logistic_liver.predict(X_test)
y_pred_proba = pipeline_logistic_liver.predict_proba(X_test)[:, 1]

print("\n----------------------------------------------------------")
print("logisitic Classifier Metrics (Liver Patient Dataset)")
print("----------------------------------------------------------")
print(f"Test Set ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Patient (0)', 'Liver Patient (1)']))
print("\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
print(confusion_matrix(y_test, y_pred))

# %%



