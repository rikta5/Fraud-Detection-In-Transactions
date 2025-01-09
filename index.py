import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling
import shap  # For SHAP feature importance analysis

# Paths for saving/loading
PREPROCESSED_DATA_PATH = "dataSet/Fraud_preprocessed.parquet"
RF_MODEL_PATH = "models/random_forest_model.pkl"
XGB_MODEL_PATH = "models/xgboost_model.pkl"

# Function to preprocess the data
def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Time Features
    df['hour_of_transaction'] = (df['step'] % 24).astype(int)
    df['day_of_week'] = (df['step'] // 24) % 7
    df['time_since_last'] = df.groupby('nameOrig')['step'].diff().fillna(0)

    # Transaction Features
    transaction_group = df.groupby('nameOrig')
    df['transaction_frequency'] = transaction_group['step'].transform('count')
    df['average_transaction_amount'] = transaction_group['amount'].transform('mean')
    df['transaction_variance'] = transaction_group['amount'].transform('var').fillna(0)
    df['cumulative_transaction_amount'] = transaction_group['amount'].cumsum()

    # Account and Transaction
    df['origin_dest_similarity'] = (df['nameOrig'].str[:3] == df['nameDest'].str[:3]).astype(int)
    df['transaction_direction'] = np.where(df['oldbalanceOrg'] > 0, 'outgoing', 'incoming')

    # Behaviour-Based Features
    df['is_same_balance_after'] = (df['newbalanceOrig'] == df['oldbalanceOrg']).astype(int)

    # Fraud-Centric Features
    df['dest_fraud_rate'] = df.groupby('nameDest')['isFraud'].transform('mean').fillna(0)
    df['suspicious_name_dest'] = ((df['type'] == 'TRANSFER') & (df['newbalanceDest'] == 0)).astype(int)

    # Amount Features
    df['transaction_amount_deviation'] = df['amount'] - df['average_transaction_amount']
    threshold = df['amount'].quantile(0.95)
    df['large_transaction_flag'] = (df['amount'] > threshold).astype(int)
    df['proportion_of_balance'] = (df['amount'] / (df['oldbalanceOrg'] + 1)).fillna(0)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    df.rename(columns=lambda x: x.replace('type_', 'transaction_type_'), inplace=True)
    df = pd.get_dummies(df, columns=['transaction_direction'], drop_first=True)

    return df

# Check if preprocessed data exists
if not os.path.exists(PREPROCESSED_DATA_PATH):
    # Preprocess the data and save
    df = preprocess_data("dataSet/Fraud.csv")
    df.to_parquet(PREPROCESSED_DATA_PATH)
else:
    # Load preprocessed data
    df = pd.read_parquet(PREPROCESSED_DATA_PATH)

# Define Features and Target
X = df.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'step', 'isFlaggedFraud'])
y = df['isFraud']

# Split Dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train or Load Random Forest Model
if not os.path.exists(RF_MODEL_PATH):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_clf.fit(X_train_smote, y_train_smote)
    joblib.dump(rf_clf, RF_MODEL_PATH)
else:
    rf_clf = joblib.load(RF_MODEL_PATH)

# Evaluate Random Forest with SMOTE
y_rf_pred = rf_clf.predict(X_test)
y_rf_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
print("Random Forest - Classification Report (SMOTE):\n", classification_report(y_test, y_rf_pred))
print("Random Forest - ROC-AUC Score (SMOTE):", roc_auc_score(y_test, y_rf_pred_proba))

# Train or Load XGBoost Model
if not os.path.exists(XGB_MODEL_PATH):
    scale_pos_weight = len(y_train_smote[y_train_smote == 0]) / len(y_train_smote[y_train_smote == 1])
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [scale_pos_weight],
    }
    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=50,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        random_state=42
    )
    random_search.fit(X_train_smote, y_train_smote)
    final_xgb_clf = XGBClassifier(**random_search.best_params_, random_state=42, n_jobs=-1)
    final_xgb_clf.fit(X_train_smote, y_train_smote)
    joblib.dump(final_xgb_clf, XGB_MODEL_PATH)
else:
    final_xgb_clf = joblib.load(XGB_MODEL_PATH)

# Evaluate XGBoost with SMOTE
y_xgb_pred = final_xgb_clf.predict(X_test)
y_xgb_pred_proba = final_xgb_clf.predict_proba(X_test)[:, 1]
print("XGBoost - Classification Report (SMOTE):\n", classification_report(y_test, y_xgb_pred))
print("XGBoost - ROC-AUC Score (SMOTE):", roc_auc_score(y_test, y_xgb_pred_proba))

# Visualize Random Forest
# Confusion Matrix
rf_cm = confusion_matrix(y_test, y_rf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='b', label='Random Forest (AUC = %0.2f)' % roc_auc_score(y_test, y_rf_pred_proba))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# SHAP Feature Importance for XGBoost
explainer_xgb = shap.TreeExplainer(final_xgb_clf)
shap_values_xgb = explainer_xgb.shap_values(X_test)

# Summary plot for SHAP values
shap.summary_plot(shap_values_xgb, X_test, plot_type="bar")
