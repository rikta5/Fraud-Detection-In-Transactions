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

# Train or Load Random Forest Model
if not os.path.exists(RF_MODEL_PATH):
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, RF_MODEL_PATH)
else:
    rf_clf = joblib.load(RF_MODEL_PATH)

# Evaluate Random Forest
y_rf_pred = rf_clf.predict(X_test)
y_rf_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
print("Random Forest - Classification Report:\n", classification_report(y_test, y_rf_pred))
print("Random Forest - ROC-AUC Score:", roc_auc_score(y_test, y_rf_pred_proba))

# Train or Load XGBoost Model
if not os.path.exists(XGB_MODEL_PATH):
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
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
    random_search.fit(X_train, y_train)
    final_xgb_clf = XGBClassifier(**random_search.best_params_, random_state=42, n_jobs=-1)
    final_xgb_clf.fit(X_train, y_train)
    joblib.dump(final_xgb_clf, XGB_MODEL_PATH)
else:
    final_xgb_clf = joblib.load(XGB_MODEL_PATH)

# Evaluate XGBoost
y_xgb_pred = final_xgb_clf.predict(X_test)
y_xgb_pred_proba = final_xgb_clf.predict_proba(X_test)[:, 1]
print("XGBoost - Classification Report:\n", classification_report(y_test, y_xgb_pred))
print("XGBoost - ROC-AUC Score:", roc_auc_score(y_test, y_xgb_pred_proba))

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

# Feature Importance
rf_importances = rf_clf.feature_importances_
indices_rf = np.argsort(rf_importances)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(range(X_train.shape[1]), rf_importances[indices_rf], align='center')
plt.yticks(range(X_train.shape[1]), X_train.columns[indices_rf])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# Visualize XGBoost
# Confusion Matrix
xgb_cm = confusion_matrix(y_test, y_xgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.title('XGBoost Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_xgb_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='r', label='XGBoost (AUC = %0.2f)' % roc_auc_score(y_test, y_xgb_pred_proba))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('XGBoost ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Feature Importance
xgb_importances = final_xgb_clf.feature_importances_
indices_xgb = np.argsort(xgb_importances)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(range(X_train.shape[1]), xgb_importances[indices_xgb], align='center')
plt.yticks(range(X_train.shape[1]), X_train.columns[indices_xgb])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()
