# Fraud Detection in Transactions

This repository contains a project focused on detecting fraudulent transactions using classification models. The dataset and the methodology are detailed below.

## Dataset
The dataset used in this project is sourced from Kaggle:
[Fraudulent Transactions Data](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data)

### Dataset Overview
- **Source**: Kaggle
- **Description**: The dataset contains transaction records, including features indicative of fraudulent activity.
- **Size**: Over 6 million records
- **Target**: Binary classification to determine if a transaction is fraudulent (1) or non-fraudulent (0).

## Models Used
This project employs the following machine learning models:
- **Random Forest**: A robust ensemble learning method leveraging decision trees for classification.
- **XGBoost**: An advanced gradient boosting algorithm designed for optimized performance and accuracy.

## Objective
To accurately detect fraudulent transactions by analyzing patterns in transaction data using machine learning classification techniques.

## Project Structure
- `dataSet/`: Contains raw and processed dataset files.
- `models/`: Trained models and scripts for training.

## Setup
To run this project locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/rikta5/FraudDetectionInTransactions.git
   ```
2. Navigate to the project directory:
   ```bash
   cd FraudDetectionInTransactions
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Results
The project evaluates the performance of the classification models using metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score

## Acknowledgments
Special thanks to [Chitwan Manchanda](https://www.kaggle.com/chitwanmanchanda) for providing the dataset used in this project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

