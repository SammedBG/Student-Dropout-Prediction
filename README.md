# Student Dropout Prediction

## Project Overview
This project predicts which students are at risk of dropping out from college using machine learning. By identifying at-risk students early, educators can provide timely interventions to improve student retention and success.

---

## Dataset
- **Source:** Publicly available student performance dataset (`student-mat.csv`)  
- **Size:** 395 entries, 33 columns  
- **Features:** Demographic, academic, social, and lifestyle information  
- **Target:**  
  - **Regression:** `G3` (final grade)  
  - **Classification:** `dropout` (1 = at risk, 0 = safe)  

**Important Features:**
- `age` – Age of student  
- `Medu`, `Fedu` – Mother’s and Father’s education level  
- `studytime` – Weekly study hours  
- `failures` – Number of past class failures  
- `absences` – Number of school absences  
- `G1`, `G2`, `G3` – Grades for periods 1, 2, and final  
- Other social/lifestyle features like `activities`, `internet`, `romantic`, etc.

---

## Data Preprocessing
1. Dropped target columns from features (`G3` or `dropout`)  
2. Encoded categorical variables using `OneHotEncoder`  
3. Scaled numeric variables using `StandardScaler`  
4. Split the dataset into training (80%) and test (20%) sets  

No missing values were present in the dataset.

---

## Exploratory Data Analysis (EDA)
- Checked data types, summary statistics, and null values  
- Correlation with final grade (`G3`):  
  - Positive: `G1`, `G2`, `Medu`  
  - Negative: `failures`, `age`, `goout`  
  - Dropout negatively correlated with final grade

---

## Modeling

### Regression (Predict G3)
- **Linear Regression**  
  - MSE: 5.66  
  - R² Score: 0.72  

### Classification (Predict Dropout)
- **Logistic Regression**  
  - Accuracy: 0.91  
  - Confusion Matrix:  
    ```
    [[48  4]
     [ 3 24]]
    ```  

- **Random Forest Classifier**  
  - Accuracy: 0.91  
  - Confusion Matrix:  
    ```
    [[48  4]
     [ 3 24]]
    ```  

- **XGBoost Classifier**  
  - Accuracy: 0.86  
  - Confusion Matrix:  
    ```
    [[46  6]
     [ 5 22]]
    ```  

**Observation:** Random Forest and Logistic Regression performed best on this dataset.

---

## Feature Importance
- Most influential features in predicting dropout:  
  - `G1`, `G2` (previous grades)  
  - `studytime`  
  - `failures`  
  - `absences`  
  - `Medu`, `Fedu` (parental education)

---

## Conclusion
- The ML pipeline accurately predicts at-risk students (~91% accuracy with Random Forest)  
- Educators can use the model for early intervention  
- The pipeline can be extended with more features or deployed as a web/desktop app

---

## Future Work
- Hyperparameter tuning using GridSearchCV  
- Ensemble methods for improved performance  
- Model explainability using SHAP/LIME  
- Incorporate additional student data for more accurate predictions

---

## Technologies Used
- **Programming Language:** Python 3  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, xgboost  
- **Environment:** Virtualenv / Jupyter Notebook / VS Code  

---

## How to Run
1. Clone the repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
3. Run the scripts:
    -Data exploration: ```python data_exploration.py```
    -Preprocessing:``` python preprocessing.py```
    -Modeling: ```python modeling.py```
    -Random Forest & XGBoost:``` python xgboost_rf.py```
