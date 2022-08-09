# Heart Disease
* This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
* Original Data -- https://archive.ics.uci.edu/ml/datasets/heart+Disease
* Kaggle Data --  https://www.kaggle.com/ronitf/heart-disease-uci

* We're going to take the following approach:

1. Problem definition
2. Data
3. Evaluation
4. Features
5. Modelling
6. Experimentation

# AIM
To find whether the person is having heart disease or not.

# Resource Used
* Python : v3.8 Jupyter Notebook : v6.4
* **Packages** : Pandas : v1.4.3, Numpy : v1.23.0, Matplotlib : v3.5.2, Matplotlib-inline : v0.1.3, Seaborn : v0.10.1, Scikit-Learn : v1.1.1
* Different Machine Learning Model such as RandomForestClassifier, LogisticRegression, KNeighborsClassifier used to get the high **Accuracy Score** and also Hyperparameter tuning is done to get the best result using **RandomizedSearchCV** and **GridSearchCV**.

# EDA (Exploratory Data Analysis)
* There were no null values in the dataset so EDA was done directly.

1. age - age in years
2. sex - (1 = male; 0 = female)
3. cp - chest pain type
        0: Typical angina: chest pain related decrease blood supply to the heart
        1: Atypical angina: chest pain not related to heart
        2: Non-anginal pain: typically esophageal spasms (non heart related)
        3: Asymptomatic: chest pain not showing signs of disease
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
5. chol - serum cholestoral in mg/dl
        serum = LDL + HDL + .2 * triglycerides
        above 200 is cause for concern
6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
        '>126' mg/dL signals diabetes
7. restecg - resting electrocardiographic results
        0: Nothing to note
        1: ST-T Wave abnormality
            can range from mild symptoms to severe problems
            signals non-normal heart beat
        2: Possible or definite left ventricular hypertrophy
            Enlarged heart's main pumping chamber
8. thalach - maximum heart rate achieved
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
11. slope - the slope of the peak exercise ST segment
        0: Upsloping: better heart rate with excercise (uncommon)
        1: Flatsloping: minimal change (typical healthy heart)
        2: Downslopins: signs of unhealthy heart
12. ca - number of major vessels (0-3) colored by flourosopy
        colored vessel means the doctor can see the blood passing through
        the more blood movement the better (no clots)
13. thal - thalium stress result
        1,3: normal
        6: fixed defect: used to be defect but ok now
        7: reversable defect: no proper blood movement when excercising
14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

* The dataset have total of **303 rows × 14 columns**.

* Getting the target value_counts
![Target](https://github.com/sumeet860/Heart-Disease/blob/main/target_heart.png?raw=True "Target Count")

* Checking the Heart Disease frequency with respect to Sex.
![Heart Disease with Sex](https://github.com/sumeet860/Heart-Disease/blob/main/frequency_heart.png?raw=True "Heart Disease with Sex")
* More number of males have Heart Disease compare to females.

* Heart Disease frequency with age, with respect to **max heart rate** .
![Heart Disease with Max Heart Rate](https://github.com/sumeet860/Heart-Disease/blob/main/target_heart.png?raw=True "Heart Disease with Max Heart Rate")

* Age count
![Age](https://github.com/sumeet860/Heart-Disease/blob/main/age_heart.png?raw=True "Age")


* Heart Disease with respect to Chest Pain Type
![Chest type Heart Disease](https://github.com/sumeet860/Heart-Disease/blob/main/frequency_chest_heart.png?raw=True "Chest type Heart Disease")


# Correlation Matrix
![Correlation Matrix](https://github.com/sumeet860/Heart-Disease/blob/main/correlation_matrix_heart.png?raw=True "correlation matrix")

# **Machine Learning**

* Splitting the data into train and test model. Have to find how many have heart disease or not assigning target column to 'y' and other columns to 'X'.
* Using following machine learning models from scikit-learn.
* **RandomForestClassifier()**
* **KNearestNeighbor()**
* **LogisticRegression()**

* For each models we get different scores as follows:
* {'Logistic Regression': 0.8852459016393442,
 'KNN': 0.6885245901639344,
 'Random Forest': 0.8360655737704918}
* Model Comparison
* # Correlation Matrix
![Model Comparison](https://github.com/sumeet860/Heart-Disease/blob/main/model_heart.png?raw=True "correlation matrix")

* From the model it is clear that KNN performs very poor compared to other two models.
* We will do the **hyperparameter tuning** using RandomizedSearchCV and GridSearchCV on the other two models.

# RandomizedSearchCV
* **Hyperparameter tuning of LogisticRegression**
* Following parameters were used {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}
* The best parameters are {'solver': 'liblinear', 'C': 0.23357214690901212}
* With RandomizedSearchCV tuning the score is 0.885

* **Hyperparameter tuning of RandomForestClassifier**
* Following parameters were used {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
* The best parameters are {'n_estimators': 210,
            'min_samples_split': 4,
            'min_samples_leaf': 19,
            'max_depth': 3}
* With RandomizedSearchCV tuning the score is 0.885

# GridSearchCV
* Since LogisticRegression score was better than RandomForestClassifier will do tuning of LogisticRegression again.
* **Hyperparameter tuning of LogisticRegression**
* Following parameters were used {"C": np.logspace(-4, 4, 30),
                "solver": ["liblinear"]}
* The best parameters are {'C': 0.20433597178569418, 'solver': 'liblinear'}

# Evaluation
* Evaluation of the model is done using classification_report and confusion_matrix.
![Confusion matrix Heatmap](https://github.com/sumeet860/Heart-Disease/blob/main/confusion_matrix_heart.png?raw=True "Confusion matrix Heatmap")


# Conclusion 
* From Hyperparameter tuning of models we can conclude that LogisticRegression has more accuracy score, which will help in predicting the Heart Disease among the people.
