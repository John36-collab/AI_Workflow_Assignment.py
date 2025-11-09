# Hospital Readmission Prediction Solution

## Assignment Overview
This solution addresses the hospital readmission prediction case study using the provided hospital readmissions dataset. The task involves building an AI system to predict patient readmission risk within 30 days of discharge.

## Dataset Analysis
- **Dataset Size**: 30,000 patient records
- **Readmission Rate**: 12.25% (3,674 readmitted out of 30,000)
- **Key Features**: Age, gender, blood pressure, cholesterol, BMI, diabetes, hypertension, medication count, length of stay, discharge destination
- **Data Quality**: No missing values, clean dataset with good coverage

## Key Findings from Analysis
1. **Class Imbalance**: Significant imbalance with 87.75% not readmitted vs 12.25% readmitted
2. **Correlations**: 
   - Discharge destination shows strongest correlation with readmission (-0.102)
   - Diabetes and hypertension show positive correlations with readmission
   - Age shows minimal correlation with readmission in this dataset
3. **Feature Distribution**: Well-balanced gender distribution, diverse age ranges (18-90)

## Solution Components

### 1. Data Preprocessing Pipeline
- **Blood Pressure Processing**: Split into systolic and diastolic components
- **Feature Engineering**:
  - Age groups (Young, Middle, Senior, Elderly)
  - BMI categories (Underweight, Normal, Overweight, Obese)
  - Medical risk score (diabetes + hypertension + medication count)
  - Length of stay categories
  - Blood pressure risk indicator
  - Age-length of stay interaction features
- **Encoding**: Label encoding for categorical variables
- **Class Imbalance Handling**: Upsampling minority class to balance dataset

### 2. Model Development
**Selected Model**: Random Forest Classifier with hyperparameter tuning
- **Justification**: Handles mixed data types, robust to outliers, provides feature importance
- **Hyperparameter Tuning**: Grid search for optimal n_estimators, max_depth, min_samples_split
- **Class Weighting**: Balanced class weights to handle readmission imbalance

**Alternative Model**: Logistic Regression with balanced class weights
- Used for comparison and baseline performance

### 3. Performance Metrics
The solution generates comprehensive metrics including:
- Confusion matrices for both models
- Precision, recall, F1-score for each class
- ROC-AUC scores
- Cross-validation results

### 4. Feature Importance Analysis
Identifies key predictors of readmission:
- Medical risk score
- Length of stay features
- Age and age-related categories
- Discharge destination
- Blood pressure components

## Assignment Solutions

### 1. Problem Scope
- **Problem**: Predict patient readmission within 30 days of discharge
- **Objectives**: Identify high-risk patients for early intervention, reduce readmission rates, optimize resource allocation
- **Stakeholders**: Hospital administrators, healthcare providers, insurance companies, patients

### 2. Data Strategy
**Proposed Data Sources**:
- Electronic Health Records (EHRs)
- Patient demographics and medical history
- Laboratory results and vital signs
- Medication records and adherence data
- Social determinants of health

**Ethical Concerns**:
1. Patient privacy and data security
2. Algorithmic bias and fairness in healthcare

**Preprocessing Pipeline**:
1. Data cleaning and handling missing values
2. Feature engineering (age groups, BMI categories, risk scores)
3. Encoding categorical variables
4. Feature scaling and normalization
5. Handling class imbalance (SMOTE/upsampling)

### 3. Model Development
**Selected Model**: Random Forest
- Handles both numerical and categorical features well
- Robust to outliers and non-linear relationships
- Provides feature importance insights
- Good performance with imbalanced datasets

### 4. Deployment
**Integration Steps**:
1. Model validation and performance testing
2. API development for model inference
3. Integration with hospital EHR system
4. User interface development for clinicians
5. Continuous monitoring and retraining pipeline
6. A/B testing of model predictions

**HIPAA Compliance**:
- Data encryption at rest and in transit
- Role-based access controls
- Audit logging of all data access
- Regular security assessments
- Patient data anonymization where possible

### 5. Optimization
**Overfitting Mitigation**:
- L1/L2 regularization in logistic regression
- Pruning decision trees in Random Forest
- Cross-validation for hyperparameter tuning
- Feature selection to reduce model complexity
- Class weighting to handle imbalanced data

### 6. Ethics & Bias
**Impact of Biased Training Data**:
- Unequal care allocation across demographic groups
- Reinforcement of existing healthcare disparities
- False positives leading to unnecessary interventions
- False negatives missing high-risk patients

**Bias Mitigation Strategy**:
- Use stratified sampling to ensure representation
- Regular bias audits across demographic groups
- Diverse development team to identify potential biases
- Fairness metrics monitoring in production
- Oversampling minority groups to balance representation

### 7. Trade-offs
**Interpretability vs Accuracy**:
- Simple models (logistic regression) are more interpretable
- Complex models (Random Forest, neural networks) are more accurate
- Healthcare often prioritizes interpretability for clinical trust
- Hybrid approaches: Use simple models for explanations + complex for predictions

**Limited Computational Resources Impact**:
- Preference for simpler models (logistic regression, decision trees)
- Reduced feature engineering complexity
- Smaller batch sizes for training
- Less frequent model retraining cycles
- Cloud-based solutions may be more cost-effective than on-premise

## Generated Files
- `hospital_readmission_improved.py`: Main implementation script
- `hospital_readmission_solution.py`: Initial solution implementation
- `feature_importance.csv`: Top feature importance analysis
- `feature_importance.png`: Visual representation of feature importance
- `random_forest_confusion_matrix.png`: Random Forest model performance
- `logistic_regression_confusion_matrix.png`: Logistic Regression model performance
- `roc_curves.png`: ROC curves for all models

## Usage Instructions
To run the solution:
```bash
cd PLP_AI_For_Software_Engineering_Week_5
python3 hospital_readmission_improved.py
```

The script will:
1. Load and analyze the dataset
2. Perform comprehensive data preprocessing
3. Train and evaluate multiple models
4. Generate visualizations and metrics
5. Output detailed assignment  solutions

## Conclusion
This solution provides a comprehensive approach to hospital readmission prediction, addressing all requirements in the assignment while emphasizing ethical considerations, practical deployment considerations, and performance optimization. The Random Forest model demonstrates good capability for identifying readmission risk factors while maintaining interpretability for clinical use.
