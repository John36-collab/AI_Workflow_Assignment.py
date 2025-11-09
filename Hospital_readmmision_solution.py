import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_data():
    """Load the hospital readmission dataset"""
    df = pd.read_csv('data/hospital_readmissions_30k.csv')
    return df

def analyze_data(df):
    """Perform comprehensive data analysis"""
    print("=== DATASET ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nTarget variable distribution:")
    print(df['readmitted_30_days'].value_counts())
    print(f"\nReadmission rate: {df['readmitted_30_days'].value_counts(normalize=True)['Yes']:.2%}")
    
    print("\n=== CATEGORICAL VARIABLES ===")
    print("Gender distribution:")
    print(df['gender'].value_counts())
    print("\nDischarge destination distribution:")
    print(df['discharge_destination'].value_counts())
    print("\nDiabetes status:")
    print(df['diabetes'].value_counts())
    print("\nHypertension status:")
    print(df['hypertension'].value_counts())
    
    print("\n=== NUMERICAL VARIABLES ===")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """Preprocess the data for modeling"""
    print("\n=== DATA PREPROCESSING ===")
    
    # Convert blood pressure to separate systolic and diastolic
    bp_split = df['blood_pressure'].str.split('/', expand=True)
    df['systolic_bp'] = pd.to_numeric(bp_split[0])
    df['diastolic_bp'] = pd.to_numeric(bp_split[1])
    
    # Drop original blood pressure column
    df = df.drop('blood_pressure', axis=1)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_diabetes = LabelEncoder()
    le_hypertension = LabelEncoder()
    le_discharge = LabelEncoder()
    le_readmitted = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['diabetes_encoded'] = le_diabetes.fit_transform(df['diabetes'])
    df['hypertension_encoded'] = le_hypertension.fit_transform(df['hypertension'])
    df['discharge_destination_encoded'] = le_discharge.fit_transform(df['discharge_destination'])
    df['readmitted_encoded'] = le_readmitted.fit_transform(df['readmitted_30_days'])
    
    # Drop original categorical columns
    df = df.drop(['gender', 'diabetes', 'hypertension', 'discharge_destination', 'readmitted_30_days', 'patient_id'], axis=1)
    
    # Feature engineering: Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
    le_age = LabelEncoder()
    df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
    df = df.drop('age_group', axis=1)
    
    # Feature engineering: BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    le_bmi = LabelEncoder()
    df['bmi_category_encoded'] = le_bmi.fit_transform(df['bmi_category'])
    df = df.drop('bmi_category', axis=1)
    
    # Feature engineering: Create risk score based on medical conditions
    df['medical_risk_score'] = df['diabetes_encoded'] + df['hypertension_encoded'] + df['medication_count']
    
    # Feature engineering: Length of stay category
    df['los_category'] = pd.cut(df['length_of_stay'], bins=[0, 3, 7, 10, 20], labels=['Short', 'Medium', 'Long', 'Very Long'])
    le_los = LabelEncoder()
    df['los_category_encoded'] = le_los.fit_transform(df['los_category'])
    df = df.drop('los_category', axis=1)
    
    print("Feature engineering completed:")
    print("- Blood pressure split into systolic and diastolic")
    print("- Age groups created")
    print("- BMI categories created")
    print("- Medical risk score calculated")
    print("- Length of stay categories created")
    
    return df

def train_models(X, y):
    """Train multiple models and compare their performance"""
    print("\n=== MODEL TRAINING ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # Evaluate models
    print("Random Forest Performance:")
    print(classification_report(y_test, rf_pred))
    print(f"Precision: {precision_score(y_test, rf_pred, pos_label=1):.3f}")
    print(f"Recall: {recall_score(y_test, rf_pred, pos_label=1):.3f}")
    
    print("\nLogistic Regression Performance:")
    print(classification_report(y_test, lr_pred))
    print(f"Precision: {precision_score(y_test, lr_pred, pos_label=1):.3f}")
    print(f"Recall: {recall_score(y_test, lr_pred, pos_label=1):.3f}")
    
    # Cross-validation
    rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='f1')
    lr_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='f1')
    
    print(f"\nRandom Forest F1 Score (CV): {rf_scores.mean():.3f} (+/- {rf_scores.std() * 2:.3f})")
    print(f"Logistic Regression F1 Score (CV): {lr_scores.mean():.3f} (+/- {lr_scores.std() * 2:.3f})")
    
    return rf_model, lr_model, X_test, y_test, rf_pred, lr_pred, scaler

def create_confusion_matrix(y_test, y_pred, model_name):
    """Create and display confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved for {model_name}")

def analyze_feature_importance(model, feature_names):
    """Analyze and plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_imp.head(10))
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved")
        
        return feature_imp

def generate_assignment_report():
    """Generate comprehensive assignment solution report"""
    print("\n=== ASSIGNMENT SOLUTION REPORT ===")
    
    # Load and analyze data
    df = load_data()
    analyze_data(df)
    
    # Preprocess data
    df_processed = preprocess_data(df.copy())
    
    # Prepare features and target
    X = df_processed.drop('readmitted_encoded', axis=1)
    y = df_processed['readmitted_encoded']
    feature_names = X.columns.tolist()
    
    # Train models
    rf_model, lr_model, X_test, y_test, rf_pred, lr_pred, scaler = train_models(X, y)
    
    # Create confusion matrices
    create_confusion_matrix(y_test, rf_pred, "Random Forest")
    create_confusion_matrix(y_test, lr_pred, "Logistic Regression")
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(rf_model, feature_names)
    
    # Print assignment answers
    print("\n" + "="*50)
    print("ASSIGNMENT SOLUTIONS")
    print("="*50)
    
    print("\n1. PROBLEM SCOPE:")
    print("   Problem: Predict patient readmission within 30 days of discharge")
    print("   Objectives:")
    print("   - Identify high-risk patients for early intervention")
    print("   - Reduce readmission rates through targeted care")
    print("   - Optimize resource allocation for at-risk patients")
    print("   Stakeholders: Hospital administrators, healthcare providers, insurance companies, patients")
    
    print("\n2. DATA STRATEGY:")
    print("   Proposed data sources:")
    print("   - Electronic Health Records (EHRs)")
    print("   - Patient demographics and medical history")
    print("   - Laboratory results and vital signs")
    print("   - Medication records and adherence data")
    print("   - Social determinants of health")
    
    print("\n   Ethical concerns:")
    print("   1. Patient privacy and data security")
    print("   2. Algorithmic bias and fairness in healthcare")
    
    print("\n   Preprocessing pipeline:")
    print("   1. Data cleaning and handling missing values")
    print("   2. Feature engineering (age groups, BMI categories, risk scores)")
    print("   3. Encoding categorical variables")
    print("   4. Feature scaling and normalization")
    print("   5. Handling class imbalance (SMOTE if needed)")
    
    print("\n3. MODEL DEVELOPMENT:")
    print("   Selected model: Random Forest")
    print("   Justification:")
    print("   - Handles both numerical and categorical features well")
    print("   - Robust to outliers and non-linear relationships")
    print("   - Provides feature importance insights")
    print("   - Good performance with imbalanced datasets")
    
    print("\n4. DEPLOYMENT:")
    print("   Integration steps:")
    print("   1. Model validation and performance testing")
    print("   2. API development for model inference")
    print("   3. Integration with hospital EHR system")
    print("   4. User interface development for clinicians")
    print("   5. Continuous monitoring and retraining pipeline")
    print("   6. A/B testing of model predictions")
    
    print("\n   HIPAA compliance:")
    print("   - Data encryption at rest and in transit")
    print("   - Role-based access controls")
    print("   - Audit logging of all data access")
    print("   - Regular security assessments")
    print("   - Patient data anonymization where possible")
    
    print("\n5. OPTIMIZATION:")
    print("   Method to address overfitting: Regularization techniques")
    print("   - L1/L2 regularization in logistic regression")
    print("   - Pruning decision trees in Random Forest")
    print("   - Cross-validation for hyperparameter tuning")
    print("   - Feature selection to reduce model complexity")
    
    print("\n6. ETHICS & BIAS:")
    print("   Impact of biased training data:")
    print("   - Unequal care allocation across demographic groups")
    print("   - Reinforcement of existing healthcare disparities")
    print("   - False positives leading to unnecessary interventions")
    print("   - False negatives missing high-risk patients")
    
    print("\n   Bias mitigation strategy:")
    print("   - Use stratified sampling to ensure representation")
    print("   - Regular bias audits across demographic groups")
    print("   - Diverse development team to identify potential biases")
    print("   - Fairness metrics monitoring in production")
    
    print("\n7. TRADE-OFFS:")
    print("   Model interpretability vs accuracy:")
    print("   - Simple models (logistic regression) are more interpretable")
    print("   - Complex models (Random Forest, neural networks) are more accurate")
    print("   - Healthcare often prioritizes interpretability for clinical trust")
    print("   - Hybrid approaches: Use simple models for explanations + complex for predictions")
    
    print("\n   Impact of limited computational resources:")
    print("   - Preference for simpler models (logistic regression, decision trees)")
    print("   - Reduced feature engineering complexity")
    print("   - Smaller batch sizes for training")
    print("   - Less frequent model retraining cycles")
    print("   - Cloud-based solutions may be more cost-effective than on-premise")
    
    # Save feature importance
    if feature_importance is not None:
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("\nFeature importance analysis saved to 'feature_importance.csv'")

if __name__ == "__main__":
    generate_assignment_report()
