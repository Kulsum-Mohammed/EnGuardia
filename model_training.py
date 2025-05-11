# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter
import os

def main():
    # Create saved_models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Load dataset - UPDATE THIS PATH TO WHERE YOUR DATA WILL BE
    df = pd.read_csv('data/UNSW_NB15.csv')

    # Drop irrelevant columns
    df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'time' in col.lower()], 
            inplace=True, errors='ignore')

    # Clean categorical data
    df['service'] = df['service'].replace('-', np.nan)
    df['attack_cat'] = df['attack_cat'].replace('-', np.nan)
    df.dropna(subset=['service', 'attack_cat'], inplace=True)

    # Save original labels before encoding
    y_labels = df['attack_cat'].astype(str)

    # Encode categorical features (excluding the label column)
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('attack_cat')
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Encode target labels
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y_labels)

    # Final features and target
    X = df.drop(columns=['attack_cat'])
    y = y_encoded

    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_feature_names = X.columns[selector.get_support()].tolist()

    # Handle class imbalance
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_selected, y)
    print("\nResampled class distribution:", Counter(y_resampled))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Save feature names with the scaler
    scaler.feature_names_in_ = selected_feature_names
    joblib.dump(scaler, 'saved_models/nids_scaler.joblib')

    # Initialize classifier
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=len(label_encoder_y.classes_),
        eval_metric='mlogloss',
        random_state=42
    )

    # Perform Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate with cross_val_score (F1 weighted)
    scores = cross_val_score(model, X_scaled, y_resampled, cv=skf, scoring='f1_weighted')
    print(f"\nCross-validated weighted F1 scores: {scores}")
    print(f"Mean F1-score: {scores.mean():.4f}")

    # Predict with cross_val_predict for a full report
    y_pred_cv = cross_val_predict(model, X_scaled, y_resampled, cv=skf)
    print("\nCross-Validation Classification Report:")
    print(classification_report(y_resampled, y_pred_cv, target_names=label_encoder_y.classes_))

    # Fit final model on full data
    model.fit(X_scaled, y_resampled)

    # Save model and components
    joblib.dump(model, 'saved_models/nids_xgb_model.joblib')
    joblib.dump(scaler, 'saved_models/nids_scaler.joblib')
    joblib.dump(selector, 'saved_models/nids_feature_selector.joblib')
    joblib.dump(label_encoder_y, 'saved_models/nids_label_encoder.joblib')
    joblib.dump(label_encoders, 'saved_models/nids_input_encoders.joblib')
    joblib.dump(selected_feature_names, 'saved_models/nids_selected_features.joblib')

    # Show label mapping
    print("\nLabel encoding mapping (label index → attack category):")
    for i, label in enumerate(label_encoder_y.classes_):
        print(f"{i}: {label}")

    # Optional: Generate sample predictions
    print_sample_predictions()

def print_sample_predictions():
    """Function to print sample predictions from the test data"""
    try:
        # Load the dataset - UPDATE THIS PATH TO MATCH YOUR DATA LOCATION
        df_sample = pd.read_csv('data/UNSW_NB15.csv')

        # Clean the data (same as training)
        df_sample['service'] = df_sample['service'].replace('-', pd.NA)
        df_sample['attack_cat'] = df_sample['attack_cat'].replace('-', pd.NA)
        df_sample.dropna(subset=['service', 'attack_cat'], inplace=True)

        # Load saved components
        selected_feature_names = joblib.load('saved_models/nids_selected_features.joblib')
        label_encoders = joblib.load('saved_models/nids_input_encoders.joblib')
        label_encoder_y = joblib.load('saved_models/nids_label_encoder.joblib')
        scaler = joblib.load('saved_models/nids_scaler.joblib')
        model = joblib.load('saved_models/nids_xgb_model.joblib')

        # Get unique attack categories
        attack_categories = df_sample['attack_cat'].unique()
        samples = {}

        # Prepare samples for each attack category
        for cat in attack_categories:
            row = df_sample[df_sample['attack_cat'] == cat].iloc[0]
            sample_input = {}
            for feature in selected_feature_names:
                if feature in label_encoders:
                    sample_input[feature] = str(row[feature])
                else:
                    sample_input[feature] = row[feature]
            samples[cat] = sample_input

        # Print predictions
        print("\nSample Predictions:")
        print("=" * 40)
        for actual, sample_input in samples.items():
            input_df = pd.DataFrame([sample_input], columns=selected_feature_names)
            for col in label_encoders:
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            predicted_label = label_encoder_y.inverse_transform([prediction])[0]

            print(f"Actual: {actual:15s} → Predicted: {predicted_label}")
            print(f"Input Features: {list(sample_input.keys())}\n")
    except Exception as e:
        print(f"Could not generate sample predictions: {str(e)}")

if __name__ == "__main__":
    main()
