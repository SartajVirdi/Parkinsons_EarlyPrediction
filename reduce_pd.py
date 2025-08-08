import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from scipy import signal
from scipy.stats import skew, kurtosis
import pickle
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
root_dir = "FullDataSet_PD-BioStampRC21"
clinical_path = "Clinic_DataPDBioStampRCStudy.csv"
sampling_rate = 512  # Hz
window_size = 2 * sampling_rate  # 2 seconds
step_size = window_size  # No overlap to reduce data size
target_size_gb = 1  # Target dataset size in GB


# --- DATA QUALITY FUNCTIONS ---
def calculate_signal_quality(data):
    """Calculate signal quality metrics"""
    # Standard deviation (activity level)
    std_dev = np.std(data, axis=0)

    # Signal magnitude area
    sma = np.mean(np.sum(np.abs(data), axis=1))

    # Energy
    energy = np.mean(np.sum(data ** 2, axis=1))

    return np.mean(std_dev), sma, energy


def extract_features(window):
    """Extract engineered features from time series window"""
    features = []

    # For each axis (X, Y, Z)
    for axis in range(3):
        axis_data = window[:, axis]

        # Time domain features
        features.extend([
            np.mean(axis_data),
            np.std(axis_data),
            np.max(axis_data),
            np.min(axis_data),
            skew(axis_data),
            kurtosis(axis_data),
            np.percentile(axis_data, 25),
            np.percentile(axis_data, 75)
        ])

        # Frequency domain features
        freqs, psd = signal.welch(axis_data, fs=sampling_rate, nperseg=min(256, len(axis_data)))
        features.extend([
            np.mean(psd),
            np.std(psd),
            freqs[np.argmax(psd)],  # dominant frequency
            np.sum(psd[:10]) / np.sum(psd)  # low frequency power ratio
        ])

    # Cross-axis features
    magnitude = np.sqrt(np.sum(window ** 2, axis=1))
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude),
        np.min(magnitude)
    ])

    # Correlation between axes
    corr_xy = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
    corr_xz = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
    corr_yz = np.corrcoef(window[:, 1], window[:, 2])[0, 1]
    features.extend([corr_xy, corr_xz, corr_yz])

    return np.array(features)


def filter_high_quality_windows(windows, labels, quality_threshold=0.8):
    """Keep only high-quality windows based on signal characteristics"""
    quality_scores = []

    for window in windows:
        std_dev, sma, energy = calculate_signal_quality(window)
        # Combine metrics (you can adjust weights)
        quality_score = (std_dev * 0.4 + sma * 0.3 + energy * 0.3)
        quality_scores.append(quality_score)

    # Keep top quality_threshold% of windows
    threshold_idx = int(len(quality_scores) * quality_threshold)
    top_indices = np.argsort(quality_scores)[-threshold_idx:]

    return np.array(windows)[top_indices], np.array(labels)[top_indices]


def create_improved_model(input_shape, num_classes):
    """Create the DNN model"""
    model = Sequential()

    # Reshape for CNN if using raw windows
    if len(input_shape) == 1:
        # For feature-based approach, use Dense layers
        model.add(Dense(256, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())

    else:
        # For raw time series, use CNN
        model.add(Conv1D(64, kernel_size=7, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(128, kernel_size=5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(256, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())


    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# === NEW: SINGLE PATIENT PREDICTION SYSTEM ===

class PDPredictor:
    """Parkinson's Disease Prediction System for Single Patients"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.is_trained = False

    def process_patient_data(self, patient_file_path):
        """Process accelerometer data for a single patient"""
        print(f"Processing patient data from: {patient_file_path}")

        # Load patient data
        df = pd.read_csv(patient_file_path)
        if df.shape[0] < window_size:
            raise ValueError(f"Insufficient data: {df.shape[0]} samples, need at least {window_size}")

        # Extract accelerometer data (skip timestamp column)
        data = df.iloc[:, 1:4].values  # Assumes columns: timestamp, X, Y, Z

        # Apply same preprocessing as training
        downsample_factor = 2
        data_downsampled = data[::downsample_factor]

        # Select middle 80% of recording
        start_idx = int(0.1 * len(data_downsampled))
        end_idx = int(0.9 * len(data_downsampled))
        data_clean = data_downsampled[start_idx:end_idx]

        # Create windows
        new_window_size = window_size // downsample_factor
        windows = []

        num_windows = min(20, len(data_clean) // new_window_size)

        for i in range(num_windows):
            start = i * new_window_size
            if start + new_window_size <= len(data_clean):
                window = data_clean[start:start + new_window_size]
                windows.append(window)

        if len(windows) == 0:
            raise ValueError("No valid windows could be created from the data")

        return np.array(windows)

    def extract_patient_features(self, windows):
        """Extract features from patient windows"""
        print(f"Extracting features from {len(windows)} windows...")

        patient_features = []
        for window in windows:
            features = extract_features(window)
            patient_features.append(features)

        return np.array(patient_features)

    def predict_patient(self, patient_file_path, return_probabilities=False, return_details=False):
        """
        Predict if a patient has Parkinson's Disease

        Args:
            patient_file_path: Path to patient's accelerometer CSV file
            return_probabilities: If True, return prediction probabilities
            return_details: If True, return detailed information about prediction

        Returns:
            prediction: 'PD' or 'Control'
            probabilities: (if requested) [Control_prob, PD_prob]
            details: (if requested) Dictionary with detailed information
        """

        if not self.is_trained:
            raise ValueError("Model must be trained first! Call train_model() before prediction.")

        try:
            # Process patient data
            windows = self.process_patient_data(patient_file_path)

            # Extract features
            features = self.extract_patient_features(windows)

            # Handle NaN values
            features = np.nan_to_num(features)

            # Apply feature selection
            features_selected = self.feature_selector.transform(features)

            # Apply scaling
            features_scaled = self.scaler.transform(features_selected)

            # Make predictions for all windows
            predictions = self.model.predict(features_scaled, verbose=0)

            # Average predictions across all windows
            avg_prediction = np.mean(predictions, axis=0)

            # Get final prediction
            predicted_class_idx = np.argmax(avg_prediction)
            predicted_class = self.label_encoder.classes_[predicted_class_idx]

            # Calculate confidence
            confidence = avg_prediction[predicted_class_idx]

            print(f"\n{'=' * 50}")
            print(f"PATIENT PREDICTION RESULT")
            print(f"{'=' * 50}")
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Probabilities: Control={avg_prediction[0]:.2%}, PD={avg_prediction[1]:.2%}")
            print(f"Number of windows analyzed: {len(windows)}")

            # Prepare return values
            result = predicted_class

            if return_probabilities:
                result = (result, avg_prediction)

            if return_details:
                details = {
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'probabilities': avg_prediction,
                    'num_windows': len(windows),
                    'individual_predictions': predictions,
                    'class_names': self.label_encoder.classes_
                }
                if return_probabilities:
                    result = (result[0], result[1], details)
                else:
                    result = (result, details)

            return result

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, model_path="pd_model.h5", components_path="pd_components.pkl"):
        """Save trained model and preprocessing components"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")

        # Save model
        self.model.save(model_path)

        # Save preprocessing components
        components = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'label_encoder': self.label_encoder
        }

        with open(components_path, 'wb') as f:
            pickle.dump(components, f)

        print(f"Model saved to: {model_path}")
        print(f"Components saved to: {components_path}")

    def load_model(self, model_path="pd_model.h5", components_path="pd_components.pkl"):
        """Load trained model and preprocessing components"""
        # Load model
        self.model = tf.keras.models.load_model(model_path)

        # Load preprocessing components
        with open(components_path, 'rb') as f:
            components = pickle.load(f)

        self.scaler = components['scaler']
        self.feature_selector = components['feature_selector']
        self.label_encoder = components['label_encoder']
        self.is_trained = True

        print(f"Model loaded from: {model_path}")
        print(f"Components loaded from: {components_path}")


# === MAIN TRAINING CODE (SAME AS BEFORE) ===

def train_model():
    """Train the model and return a trained PDPredictor"""

    # Initialize predictor
    predictor = PDPredictor()

    # --- LOAD CLINICAL DATA ---
    print("Loading clinical data...")
    clinical_df = pd.read_csv(clinical_path)
    clinical_df['ID'] = clinical_df['ID'].astype(str).str.zfill(3)
    label_map = dict(zip(clinical_df['ID'], clinical_df['Status']))
    print("Clinical data loaded.")

    # --- PROCESS FILES WITH QUALITY FILTERING ---
    print("Processing accelerometer files with quality filtering...")
    all_windows = []
    all_labels = []
    folders = sorted(os.listdir(root_dir))

    # Strategy 1: Reduce temporal resolution (downsample)
    downsample_factor = 2  # Reduce from 512Hz to 256Hz
    new_sampling_rate = sampling_rate // downsample_factor
    new_window_size = window_size // downsample_factor

    for folder in tqdm(folders, desc="Patients processed"):
        patient_id = folder.zfill(3)
        accel_path = os.path.join(root_dir, folder, f"lh_ID{patient_id}Accel.csv")

        if not os.path.exists(accel_path):
            continue

        df = pd.read_csv(accel_path)
        if df.shape[0] < window_size:
            continue

        data = df.iloc[:, 1:4].values  # skip timestamp

        # Downsample the data
        data_downsampled = data[::downsample_factor]

        # Strategy 2: Select only informative segments
        # Skip first and last 10% of recording (often contains artifacts)
        start_idx = int(0.1 * len(data_downsampled))
        end_idx = int(0.9 * len(data_downsampled))
        data_clean = data_downsampled[start_idx:end_idx]

        # Strategy 3: Create fewer, non-overlapping windows
        num_windows = min(20, len(data_clean) // new_window_size)  # Limit to 20 windows per patient

        for i in range(num_windows):
            start = i * new_window_size
            if start + new_window_size <= len(data_clean):
                window = data_clean[start:start + new_window_size]
                label = label_map.get(patient_id)

                if label in ['PD', 'Control']:
                    all_windows.append(window)
                    all_labels.append(label)

    print(f"Initial samples: {len(all_windows)}")

    # --- STRATEGY 4: QUALITY-BASED FILTERING ---
    print("Filtering high-quality windows...")
    all_windows, all_labels = filter_high_quality_windows(all_windows, all_labels, quality_threshold=0.6)
    print(f"After quality filtering: {len(all_windows)} samples")

    # --- STRATEGY 5: FEATURE ENGINEERING APPROACH ---
    print("Extracting engineered features...")
    X_features = []
    for window in tqdm(all_windows, desc="Feature extraction"):
        features = extract_features(window)
        X_features.append(features)

    X_features = np.array(X_features)
    print(f"Feature shape: {X_features.shape}")

    # Handle NaN values
    X_features = np.nan_to_num(X_features)

    # --- PREPARE LABELS ---
    y_raw = np.array(all_labels)
    predictor.label_encoder = LabelEncoder()
    y = predictor.label_encoder.fit_transform(y_raw)

    # --- STRATEGY 6: FEATURE SELECTION ---
    print("Performing feature selection...")
    predictor.feature_selector = SelectKBest(f_classif, k=min(50, X_features.shape[1]))
    X_selected = predictor.feature_selector.fit_transform(X_features, y)
    print(f"Selected feature shape: {X_selected.shape}")

    # --- STRATEGY 7: BALANCED SAMPLING ---
    print("Balancing classes...")
    from collections import Counter
    class_counts = Counter(y)
    min_count = min(class_counts.values())

    # Randomly sample equal numbers from each class
    balanced_indices = []
    for class_label in class_counts.keys():
        class_indices = np.where(y == class_label)[0]
        selected_indices = np.random.choice(class_indices, min_count, replace=False)
        balanced_indices.extend(selected_indices)

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    X_balanced = X_selected[balanced_indices]
    y_balanced = y[balanced_indices]

    print(f"Final balanced dataset shape: {X_balanced.shape}")
    print(f"Class distribution: {Counter(y_balanced)}")

    # --- CROSS VALIDATION WITH IMPROVED MODEL ---
    print("\nStarting 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    all_y_true = []
    all_y_pred = []

    # Convert labels to categorical
    y_cat = to_categorical(y_balanced)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_balanced, y_balanced)):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold + 1}/5")
        print(f"{'=' * 50}")

        # Split data
        X_train, X_test = X_balanced[train_idx], X_balanced[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"Train samples: {X_train_scaled.shape[0]}, Test samples: {X_test_scaled.shape[0]}")

        # Create and train model
        model = create_improved_model(X_train_scaled.shape[1:], y_cat.shape[1])

        # Callbacks
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)

        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            verbose=1,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr]
        )

        # Evaluate
        loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Fold {fold + 1} Test Accuracy: {acc:.4f}")

        # Predictions
        y_pred = model.predict(X_test_scaled, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        fold_accuracies.append(acc)
        all_y_true.extend(y_true_labels)
        all_y_pred.extend(y_pred_labels)

        print(f"Classification Report - Fold {fold + 1}:")
        print(classification_report(y_true_labels, y_pred_labels, target_names=predictor.label_encoder.classes_))

    # --- FINAL RESULTS ---
    print(f"\n{'=' * 60}")
    print("FINAL CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")

    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

    print(f"\nOverall Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=predictor.label_encoder.classes_))

    # --- TRAIN FINAL MODEL ON ALL DATA ---
    print(f"\n{'=' * 60}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'=' * 60}")

    # Train scaler and final model on all data
    predictor.scaler = StandardScaler()
    X_final_scaled = predictor.scaler.fit_transform(X_balanced)

    # Train final model
    predictor.model = create_improved_model(X_final_scaled.shape[1:], y_cat.shape[1])

    early_stopping = EarlyStopping(patience=15, restore_best_weights=True, verbose=0)

    predictor.model.fit(
        X_final_scaled, y_cat,
        epochs=150,
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    predictor.is_trained = True

    # --- ESTIMATE DATASET SIZE ---
    estimated_size_mb = (X_balanced.nbytes + y_cat.nbytes) / (1024 * 1024)
    print(f"\nEstimated dataset size: {estimated_size_mb:.2f} MB")

    print(f"\n🎯 MODEL TRAINING COMPLETE!")
    print(f"Expected accuracy: {np.mean(fold_accuracies):.1%} ± {np.std(fold_accuracies):.1%}")
    print(f"Model ready for single patient predictions!")

    return predictor


# === SIMPLE UPLOAD & PREDICT INTERFACE ===

def upload_and_predict(patient_file_path):
    """
    Simple function to upload patient file and get prediction

    Args:
        patient_file_path: Path to patient's accelerometer CSV file

    Returns:
        Prediction result with confidence
    """

    print(f"\n{'=' * 60}")
    print("🏥 PARKINSON'S DISEASE PREDICTION SYSTEM")
    print(f"{'=' * 60}")
    print(f"📁 Processing file: {patient_file_path}")

    try:
        # Check if trained model exists
        if os.path.exists("tremor_model.h5") and os.path.exists("tremor_model.pkl"):
            print("📊 Loading trained model...")
            predictor = PDPredictor()
            predictor.load_model("tremor_model.h5", "tremor_model.pkl")
        else:
            print("❌ No trained model found. Training new model...")
            predictor = train_model()
            predictor.save_model("trained_pd_model.h5", "trained_pd_components.pkl")
            print("✅ Model trained and saved!")

        # Make prediction
        print("🔍 Analyzing patient data...")
        prediction, probabilities, details = predictor.predict_patient(
            patient_file_path,
            return_probabilities=True,
            return_details=True
        )

        # Display results in a nice format
        print(f"\n{'=' * 60}")
        print("📋 PREDICTION RESULTS")
        print(f"{'=' * 60}")

        if prediction == 'PD':
            print("🔴 DIAGNOSIS: PARKINSON'S DISEASE DETECTED")
            risk_level = "HIGH" if probabilities[1] > 0.8 else "MODERATE" if probabilities[1] > 0.6 else "LOW"
            print(f"🎯 Risk Level: {risk_level}")
        else:
            print("🟢 DIAGNOSIS: HEALTHY CONTROL")
            print("🎯 Risk Level: LOW")

        print(f"📊 Confidence: {details['confidence']:.1%}")
        print(f"📈 Probabilities:")
        print(f"   • Control: {probabilities[0]:.1%}")
        print(f"   • Parkinson's: {probabilities[1]:.1%}")
        print(f"🔬 Windows Analyzed: {details['num_windows']}")

        # Clinical interpretation
        print(f"\n{'=' * 60}")
        print("🩺 CLINICAL INTERPRETATION")
        print(f"{'=' * 60}")

        if prediction == 'PD':
            if probabilities[1] > 0.85:
                print("• Strong indicators of Parkinson's disease motor symptoms")
                print("• Recommend immediate neurological consultation")
            elif probabilities[1] > 0.70:
                print("• Moderate indicators of Parkinson's disease motor symptoms")
                print("• Recommend neurological evaluation")
            else:
                print("• Mild indicators detected")
                print("• Consider follow-up assessment")
        else:
            if probabilities[0] > 0.85:
                print("• No significant motor symptoms detected")
                print("• Normal movement patterns observed")
            else:
                print("• Borderline normal patterns")
                print("• Monitor for future changes")

        print("\n⚠️  DISCLAIMER: This is a research tool. Always consult healthcare professionals.")

        return {
            'prediction': prediction,
            'confidence': details['confidence'],
            'probabilities': probabilities,
            'recommendation': 'Consult neurologist' if prediction == 'PD' else 'Continue monitoring'
        }

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        print("💡 Make sure your CSV file has columns: Timestamp, Accel X, Accel Y, Accel Z")
        return None


# === MAIN EXECUTION ===

if __name__ == "__main__":
    print("🏥 Parkinson's Disease Prediction System")
    print("=" * 50)

    # First, train the model (or load if already exists)
    if not (os.path.exists("tremor_model.h5") and os.path.exists("tremor_model.pkl")):
        print("🔄 Training model for the first time...")
        predictor = train_model()
        predictor.save_model("trained_model.h5", "trained_model.pkl")
        print("✅ Model training complete!")
    else:
        print("✅ Using existing trained model")

    print(f"\n{'=' * 70}")
    print("🎯 READY FOR PATIENT PREDICTIONS!")
    print(f"{'=' * 70}")
    print("""
📁 To predict for a new patient, use:

    result = upload_and_predict("path/to/patient_file.csv")

📋 Patient CSV format should be:
    Timestamp (ms), Accel X (g), Accel Y (g), Accel Z (g)
    18202, -0.0663, 0.8838, 0.4815
    18234, -0.0775, 0.8775, 0.4898
    ...

🔍 Example usage:
    """)

    # Example prediction (uncomment and modify path to test)
    # patient_file = "path/to/your/patient/data.csv"
    # result = upload_and_predict(patient_file)

    print("📞 System ready! Call upload_and_predict(file_path) to analyze a patient.")


# === SIMPLE PREDICTION FUNCTION FOR EASY USE ===

def predict_patient_simple(file_path):
    """Ultra-simple function: just give file path, get result"""
    return upload_and_predict(file_path)


upload_and_predict('imu_data_pd.csv')
