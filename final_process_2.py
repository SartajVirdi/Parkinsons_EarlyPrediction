import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from voice_extract import extract_features as extract_voice_features
from reduce_pd import PDPredictor

# === Load Models and Preprocessors ===
voice_model = load_model("voice_model.h5")
voice_scaler = joblib.load("scaler.pkl")

imu_predictor = PDPredictor()
imu_predictor.load_model("tremor_model.h5", "tremor_model.pkl")

# === Voice Prediction Function ===
def predict_from_voice(audio_file):
    features = extract_voice_features(audio_file)
    if features is None:
        raise ValueError("Could not extract features from audio")

    expected_columns = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ]

    row = {col: features.get(col, 0) for col in expected_columns}
    df = pd.DataFrame([row])

    X_scaled = voice_scaler.transform(df)
    X_cnn = X_scaled.reshape(1, X_scaled.shape[1], 1)

    voice_prob = voice_model.predict(X_cnn, verbose=0)[0][0]
    return voice_prob

# === IMU Prediction Function ===
def predict_from_imu(imu_file):
    prediction, probs, _ = imu_predictor.predict_patient(
        imu_file, return_probabilities=True, return_details=True
    )
    imu_prob = probs[1]  # Parkinson's probability
    return imu_prob

# === Late Fusion Predictor ===
def combined_prediction(voice_file, imu_file, weight_voice=0.519, weight_imu=0.481):
    voice_prob = predict_from_voice(voice_file)
    imu_prob = predict_from_imu(imu_file)

    combined_prob = weight_voice * voice_prob + weight_imu * imu_prob
    final_pred = "Parkinson's" if combined_prob > 0.5 else "Healthy"

    print("\n=== Combined Prediction ===")
    print(f"Combined Prob (PD):    {combined_prob:.2%}")
    print(f"Final Diagnosis:       {final_pred}")

    return {
        "combined_prob": combined_prob,
        "prediction": final_pred
    }

# === Run the Prediction ===
result = combined_prediction("test_audio1.wav", "imu_data_pd.csv")
