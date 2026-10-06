import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import preprocessing

# 1. Konfigurasi Token DagsHub Pakai os.environ
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/Zulfan-Afandi/Proyek-Akhir-SMSML.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Zulfan-Afandi'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'b11d9ca55b58f011e48c7d3fbd5be22f011f9591'

# 2. Set Tracking URI agar mengarah ke DagsHub
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment('Mobile_Legends_M5_Classification')

def train_and_track_model(file_path):
    print("Memulai proses training...")
    
    # A. Preprocessing
    df_raw = preprocessing.load_data(file_path)
    df_clean = preprocessing.clean_data(df_raw)
    df_transformed, encoders = preprocessing.transform_data(df_clean)
    X_train, X_test, y_train, y_test = preprocessing.split_dataset(df_transformed)

    # B. Mulai tracking MLflow
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        
        # PERBAIKAN DI SINI: Menggunakan format cloudpickle agar tidak diblokir skops
        mlflow.sklearn.log_model(
            model, 
            "random_forest_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        print(f"Model berhasil dilatih dan dikirim ke DagsHub! Akurasi: {acc:.2f}")

if __name__ == "__main__":
    file_path = '/content/Dataset_ML_TugasDicoding/Full Match Results Data.csv'
    train_and_track_model(file_path)
