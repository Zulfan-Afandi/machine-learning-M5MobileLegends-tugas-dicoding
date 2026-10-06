from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from prometheus_client import make_asgi_app, Counter, Histogram

# Inisiasi aplikasi FastAPI
app = FastAPI(title="Mobile Legends M5 Prediction API")

# Metrik Prometheus (Kriteria 4)
REQUEST_COUNTER = Counter("api_requests_total", "Total requests received by the API")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of predictions")

# Tambahkan endpoint metrik Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Skema Input Data JSON (Disesuaikan dengan fitur dari preprocessing)
class MatchData(BaseModel):
    team: str
    ban: str
    role_ban: str
    pick: str
    role_pick: str

# Fungsi untuk memuat model (nanti akan disesuaikan path-nya di server/lokal)
model = None
try:
    # Menggunakan model dari path lokal MLflow (contoh fallback)
    model = mlflow.pyfunc.load_model("mlruns/0/random_forest_model")
except:
    pass

@app.get("/")
def home():
    return {"message": "Mobile Legends Prediction API is Running!"}

@app.post("/predict")
@PREDICTION_LATENCY.time() # Menghitung latensi untuk Prometheus
def predict(data: MatchData):
    REQUEST_COUNTER.inc() # Menambah counter request
    
    if not model:
        return {"error": "Model belum dimuat. Periksa direktori model."}
    
    # Konversi input JSON ke DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Lakukan prediksi
    pred = model.predict(df)
    
    return {
        "prediksi_mvp": int(pred[0]),
        "status": "success"
    }
