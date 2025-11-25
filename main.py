# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json, os, traceback
import subprocess
import sys
import os

# Import the engine and model functions
# Ensure these modules are on PYTHONPATH or in the same package
from eh_menu_engine import Meal, MenuGeneratorEngine
import ml_popularity

# Path constants
MEALS_JSON_PATH = "data/meals_data.json"
MODEL_PATH = "data/popularity_model.joblib"

app = FastAPI(title="EH Meals Production API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global in-memory store
meals_store: List[Meal] = []

def load_meals():
    global meals_store
    if not os.path.exists(MEALS_JSON_PATH):
        meals_store = []
        return
    with open(MEALS_JSON_PATH, "r") as fh:
        raw = json.load(fh)
    # convert dicts to Meal objects
    meals_store = [Meal(**m) for m in raw]

@app.on_event("startup")
def startup_event():
    load_meals()

@app.get("/", tags=["system"])
def root():
    return {"message": "EH Meals API", "endpoints": {"/api/generate-menu": "POST", "/api/refresh-popularity-model": "POST", "/api/predict-popularity": "GET"}}

@app.get("/health", tags=["system"])
def health():
    return {"status": "healthy", "meals_loaded": len(meals_store), "popularity_model_present": os.path.exists(MODEL_PATH)}

@app.get("/api/meals", tags=["meals"])
def get_meals():
    if not meals_store:
        raise HTTPException(status_code=500, detail="Meals not loaded.")
    return meals_store

@app.post("/run-script")
def run_script():

    script_path = os.path.join(os.getcwd(), "Scripts", "convert_api_data.py")

    try:
        # run: python scripts/my_task.py
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )

        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    

@app.post("/api/refresh-popularity-model", tags=["ml"])
def refresh_popularity_model():
    """
    Retrain the popularity model from current /mnt/data/meals_data.json
    """
    try:
        res = ml_popularity.train_and_persist_model()
        return {"success": True, "detail": res}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/predict-popularity", tags=["ml"])
def predict_popularity():
    """
    Return normalized predicted popularity scores for all meals as {uuid: score}
    """
    if not meals_store:
        load_meals()
    try:
        # raw dicts required by ml_popularity.predict_popularity_scores
        raw = [m.dict() for m in meals_store]
        preds = ml_popularity.predict_popularity_scores(raw)
        return {"success": True, "predictions": preds}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Popularity model not trained. Call /api/refresh-popularity-model")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    

@app.post("/api/generate-menu", tags=["menu"])
def generate_menu(body: dict):
    """
    Request body should be:
    {
        "min_meals": 50,
        "max_meals": 70,
        "dietary_balance": true,
        "include_add_ons": false
    }
    """
    if not meals_store:
        load_meals()
    try:
        min_meals = int(body.get("min_meals", 50))
        max_meals = int(body.get("max_meals", 70))
        dietary_balance = bool(body.get("dietary_balance", True))
        include_add_ons = bool(body.get("include_add_ons", False))

        # Load predictions if available, else empty dict
        pred_map = {}
        try:
            raw = [m.dict() for m in meals_store]
            pred_map = ml_popularity.predict_popularity_scores(raw)
        except Exception:
            pred_map = {}

        engine = MenuGeneratorEngine(meals_store)
        selected = engine.generate_menu(min_meals=min_meals, max_meals=max_meals, dietary_balance=dietary_balance, include_add_ons=include_add_ons, popularity_scores=pred_map)
        stats = engine.get_menu_statistics(selected)
        stats['predicted_popularity_count'] = len(pred_map)

        # Return meals as dicts (serializable)
        return {"success": True, "total_meals": len(selected), "meals": [m.dict() for m in selected], "statistics": stats}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Menu generation failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
