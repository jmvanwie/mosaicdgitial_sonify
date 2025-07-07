# app.py

import os
from flask import Flask, jsonify
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud import texttospeech
import google.generativeai as genai
from google.cloud import aiplatform

# --- App & CORS Configuration ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Simplified for multi-service setup

# --- Global Service Clients (will be initialized within app context) ---
db = None
bucket = None
tts_client = None
genai_model = None
aip_client = None # <-- For Vertex AI
pixabay_api_key = None # <-- For Pixabay
pexels_api_key = None # <-- For Pexels

# --- Service Initialization ---
def initialize_services():
    """Initializes all external services. Called within app context."""
    global db, bucket, tts_client, genai_model, aip_client, pixabay_api_key, pexels_api_key

    if not firebase_admin._apps:
        print("Initializing Firebase...")
        firebase_admin.initialize_app()
        db = firestore.client()
        bucket = storage.bucket(os.environ.get('FIREBASE_STORAGE_BUCKET'))
        print("Firebase Initialized.")

    if tts_client is None:
        print("Initializing Google TTS...")
        tts_client = texttospeech.TextToSpeechClient()
        print("Google TTS Initialized.")

    if genai_model is None:
        print("Initializing Google Gemini...")
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=gemini_api_key)
        genai_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        print("Google Gemini Initialized.")

    if aip_client is None:
        try:
            print("Initializing Google Cloud AI Platform (Vertex AI)...")
            project_id = os.environ.get('GCP_PROJECT_ID')
            location = 'us-central1' # Or your preferred location
            aiplatform.init(project=project_id, location=location)
            aip_client = True # Set a flag to show it's initialized
            print("AI Platform client initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize AI Platform client: {e}")
            raise e
        
    if pixabay_api_key is None:
        pixabay_api_key = os.environ.get('PIXABAY_API_KEY') # Use your variable name
        if pixabay_api_key:
             print("Pixabay API key loaded.")
        else:
             print("WARNING: PIXABAY_API_KEY not set.")

    if pexels_api_key is None:
        pexels_api_key = os.environ.get('PEXELS_API_KEY') # Use your variable name
        if pexels_api_key:
             print("Pexels API key loaded.")
        else:
             print("WARNING: PEXELS_API_KEY not set.")

# --- Celery Configuration ---
def make_celery(app):
    broker_url = os.environ.get('CELERY_BROKER_URL')
    if not broker_url:
        raise RuntimeError("CELERY_BROKER_URL is not set.")

    celery = Celery(app.import_name, backend=broker_url, broker=broker_url)
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                # Initialize services for every worker task
                initialize_services()
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# --- Register Blueprints ---
# Import the blueprints from your route files
from sonify_routes import sonify_bp
from visify_routes import visify_bp

app.register_blueprint(sonify_bp)
app.register_blueprint(visify_bp)


# --- Root Endpoint ---
@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Mosaic Digital Content API!"})

if __name__ == '__main__':
    # This is for local development only.
    # On Render, Gunicorn will be used to run the app.
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))