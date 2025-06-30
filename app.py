# app.py - DEBUGGING VERSION

import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS  # Make sure this is imported
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PyPDF2 import PdfReader
from google.cloud import texttospeech
import google.generativeai as genai

# --- App & CORS Configuration ---
app = Flask(__name__)

# --- TEMPORARY DEBUGGING STEP ---
# This line allows requests from ANY origin. It's not secure for long-term
# production, but it will definitively solve any CORS error for our test.
CORS(app)

# --- Firebase & Google AI Initialization ---
db = None
bucket = None
tts_client = None
genai_model = None

def initialize_services():
    """Initializes all external services using environment variables."""
    global db, bucket, tts_client, genai_model

    if not firebase_admin._apps:
        try:
            print("Attempting to initialize Firebase...")
            cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'firebase_service_account.json')
            if not os.path.exists(cred_path):
                 raise FileNotFoundError(f"Service account key not found at path: {cred_path}. Ensure the secret file is correctly configured in Render.")

            cred = credentials.Certificate(cred_path)
            storage_bucket_url = os.environ.get('FIREBASE_STORAGE_BUCKET')
            if not storage_bucket_url:
                raise ValueError("FIREBASE_STORAGE_BUCKET environment variable not set.")

            firebase_admin.initialize_app(cred, {'storageBucket': storage_bucket_url})
            db = firestore.client()
            bucket = storage.bucket()
            print("Successfully connected to Firebase.")
        except Exception as e:
            print(f"FATAL: Could not connect to Firebase: {e}")
            raise e

    if tts_client is None:
        try:
            print("Initializing Google Cloud Text-to-Speech client...")
            tts_client = texttospeech.TextToSpeechClient()
            print("Text-to-Speech client initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize TTS client: {e}")
            raise e
            
    if genai_model is None:
        try:
            print("Initializing Google Gemini model...")
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=gemini_api_key)
            genai_model = genai.GenerativeModel('gemini-pro')
            print("Gemini model initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize Gemini model: {e}")
            raise e

# --- Celery Configuration ---
def make_celery(app):
    broker_url = os.environ.get('CELERY_BROKER_URL')
    if not broker_url:
        raise RuntimeError("CELERY_BROKER_URL environment variable is not set.")

    celery = Celery(app.import_name, backend=broker_url, broker=broker_url)
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                initialize_services()
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# --- Core Logic Functions (abbreviated for clarity, no changes needed here) ---
def generate_script_from_idea(topic, context, duration):
    # (Existing function code)
    prompt = (f"You are a podcast scriptwriter. Write a script about {topic} with the context: {context}.")
    response = genai_model.generate_content(prompt)
    return response.text

def generate_podcast_audio(text_content, output_filepath):
    # (Existing function code)
    synthesis_input = texttospeech.SynthesisInput(text=text_content)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_filepath, "wb") as out:
        out.write(response.audio_content)
    return True

def _finalize_job(job_id, local_audio_path, generated_script=None):
    # (Existing function code)
    storage_path = f"podcasts/{job_id}.mp3"
    blob = bucket.blob(storage_path)
    blob.upload_from_filename(local_audio_path)
    blob.make_public()
    podcast_url = blob.public_url
    os.remove(local_audio_path)
    update_data = {'status': 'complete', 'podcast_url': podcast_url, 'completed_at': firestore.SERVER_TIMESTAMP}
    if generated_script:
        update_data['generated_script'] = generated_script
    db.collection('podcasts').document(job_id).update(update_data)
    return {"status": "Complete", "podcast_url": podcast_url}

# --- Celery Task Definitions (abbreviated for clarity, no changes needed here) ---
@celery.task
def generate_podcast_from_idea_task(job_id, topic, context, duration):
    doc_ref = db.collection('podcasts').document(job_id)
    output_filepath = f"{job_id}.mp3"
    try:
        doc_ref.set({'topic': topic, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP})
        podcast_script = generate_script_from_idea(topic, context, duration)
        if not podcast_script: raise Exception("Script generation failed.")
        if not generate_podcast_audio(podcast_script, output_filepath): raise Exception("Audio generation failed.")
        return _finalize_job(job_id, output_filepath, generated_script=podcast_script)
    except Exception as e:
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        if os.path.exists(output_filepath): os.remove(output_filepath)
        return {"status": "Failed", "error": str(e)}

# --- API Endpoints ---
@app.before_request
def before_first_request_func():
    initialize_services()

# --- NEW DEBUGGING ENDPOINT ---
@app.route("/version")
def get_version():
    """A simple endpoint to verify which version of the code is live."""
    return jsonify({"version": "1.3 - Final Debug"})

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context']):
        return jsonify({"error": "topic and context are required"}), 400
    
    job_id = str(uuid.uuid4())
    generate_podcast_from_idea_task.delay(
        job_id, data['topic'], data['context'], data.get('duration', '5 minutes')
    )
    print(f"API: Queued IDEA job {job_id}")
    return jsonify({"message": "Podcast generation from idea has been queued!", "job_id": job_id}), 202

@app.route("/podcast-status/<job_id>", methods=["GET"])
def get_podcast_status(job_id):
    try:
        doc_ref = db.collection('podcasts').document(job_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(doc.to_dict()), 200
    except Exception as e:
        print(f"Error getting status for job {job_id}: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)