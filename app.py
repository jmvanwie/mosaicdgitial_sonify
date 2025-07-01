# app.py - FINAL VERSION with manual preflight handling

import os
import uuid
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud import texttospeech
import google.generativeai as genai

app = Flask(__name__)

# This is the correct CORS configuration to allow requests from your Netlify site.
origins = [
    "https://statuesque-tiramisu-4b5936.netlify.app",
    "https://www.mosaicdigital.ai"
]
# We apply CORS globally but will handle the preflight manually for robustness
CORS(app, origins=origins)

# --- Service Initialization ---
_services_initialized = False
def initialize_all_services():
    global _services_initialized, db, bucket, tts_client, genai_model
    if _services_initialized:
        return
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    
    db = firestore.client()
    bucket = storage.bucket(os.environ.get('FIREBASE_STORAGE_BUCKET'))
    tts_client = texttospeech.TextToSpeechClient()
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
    genai_model = genai.GenerativeModel('gemini-1.5-pro-latest')
    _services_initialized = True
    print("All services initialized successfully.")

# --- Celery Configuration ---
celery = Celery(app.import_name, 
                backend=os.environ.get('CELERY_BROKER_URL'), 
                broker=os.environ.get('CELERY_BROKER_URL'))
celery.conf.update(app.config)

class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            initialize_all_services()
            return self.run(*args, **kwargs)
celery.Task = ContextTask

# --- Helper for Manual Preflight Response ---
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

# --- API Endpoints ---
@app.route("/generate-from-idea", methods=["POST", "OPTIONS"])
def handle_idea_generation():
    # This is the critical change to manually handle the preflight request
    if request.method == "OPTIONS": 
        return _build_cors_preflight_response()

    # This is the existing logic for the POST request
    data = request.get_json()
    if not data or 'topic' not in data or 'context' not in data:
        return jsonify({"error": "Missing topic or context"}), 400
    
    job_id = str(uuid.uuid4())
    generate_podcast_task.delay(
        job_id, 
        data['topic'], 
        data['context'], 
        data.get('duration', '5 minutes')
    )
    return jsonify({"message": "Podcast generation has been queued!", "job_id": job_id}), 202

@app.route("/podcast-status/<job_id>", methods=["GET"])
def get_podcast_status(job_id):
    try:
        doc = db.collection('podcasts').document(job_id).get()
        if not doc.exists:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(doc.to_dict()), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- Celery Task (Simple, single-speaker) ---
@celery.task
def generate_podcast_task(job_id, topic, context, duration):
    print(f"WORKER: Started SIMPLE job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    try:
        doc_ref.set({'topic': topic, 'context': context, 'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP})
        
        prompt = (f"You are a professional podcast scriptwriter. Write a compelling and engaging monologue podcast script. "
                  f"The script should be approximately {duration} in length. The topic is: '{topic}'. "
                  f"Additional context: '{context}'. Provide only the spoken words for the script.")
        
        response = genai_model.generate_content(prompt)
        podcast_script = response.text
        if not podcast_script: raise Exception("Script generation failed.")

        synthesis_input = texttospeech.SynthesisInput(text=podcast_script)
        voice = texttospeech.VoiceSelectionParams(language_code='en-US', name='en-US-WaveNet-J')
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        audio_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        
        storage_path = f"podcasts/{job_id}.mp3"
        blob = bucket.blob(storage_path)
        blob.upload_from_string(audio_response.audio_content, content_type='audio/mpeg')
        blob.make_public()
        
        doc_ref.update({'status': 'complete', 'podcast_url': blob.public_url, 'completed_at': firestore.SERVER_TIMESTAMP, 'generated_script': podcast_script})
        
        return {"status": "Complete", "podcast_url": blob.public_url}
        
    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        return {"status": "Failed", "error": str(e)}

if __name__ == '__main__':
    app.run(debug=True)