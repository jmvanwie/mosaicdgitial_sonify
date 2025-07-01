# app.py - Final Production Version (Corrected)

import os
import uuid
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PyPDF2 import PdfReader
from google.cloud import texttospeech
import google.generativeai as genai

# --- App & CORS Configuration ---
app = Flask(__name__)

# This list tells your backend that it's safe to accept requests
# from these specific web addresses.
origins = [
    "https://vermillion-otter-bfe24a.netlify.app",
    "https://statuesque-tiramisu-4b5936.netlify.app",
    "https://coruscating-hotteok-a5fb56.netlify.app",
    "https://www.mosaicdigital.ai",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    re.compile(r"https://.*\.netlify\.app"), # Allow all netlify subdomains
]
CORS(app, resources={r"/*": {"origins": origins}})

# --- Service Initialization Globals ---
db = None
bucket = None
tts_client = None
genai_model = None

def initialize_services():
    """Initializes all external services using environment variables."""
    global db, bucket, tts_client, genai_model

    if not firebase_admin._apps:
        try:
            print("Attempting to initialize Firebase using Application Default Credentials...")
            firebase_admin.initialize_app() 
            db = firestore.client()
            bucket = storage.bucket(os.environ.get('FIREBASE_STORAGE_BUCKET'))
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
            genai_model = genai.GenerativeModel('gemini-1.5-pro-latest')
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

# --- Core Logic Functions ---
def generate_script_from_idea(topic, context, duration):
    print(f"Generating AI script for topic: {topic}")
    prompt = (f"You are a professional podcast scriptwriter. Your task is to write a compelling and engaging podcast script. "
              f"The script should be approximately {duration} in length. The topic of the podcast is: '{topic}'. "
              f"Here is some additional context: '{context}'. Please provide only the script content, without any "
              f"introductory or concluding remarks about the script itself. Just write the words to be spoken.")
    response = genai_model.generate_content(prompt)
    print("AI script generated successfully.")
    return response.text

def generate_podcast_audio(text_content, output_filepath, voice_names=['en-US-Wavenet-A', 'en-US-Wavenet-B']):
    """
    Generates podcast audio from text content using specified voices.
    Alternates voices for each paragraph.
    """
    print(f"Generating audio with voices: {voice_names}")
    paragraphs = [p.strip() for p in text_content.split('\n') if p.strip()]
    
    ssml_content = '<speak>'
    for i, paragraph in enumerate(paragraphs):
        sanitized_paragraph = paragraph.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        voice_name = voice_names[i % len(voice_names)]
        ssml_content += f'<voice name="{voice_name}">{sanitized_paragraph}</voice>'
    ssml_content += '</speak>'

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_content)
    
    # FIX: Add a default voice parameter. This may be required for the API call to be valid,
    # even though the <voice> tags in the SSML will override it.
    voice_params = texttospeech.VoiceSelectionParams(language_code="en-US")
    
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    response = tts_client.synthesize_speech(
        input=synthesis_input, 
        voice=voice_params, 
        audio_config=audio_config
    )
    
    with open(output_filepath, "wb") as out:
        out.write(response.audio_content)
    
    print(f"Audio content written to file '{output_filepath}'")
    return True

def _finalize_job(job_id, local_audio_path, generated_script=None):
    print(f"Finalizing job {job_id}...")
    storage_path = f"podcasts/{job_id}.mp3"
    blob = bucket.blob(storage_path)
    
    print(f"Uploading {local_audio_path} to {storage_path}...")
    blob.upload_from_filename(local_audio_path)
    blob.make_public()
    podcast_url = blob.public_url
    print(f"Upload complete. Public URL: {podcast_url}")

    os.remove(local_audio_path)
    print(f"Removed temporary file: {local_audio_path}")

    update_data = {'status': 'complete', 'podcast_url': podcast_url, 'completed_at': firestore.SERVER_TIMESTAMP}
    if generated_script:
        update_data['generated_script'] = generated_script

    db.collection('podcasts').document(job_id).update(update_data)
    print(f"Firestore document for job {job_id} updated to complete.")
    return {"status": "Complete", "podcast_url": podcast_url}

# --- Celery Task Definitions ---
@celery.task
def generate_podcast_from_idea_task(job_id, topic, context, duration, voices):
    print(f"WORKER: Started IDEA job {job_id} for topic: {topic} with voices: {voices}")
    doc_ref = db.collection('podcasts').document(job_id)
    output_filepath = f"{job_id}.mp3"
    try:
        doc_ref.set({'topic': topic, 'context': context, 'source_type': 'idea', 'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP, 'voices': voices})
        podcast_script = generate_script_from_idea(topic, context, duration)
        if not podcast_script: raise Exception("Script generation failed.")
        
        if not generate_podcast_audio(podcast_script, output_filepath, voices): 
            raise Exception("Audio generation failed.")
            
        return _finalize_job(job_id, output_filepath, generated_script=podcast_script)
    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        if os.path.exists(output_filepath): os.remove(output_filepath)
        return {"status": "Failed", "error": str(e)}

# --- API Endpoints ---
@app.before_request
def before_first_request_func():
    initialize_services()

# ADD THIS NEW ROUTE FOR DIAGNOSTICS
@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Sonify API! The server is running."})

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context']):
        return jsonify({"error": "topic and context are required"}), 400
    
    job_id = str(uuid.uuid4())
    
    # Use Studio voices as the default, but allow frontend to override
    voices = data.get('voices', ['en-US-Studio-M', 'en-US-Studio-Q'])
    
    generate_podcast_from_idea_task.delay(
        job_id, 
        data['topic'], 
        data['context'], 
        data.get('duration', '5 minutes'),
        voices
    )
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
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
