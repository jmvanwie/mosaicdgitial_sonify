# app.py - Production-Ready Version

import os
import uuid
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
    "https://statuesque-tiramisu-4b5936.netlify.app", # Your NEW Netlify frontend
    "https://www.mosaicdigital.ai",                     # Your main Wix domain
    "http://localhost:8000",                            # For local testing
    "http://127.0.0.1:5500"                            # For local testing with VS Code Live Server
]
CORS(app, resources={r"/*": {"origins": origins}})

# --- Firebase & Google AI Initialization ---
db = None
bucket = None
tts_client = None
genai_model = None

def initialize_services():
    """Initializes all external services using environment variables."""
    global db, bucket, tts_client, genai_model

    # Initialize Firebase if not already done
    if not firebase_admin._apps:
        try:
            print("Attempting to initialize Firebase...")
            # Render provides the service account JSON via a secret file path
            # The path is set in the GOOGLE_APPLICATION_CREDENTIALS env var
            # which firebase-admin reads automatically. If not set, we can specify.
            cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'firebase_service_account.json')
            
            # Check if the file exists before trying to use it
            if not os.path.exists(cred_path):
                 raise FileNotFoundError(f"Service account key not found at path: {cred_path}. Ensure the secret file is correctly configured in Render.")

            cred = credentials.Certificate(cred_path)
            
            # Get Storage Bucket from environment variable
            storage_bucket_url = os.environ.get('FIREBASE_STORAGE_BUCKET')
            if not storage_bucket_url:
                raise ValueError("FIREBASE_STORAGE_BUCKET environment variable not set.")

            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket_url
            })
            db = firestore.client()
            bucket = storage.bucket()
            print("Successfully connected to Firebase.")
        except Exception as e:
            print(f"FATAL: Could not connect to Firebase: {e}")
            raise e

    # Initialize Google Cloud Text-to-Speech Client
    if tts_client is None:
        try:
            print("Initializing Google Cloud Text-to-Speech client...")
            tts_client = texttospeech.TextToSpeechClient()
            print("Text-to-Speech client initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize TTS client: {e}")
            raise e
            
    # Initialize Google Gemini Model
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
    """Celery factory that reads configuration from environment variables."""
    broker_url = os.environ.get('CELERY_BROKER_URL')
    if not broker_url:
        raise RuntimeError("CELERY_BROKER_URL environment variable is not set.")

    celery = Celery(
        app.import_name,
        backend=broker_url,
        broker=broker_url
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                # Initialize services within the task context
                initialize_services()
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# --- Core Logic Functions ---

def generate_script_from_idea(topic, context, duration):
    """Generates a podcast script using the configured Google Gemini API model."""
    print(f"Generating AI script for topic: {topic}")
    try:
        prompt = (
            f"You are a professional podcast scriptwriter. Your task is to write a compelling and engaging podcast script. "
            f"The script should be approximately {duration} in length. "
            f"The topic of the podcast is: '{topic}'. "
            f"Here is some additional context: '{context}'. "
            f"Please provide only the script content, without any introductory or concluding remarks about the script itself. "
            f"Just write the words to be spoken."
        )
        response = genai_model.generate_content(prompt)
        print("AI script generated successfully.")
        return response.text
    except Exception as e:
        print(f"Error during Gemini script generation: {e}")
        return None

def generate_podcast_audio(text_content, output_filepath):
    """Generates an audio file from text using the configured Google Cloud TTS client."""
    print(f"Generating audio for text (first 100 chars): {text_content[:100]}...")
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text_content)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(output_filepath, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content written to file '{output_filepath}'")
        return True
    except Exception as e:
        print(f"Error during Text-to-Speech generation: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Opens a PDF file and extracts all text content."""
    print(f"Extracting text from {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        print("Text extraction from PDF successful.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def _finalize_job(job_id, local_audio_path, generated_script=None):
    """Helper function to upload audio, update Firestore, and clean up."""
    print(f"Finalizing job {job_id}...")
    storage_path = f"podcasts/{job_id}.mp3"
    blob = bucket.blob(storage_path)
    
    print(f"Uploading {local_audio_path} to {storage_path}...")
    blob.upload_from_filename(local_audio_path)
    blob.make_public()
    podcast_url = blob.public_url
    print(f"Upload complete. Public URL: {podcast_url}")

    # Clean up the local temporary file
    os.remove(local_audio_path)
    print(f"Removed temporary file: {local_audio_path}")

    update_data = {
        'status': 'complete',
        'podcast_url': podcast_url,
        'completed_at': firestore.SERVER_TIMESTAMP
    }
    if generated_script:
        update_data['generated_script'] = generated_script

    doc_ref = db.collection('podcasts').document(job_id)
    doc_ref.update(update_data)
    print(f"Firestore document for job {job_id} updated to complete.")
    return {"status": "Complete", "podcast_url": podcast_url}

# --- Celery Task Definitions ---

@celery.task
def generate_podcast_from_idea_task(job_id, topic, context, duration):
    """Background job for generating a podcast from an idea."""
    print(f"WORKER: Started IDEA job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    output_filepath = f"{job_id}.mp3" # Files are created in the ephemeral root

    try:
        doc_ref.set({
            'topic': topic, 'context': context, 'source_type': 'idea',
            'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP
        })

        podcast_script = generate_script_from_idea(topic, context, duration)
        if not podcast_script:
            raise Exception("Failed to generate podcast script.")

        if not generate_podcast_audio(podcast_script, output_filepath):
            raise Exception("Failed to generate audio file.")
            
        return _finalize_job(job_id, output_filepath, generated_script=podcast_script)

    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        return {"status": "Failed", "error": str(e)}

@celery.task
def generate_podcast_from_pdf_task(job_id, temp_pdf_path, original_filename, duration):
    """Background job for processing a PDF file."""
    print(f"WORKER: Started PDF job {job_id} for file: {original_filename}")
    doc_ref = db.collection('podcasts').document(job_id)
    output_filepath = f"{job_id}.mp3"

    try:
        doc_ref.set({
            'original_filename': original_filename, 'source_type': 'pdf',
            'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP
        })

        extracted_text = extract_text_from_pdf(temp_pdf_path)
        os.remove(temp_pdf_path) # Clean up uploaded PDF immediately
        print(f"Task {job_id}: Extracted {len(extracted_text)} characters.")
        
        if not extracted_text:
             raise Exception("PDF contained no extractable text.")

        if not generate_podcast_audio(extracted_text, output_filepath):
            raise Exception("Failed to generate audio file.")
            
        return _finalize_job(job_id, output_filepath)

    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        # Ensure temp PDF is cleaned up on failure too
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return {"status": "Failed", "error": str(e)}

# --- API Endpoints ---

# This hook ensures services are ready before the first request is handled
@app.before_request
def before_first_request_func():
    initialize_services()

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context', 'duration']):
        return jsonify({"error": "topic, context, and duration are required"}), 400
    
    job_id = str(uuid.uuid4())
    generate_podcast_from_idea_task.delay(
        job_id, data['topic'], data['context'], data['duration']
    )
    print(f"API: Queued IDEA job {job_id}")
    return jsonify({"message": "Podcast generation from idea has been queued!", "job_id": job_id}), 202

@app.route("/generate-from-pdf", methods=["POST"])
def handle_pdf_generation():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "A selected PDF file is required"}), 400
    
    job_id = str(uuid.uuid4())
    # Save file temporarily to the ephemeral filesystem root
    temp_filepath = f"{job_id}.pdf"
    file.save(temp_filepath)
    
    generate_podcast_from_pdf_task.delay(
        job_id, temp_filepath, file.filename, request.form.get('duration', 'not specified')
    )
    print(f"API: Queued PDF job {job_id}")
    return jsonify({"message": "Podcast generation from PDF has been queued!", "job_id": job_id}), 202

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
    # This block is for local development only. 
    # Gunicorn will be used in production on Render.
    app.run(debug=True, port=5000)