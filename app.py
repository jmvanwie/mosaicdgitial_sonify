# app.py - Final Production Version (Corrected with Robust Audio Generation and Script Cleaning)

import os
import uuid
import re
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from pydub import AudioSegment
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
def clean_script_for_tts(script_text):
    """
    Removes non-dialogue elements from a generated script.
    """
    print("Cleaning script for Text-to-Speech...")
    # Remove text in parentheses, e.g., (Sound of...)
    cleaned_text = re.sub(r'\(.*?\)', '', script_text)
    # Remove text in asterisks, e.g., **Sound of...**
    cleaned_text = re.sub(r'\*\*.*?\*\*', '', cleaned_text)
    # Remove speaker labels, e.g., "AI Voice 1:", "AI Voices 1 & 2:"
    cleaned_text = re.sub(r'AI Voices?(\s\d\s?(&\s?\d)?)?:\s?', '', cleaned_text)
    # Remove any leading/trailing whitespace from each line
    cleaned_text = "\n".join([line.strip() for line in cleaned_text.split('\n')])
    print("Script cleaned.")
    return cleaned_text

def generate_script_from_idea(topic, context, duration):
    print(f"Generating AI script for topic: {topic}")
    # IMPROVED PROMPT: More explicit instructions to get cleaner output.
    prompt = (f"You are a professional podcast scriptwriter. Your task is to write a compelling and engaging podcast script "
              f"for two AI voices. The script should be approximately {duration} in length. The topic of the podcast is: '{topic}'. "
              f"Here is some additional context: '{context}'. "
              f"IMPORTANT: Provide ONLY the dialogue to be spoken. Do NOT include any speaker labels (like 'AI Voice 1:'). "
              f"Do NOT include any sound effect descriptions in parentheses or asterisks. Each speaker's part should be on a new line.")
    response = genai_model.generate_content(prompt)
    print("AI script generated successfully.")
    return response.text

def generate_podcast_audio(text_content, output_filepath, voice_names=['en-US-Wavenet-A', 'en-US-Wavenet-B']):
    """
    Generates podcast audio by synthesizing each paragraph individually and
    stitching them together. This is a robust method that avoids complex SSML issues.
    """
    print(f"Generating audio in chunks for voices: {voice_names}")
    # Filter out empty lines that might result from the cleaning process
    paragraphs = [p.strip() for p in text_content.split('\n') if p.strip()]
    
    if not paragraphs:
        raise ValueError("The script is empty after cleaning. Cannot generate audio.")

    combined_audio = AudioSegment.empty()
    
    for i, paragraph in enumerate(paragraphs):
        voice_name = voice_names[i % len(voice_names)]
        print(f"Synthesizing paragraph {i+1}/{len(paragraphs)} with voice {voice_name}...")
        
        synthesis_input = texttospeech.SynthesisInput(text=paragraph)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config
        )
        
        audio_chunk = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
        combined_audio += audio_chunk

    print(f"Exporting combined audio to {output_filepath}...")
    combined_audio.export(output_filepath, format="mp3")
    
    print(f"Audio content successfully written to file '{output_filepath}'")
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
        # Generate the original script
        original_script = generate_script_from_idea(topic, context, duration)
        if not original_script: raise Exception("Script generation failed.")
        
        # Clean the script before audio generation
        cleaned_script = clean_script_for_tts(original_script)
        
        # Generate audio from the cleaned script
        if not generate_podcast_audio(cleaned_script, output_filepath, voices): 
            raise Exception("Audio generation failed.")
            
        # Finalize the job, saving the ORIGINAL script for user reference
        return _finalize_job(job_id, output_filepath, generated_script=original_script)
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
