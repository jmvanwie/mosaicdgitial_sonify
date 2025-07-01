# app.py - Final Production Version with Multi-Speaker Support

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
from pydub import AudioSegment

# --- App & CORS Configuration ---
app = Flask(__name__)

origins = [
    "https://statuesque-tiramisu-4b5936.netlify.app",
    "https://www.mosaicdigital.ai",
    "http://localhost:8000",
    "http://127.0.0.1:5500"
]
CORS(app, origins=["https://statuesque-tiramisu-4b5936.netlify.app", "https://www.mosaicdigital.ai"])

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

def generate_script(topic, context, duration, num_speakers):
    """Generates a script from Gemini, formatted for one or two speakers."""
    print(f"Generating AI script for {num_speakers} speaker(s) on topic: {topic}")
    
    if int(num_speakers) == 1:
        prompt = (f"You are a professional podcast scriptwriter. Write a compelling and engaging monologue podcast script. "
                  f"The script should be approximately {duration} in length. The topic is: '{topic}'. "
                  f"Additional context: '{context}'. Provide only the spoken words for the script.")
    else: # num_speakers == 2
        prompt = (f"You are a professional podcast scriptwriter. Write a compelling and engaging dialogue-based podcast script for two speakers. "
                  f"The script should be approximately {duration} in length. The topic is: '{topic}'. "
                  f"Additional context: '{context}'. "
                  f"IMPORTANT: You MUST format the script by clearly labeling each speaker's lines. Use '[SPEAKER 1]' and '[SPEAKER 2]' to denote who is speaking. For example: "
                  f"'[SPEAKER 1] Hello and welcome to the show. [SPEAKER 2] It's great to be here.' "
                  f"Do not include any other text, just the formatted script.")
                  
    response = genai_model.generate_content(prompt)
    print("AI script generated successfully.")
    return response.text

def generate_multispeaker_audio(script_text, voice1, voice2, job_id):
    """Parses a 2-speaker script and generates a combined audio file."""
    print("Starting multi-speaker audio generation...")
    lines = re.split(r'(\[SPEAKER \d\])', script_text)
    
    combined_audio = AudioSegment.empty()
    temp_files = []
    
    # Process lines in chunks of two (marker + text)
    for i in range(1, len(lines), 2):
        speaker_tag = lines[i]
        text_content = lines[i+1].strip()
        
        if not text_content:
            continue

        voice_name = voice1 if '1' in speaker_tag else voice2
        
        # Generate a unique filename for each snippet
        snippet_filename = f"{job_id}_snippet_{i}.mp3"
        temp_files.append(snippet_filename)
        
        print(f"Generating audio for {speaker_tag} with voice {voice_name}")
        synthesis_input = texttospeech.SynthesisInput(text=text_content)
        voice = texttospeech.VoiceSelectionParams(language_code=voice_name.split('-W')[0], name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        
        with open(snippet_filename, "wb") as out:
            out.write(response.audio_content)
            
        # Append the new snippet to the combined audio
        snippet_audio = AudioSegment.from_mp3(snippet_filename)
        combined_audio += snippet_audio

    final_filename = f"{job_id}.mp3"
    print(f"Exporting combined audio to {final_filename}")
    combined_audio.export(final_filename, format="mp3")

    # Clean up temporary snippet files
    for f in temp_files:
        os.remove(f)
    print("Cleaned up temporary audio snippets.")
    
    return final_filename

def generate_singlespeaker_audio(script_text, voice_name, output_filepath):
    """Generates a standard single-speaker audio file."""
    print(f"Generating single-speaker audio with voice {voice_name}...")
    synthesis_input = texttospeech.SynthesisInput(text=script_text)
    voice = texttospeech.VoiceSelectionParams(language_code=voice_name.split('-W')[0], name=voice_name)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    
    with open(output_filepath, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio content written to file '{output_filepath}'")
    return output_filepath


def _finalize_job(job_id, local_audio_path, generated_script):
    """Uploads the final MP3 and updates Firestore."""
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

    update_data = {
        'status': 'complete', 
        'podcast_url': podcast_url, 
        'completed_at': firestore.SERVER_TIMESTAMP,
        'generated_script': generated_script
    }
    db.collection('podcasts').document(job_id).update(update_data)
    print(f"Firestore document for job {job_id} updated to complete.")
    return {"status": "Complete", "podcast_url": podcast_url}

# --- Celery Task Definition ---
@celery.task
def generate_podcast_task(job_id, topic, context, duration, num_speakers, voice1, voice2):
    print(f"WORKER: Started job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    
    try:
        # Initial Firestore document creation
        doc_ref.set({
            'topic': topic, 
            'context': context, 
            'source_type': 'idea', 
            'duration': duration, 
            'status': 'processing', 
            'created_at': firestore.SERVER_TIMESTAMP,
            'num_speakers': num_speakers,
            'voice1': voice1,
            'voice2': voice2
        })
        
        # 1. Generate Script
        podcast_script = generate_script(topic, context, duration, num_speakers)
        if not podcast_script: raise Exception("Script generation failed.")
        
        # 2. Generate Audio
        if int(num_speakers) == 1:
            final_audio_path = generate_singlespeaker_audio(podcast_script, voice1, f"{job_id}.mp3")
        else:
            final_audio_path = generate_multispeaker_audio(podcast_script, voice1, voice2, job_id)
            
        if not final_audio_path: raise Exception("Audio generation failed.")

        # 3. Finalize Job
        return _finalize_job(job_id, final_audio_path, generated_script=podcast_script)
        
    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        # Clean up any potential local files on failure
        if os.path.exists(f"{job_id}.mp3"): os.remove(f"{job_id}.mp3")
        return {"status": "Failed", "error": str(e)}

# --- API Endpoints ---
@app.before_request
def before_first_request_func():
    initialize_services()

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context', 'num_speakers', 'voice1']):
        return jsonify({"error": "Missing required fields"}), 400
    
    job_id = str(uuid.uuid4())
    
    # Use the new task name and pass all the new parameters
    generate_podcast_task.delay(
        job_id, 
        data['topic'], 
        data['context'], 
        data.get('duration', '5 minutes'),
        data['num_speakers'],
        data['voice1'],
        data.get('voice2') # Can be None if 1 speaker
    )
    return jsonify({"message": "Podcast generation has been queued!", "job_id": job_id}), 202

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
