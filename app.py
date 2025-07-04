# app.py - Visify Phase 2: Visual Prompt Generation

import os
import uuid
import re
import io
import json
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
def generate_script_from_idea(topic, context, duration):
    print(f"Generating AI script for topic: {topic}")
    prompt = (
        "You are a scriptwriter for a popular podcast. Your task is to write a script for two AI hosts, Trystan (male) and Saylor (female). "
        "The hosts are witty, charismatic, and engaging. The dialogue should feel natural, warm, and have a good back-and-forth conversational flow. "
        f"The topic is: '{topic}'. "
        f"Additional context: '{context}'. "
        f"The podcast should be approximately {duration} long. "
        "--- \n"
        "IMPORTANT INSTRUCTIONS: \n"
        "1.  Start each line with the speaker's tag, either '[Trystan]' or '[Saylor]'. \n"
        "2.  Alternate speakers for each line of dialogue. \n"
        "3.  Do NOT include any other text, directions, or formatting. \n"
        "4.  EXAMPLE: \n"
        "[Trystan] Welcome back to AI Insights! Today, we're tackling a huge topic: quantum computing. \n"
        "[Saylor] It sounds intimidating, but I promise we'll make it fun. Ready to dive in? \n"
        "[Trystan] Absolutely. So, at its core, what makes a quantum computer different from the one on your desk?"
    )
    response = genai_model.generate_content(prompt)
    print("AI script generated successfully.")
    return response.text

def parse_script(script_text):
    """Parses a script with named speaker tags into a list of (speaker, dialogue) tuples."""
    print("Parsing script...")
    pattern = re.compile(r'\[(Trystan|Saylor)\]\s*([^\n\[\]]*)')
    dialogue_parts = pattern.findall(script_text)
    print(f"Parsed {len(dialogue_parts)} dialogue parts.")
    return dialogue_parts

def generate_podcast_audio(script_text, output_filepath, voice_names):
    """Generates podcast audio by parsing a script and stitching the parts together."""
    print(f"Generating audio for voices: {voice_names}")
    dialogue_parts = parse_script(script_text)
    if not dialogue_parts:
        raise ValueError("The script is empty or could not be parsed. Cannot generate audio.")
    voice_map = {'Trystan': voice_names[0], 'Saylor': voice_names[1]}
    combined_audio = AudioSegment.empty()
    for speaker_name, dialogue in dialogue_parts:
        dialogue = dialogue.strip()
        if not dialogue: continue
        voice_name = voice_map.get(speaker_name)
        if not voice_name: continue
        print(f"Synthesizing dialogue for {speaker_name} with voice {voice_name}...")
        phonetic_dialogue = dialogue.replace("Saylor", "sailor")
        synthesis_input = texttospeech.SynthesisInput(text=phonetic_dialogue)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=voice_name.split('-')[0] + '-' + voice_name.split('-')[1],
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)
        audio_chunk = AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")
        combined_audio += audio_chunk + AudioSegment.silent(duration=600)
    combined_audio.export(output_filepath, format="mp3")
    print(f"Audio content successfully written to file '{output_filepath}'")
    return True

def generate_visual_prompts(script_text):
    """Takes a script and generates a list of visual prompts using Gemini."""
    print("Generating visual prompts from script...")
    dialogue_only = "\n".join([dialogue for _, dialogue in parse_script(script_text)])
    
    prompt = (
        "You are a creative director for a marketing agency. Your task is to create a series of visual prompts for an AI image generator. "
        "These visuals will accompany a podcast script. For each line of dialogue, create a vivid, descriptive, and visually interesting prompt that captures the essence of what's being said. "
        "The style should be modern, clean, and slightly futuristic. "
        "--- \n"
        "PODCAST SCRIPT: \n"
        f"{dialogue_only}"
        "--- \n"
        "IMPORTANT: Respond with ONLY a JSON array of strings. Each string in the array should be an image prompt. The number of prompts must exactly match the number of dialogue lines in the script. "
        'EXAMPLE RESPONSE: ["A vibrant, abstract representation of data flowing through circuits.", "A silhouette of a person looking at a complex holographic interface.", "A minimalist design of a brain with glowing neural networks."]'
    )
    
    response = genai_model.generate_content(prompt)
    print("Visual prompts generated.")
    # Clean the response to ensure it's valid JSON
    cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(cleaned_response)

def _finalize_job(job_id, collection_name, local_file_path, storage_path, generated_script=None):
    """Finalizes a job by uploading the file and updating Firestore."""
    print(f"Finalizing job {job_id} in collection {collection_name}...")
    blob = bucket.blob(storage_path)
    
    print(f"Uploading {local_file_path} to {storage_path}...")
    blob.upload_from_filename(local_file_path)
    blob.make_public()
    public_url = blob.public_url
    print(f"Upload complete. Public URL: {public_url}")

    os.remove(local_file_path)
    print(f"Removed temporary file: {local_file_path}")

    update_data = {'status': 'complete', 'url': public_url, 'completed_at': firestore.SERVER_TIMESTAMP}
    if generated_script:
        update_data['generated_script'] = generated_script

    db.collection(collection_name).document(job_id).update(update_data)
    print(f"Firestore document for job {job_id} updated to complete.")
    return {"status": "Complete", "url": public_url}

# --- Celery Task Definitions ---
@celery.task
def generate_podcast_from_idea_task(job_id, topic, context, duration, voices):
    print(f"WORKER: Started PODCAST job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    output_filepath = f"{job_id}.mp3"
    try:
        doc_ref.set({'topic': topic, 'context': context, 'source_type': 'idea', 'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP, 'voices': voices})
        original_script = generate_script_from_idea(topic, context, duration)
        if not generate_podcast_audio(original_script, output_filepath, voices): 
            raise Exception("Audio generation failed.")
        return _finalize_job(job_id, 'podcasts', output_filepath, f"podcasts/{output_filepath}", generated_script=original_script)
    except Exception as e:
        print(f"ERROR in podcast task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        if os.path.exists(output_filepath): os.remove(output_filepath)
        return {"status": "Failed", "error": str(e)}

@celery.task
def generate_video_from_idea_task(job_id, topic, context, duration):
    print(f"WORKER: Started VIDEO job {job_id} for topic: {topic}")
    doc_ref = db.collection('videos').document(job_id)
    try:
        doc_ref.set({'topic': topic, 'context': context, 'source_type': 'idea', 'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP})
        
        # Step 1: Generate the script
        original_script = generate_script_from_idea(topic, context, duration)
        
        # Step 2: Generate the visual prompts
        visual_prompts = generate_visual_prompts(original_script)
        print("Generated Visual Prompts:", visual_prompts)
        
        # Placeholder for future steps
        # audio_filepath = generate_podcast_audio(...)
        # images = generate_images_from_prompts(visual_prompts)
        # video_filepath = assemble_video(audio_filepath, images)
        # return _finalize_job(...)
        
        # For now, we'll just mark it as complete to test the workflow.
        doc_ref.update({
            'status': 'complete_placeholder',
            'message': 'Visual prompt generation successful. Image/video generation next.',
            'generated_script': original_script,
            'visual_prompts': visual_prompts
        })
    except Exception as e:
        print(f"ERROR in video task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})

# --- API Endpoints ---
@app.before_request
def before_first_request_func():
    initialize_services()

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Sonify API! The server is running."})

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context']):
        return jsonify({"error": "topic and context are required"}), 400
    job_id = str(uuid.uuid4())
    voices = data.get('voices', ['en-US-Chirp3-HD-Iapetus', 'en-US-Chirp3-HD-Leda'])
    generate_podcast_from_idea_task.delay(job_id, data['topic'], data['context'], data.get('duration', '5 minutes'), voices)
    return jsonify({"message": "Podcast generation has been queued!", "job_id": job_id}), 202

@app.route("/generate-video", methods=["POST"])
def handle_video_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context']):
        return jsonify({"error": "topic and context are required"}), 400
    job_id = str(uuid.uuid4())
    generate_video_from_idea_task.delay(job_id, data['topic'], data['context'], data.get('duration', '1 minute'))
    return jsonify({"message": "Video generation has been queued!", "job_id": job_id}), 202

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

# We will need a similar status endpoint for videos
@app.route("/video-status/<job_id>", methods=["GET"])
def get_video_status(job_id):
    try:
        doc_ref = db.collection('videos').document(job_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(doc.to_dict()), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
