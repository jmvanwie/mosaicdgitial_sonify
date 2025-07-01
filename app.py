# app.py - Final version serving both Frontend and API

import os
import uuid
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from pydub import AudioSegment
from google.cloud import texttospeech
import google.generativeai as genai

app = Flask(__name__)
# CORS is still good practice for APIs, this simple setup is fine.
CORS(app) 

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

# --- Core Logic Functions (unchanged) ---
def generate_script(topic, context, duration, num_speakers):
    print(f"Generating AI script for {num_speakers} speaker(s) on topic: {topic}")
    if int(num_speakers) == 1:
        prompt = (f"You are a professional podcast scriptwriter. Write a compelling and engaging monologue podcast script. "
                  f"The script should be approximately {duration} in length. The topic is: '{topic}'. "
                  f"Additional context: '{context}'. Provide only the spoken words for the script.")
    else:
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
    print("Starting multi-speaker audio generation...")
    lines = re.split(r'(\[SPEAKER \d\])', script_text)
    combined_audio = AudioSegment.empty()
    temp_files = []
    for i in range(1, len(lines), 2):
        speaker_tag = lines[i]
        text_content = lines[i+1].strip()
        if not text_content: continue
        voice_name = voice1 if '1' in speaker_tag else voice2
        snippet_filename = f"{job_id}_snippet_{i}.mp3"
        temp_files.append(snippet_filename)
        synthesis_input = texttospeech.SynthesisInput(text=text_content)
        voice = texttospeech.VoiceSelectionParams(language_code=voice_name.split('-W')[0], name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(snippet_filename, "wb") as out:
            out.write(response.audio_content)
        snippet_audio = AudioSegment.from_mp3(snippet_filename)
        combined_audio += snippet_audio
    final_filename = f"{job_id}.mp3"
    combined_audio.export(final_filename, format="mp3")
    for f in temp_files:
        os.remove(f)
    print("Cleaned up temporary audio snippets.")
    return final_filename

def generate_singlespeaker_audio(script_text, voice_name, output_filepath):
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
    print(f"Finalizing job {job_id}...")
    storage_path = f"podcasts/{job_id}.mp3"
    blob = bucket.blob(storage_path)
    blob.upload_from_filename(local_audio_path)
    blob.make_public()
    podcast_url = blob.public_url
    os.remove(local_audio_path)
    update_data = {'status': 'complete', 'podcast_url': podcast_url, 'completed_at': firestore.SERVER_TIMESTAMP, 'generated_script': generated_script}
    db.collection('podcasts').document(job_id).update(update_data)
    print(f"Firestore document for job {job_id} updated to complete.")
    return {"status": "Complete", "podcast_url": podcast_url}

# --- Celery Task ---
@celery.task
def generate_podcast_task(job_id, topic, context, duration, num_speakers, voice1, voice2):
    print(f"WORKER: Started job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    try:
        doc_ref.set({'topic': topic, 'context': context, 'source_type': 'idea', 'duration': duration, 'status': 'processing', 'created_at': firestore.SERVER_TIMESTAMP,'num_speakers': num_speakers,'voice1': voice1,'voice2': voice2})
        podcast_script = generate_script(topic, context, duration, num_speakers)
        if int(num_speakers) == 1:
            final_audio_path = generate_singlespeaker_audio(podcast_script, voice1, f"{job_id}.mp3")
        else:
            final_audio_path = generate_multispeaker_audio(podcast_script, voice1, voice2, job_id)
        return _finalize_job(job_id, final_audio_path, generated_script=podcast_script)
    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        if os.path.exists(f"{job_id}.mp3"): os.remove(f"{job_id}.mp3")
        return {"status": "Failed", "error": str(e)}

# --- API Endpoints ---
@app.route("/")
def serve_frontend():
    """Serves the main index.html file from the templates folder."""
    return render_template('index.html')

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    data = request.get_json()
    if not data or not all(k in data for k in ['topic', 'context', 'num_speakers', 'voice1']):
        return jsonify({"error": "Missing required fields"}), 400
    job_id = str(uuid.uuid4())
    generate_podcast_task.delay(job_id, data['topic'], data['context'], data.get('duration', '5 minutes'),data['num_speakers'],data['voice1'],data.get('voice2'))
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
    with app.app_context():
        initialize_all_services()
    app.run(debug=True, port=int(os.environ.get("PORT", 8080)))