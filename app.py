# app.py - Final Production Version (with Download Timeout)

import os
import uuid
import re
import io
import json
import base64
import requests
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from pydub import AudioSegment
from google.cloud import texttospeech, aiplatform
from vertexai.preview.vision_models import ImageGenerationModel
from google.protobuf import struct_pb2
import google.generativeai as genai
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip, concatenate_videoclips

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
aip_client = None
pexels_api_key = None

def initialize_services():
    """Initializes all external services using environment variables."""
    global db, bucket, tts_client, genai_model, aip_client, pexels_api_key

    if not firebase_admin._apps:
        try:
            print("Attempting to initialize Firebase using Application Default Credentials...")
            project_id = os.environ.get('GCP_PROJECT_ID')
            firebase_admin.initialize_app(options={'projectId': project_id}) 
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
    
    if aip_client is None:
        try:
            print("Initializing Google Cloud AI Platform client...")
            project_id = os.environ.get('GCP_PROJECT_ID')
            location = 'us-central1'
            aiplatform.init(project=project_id, location=location)
            print("AI Platform client initialized.")
        except Exception as e:
            print(f"FATAL: Could not initialize AI Platform client: {e}")
            raise e
            
    if pexels_api_key is None:
        try:
            print("Initializing Pexels API key...")
            pexels_api_key = os.environ.get('PEXELS_API_KEY')
            if not pexels_api_key:
                raise ValueError("PEXELS_API_KEY environment variable not set.")
            print("Pexels API key loaded.")
        except Exception as e:
            print(f"FATAL: Could not initialize Pexels key: {e}")
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
    print(f"Generating PODCAST script for topic: {topic}")
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
    print("Podcast script generated successfully.")
    return response.text

def generate_video_script_from_idea(topic, context, duration):
    print(f"Generating VIDEO script for topic: {topic}")
    prompt = (
        "You are a scriptwriter for a high-impact social media marketing video. Your task is to write a script for two professional presenters, Trystan (male) and Saylor (female). "
        "Their tone should be professional, persuasive, and confident. The dialogue should be concise, impactful, and designed to sell a product or idea. "
        f"The topic is: '{topic}'. "
        f"Additional context: '{context}'. "
        f"The video should be approximately {duration} long. "
        "--- \n"
        "IMPORTANT INSTRUCTIONS: \n"
        "1.  Start each line with the speaker's tag, either '[Trystan]' or '[Saylor]'. \n"
        "2.  Alternate speakers for each line of dialogue. \n"
        "3.  Do NOT include any other text, directions, or formatting. \n"
        "4.  EXAMPLE: \n"
        "[Trystan] Are you looking to revolutionize your content strategy? \n"
        "[Saylor] Today, we're introducing a tool that will change everything. \n"
        "[Trystan] Get ready for Visify, the future of AI-powered video creation."
    )
    response = genai_model.generate_content(prompt)
    print("Video script generated successfully.")
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

def generate_visual_plan(script_text):
    """Takes a script and decides whether to use an image or video for each part."""
    print("Generating visual plan from script...")
    dialogue_only = "\n".join([dialogue for _, dialogue in parse_script(script_text)])
    
    prompt = (
        "You are a creative director. For each line of a podcast script, decide if a generated AI image or a stock video clip would be more visually engaging. "
        "For an image, create a descriptive prompt. For a video, provide 2-3 search keywords. "
        "The style should be modern, clean, and futuristic. "
        "IMPORTANT: Prefer using stock video clips for more dynamic scenes, and use AI images for more abstract or specific concepts. Aim for a mix of about 70% video and 30% images. "
        "--- \n"
        "PODCAST SCRIPT: \n"
        f"{dialogue_only}"
        "--- \n"
        "Respond with ONLY a JSON array of objects. Each object must have a 'type' ('image' or 'video') and a 'prompt' (for images) or 'keywords' (for videos). "
        'EXAMPLE RESPONSE: [{"type": "image", "prompt": "A vibrant, abstract representation of data"}, {"type": "video", "keywords": "city skyline at night"}, {"type": "image", "prompt": "A minimalist brain with glowing neural networks"}]'
    )
    
    response = genai_model.generate_content(prompt)
    print("Visual plan generated.")
    cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(cleaned_response)

def generate_image(prompt, filename, aspect_ratio="9:16"):
    """Generates a single image from a prompt."""
    print(f"Generating image for prompt: {prompt}")
    model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    response = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio=aspect_ratio)
    response.images[0].save(location=filename, include_generation_parameters=False)
    print(f"Saved image to {filename}")
    return filename

def find_and_download_stock_video(keywords, filename):
    """Finds and downloads a stock video from Pexels using direct HTTP requests."""
    print(f"Searching for stock video with keywords: {keywords}")
    headers = {'Authorization': pexels_api_key}
    params = {
        'query': keywords,
        'per_page': 1,
        'orientation': 'portrait'
    }
    # Add a timeout to prevent hanging
    response = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not data.get('videos'):
        print(f"No stock video found for keywords: {keywords}")
        return None
    
    video_files = data['videos'][0]['video_files']
    video_url = None
    for item in video_files:
        if item.get('height') == 1920:
            video_url = item['link']
            break
    if not video_url:
        video_url = video_files[0]['link']

    print(f"Downloading video from {video_url}")
    # Add a timeout to the download request as well
    video_data = requests.get(video_url, timeout=60).content
    with open(filename, 'wb') as handler:
        handler.write(video_data)
    print(f"Saved video to {filename}")
    return filename

def assemble_video_with_ffmpeg(media_paths, audio_path, output_path):
    """Assembles a video using direct FFmpeg commands for memory efficiency."""
    print("Assembling video with FFmpeg...")

    if not media_paths:
        raise ValueError("Cannot create video with no media.")

    audio_duration = float(subprocess.check_output(['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")]))
    duration_per_clip = audio_duration / len(media_paths)

    concat_file_path = f"{uuid.uuid4()}_concat.txt"
    temp_clips = []

    with open(concat_file_path, 'w') as f:
        for i, path in enumerate(media_paths):
            temp_output = f"{uuid.uuid4()}_temp_{i}.mp4"
            
            if path.endswith('.png'):
                cmd = [
                    'ffmpeg', '-loop', '1', '-i', path, '-c:v', 'libx264', '-t', str(duration_per_clip),
                    '-pix_fmt', 'yuv420p', '-y', temp_output
                ]
            elif path.endswith('.mp4'):
                cmd = [
                    'ffmpeg', '-i', path, '-t', str(duration_per_clip), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-y', temp_output
                ]
            
            subprocess.run(cmd, check=True)
            f.write(f"file '{temp_output}'\n")
            temp_clips.append(temp_output)

    final_cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file_path,
        '-i', audio_path, '-c:v', 'libx264', '-c:a', 'aac', '-shortest',
        '-y', output_path
    ]
    subprocess.run(final_cmd, check=True)

    os.remove(concat_file_path)
    for clip in temp_clips:
        os.remove(clip)

    print(f"Video assembled and saved to {output_path}")


def _finalize_job(job_id, collection_name, local_file_path, storage_path, generated_script=None, visual_plan=None):
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
    if visual_plan:
        update_data['visual_plan'] = visual_plan

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
def generate_video_from_idea_task(job_id, topic, context, duration, aspect_ratio):
    print(f"WORKER: Started HYBRID VIDEO job {job_id} for topic: {topic}")
    doc_ref = db.collection('videos').document(job_id)
    audio_filepath = f"{job_id}_audio.mp3"
    video_filepath = f"{job_id}_video.mp4"
    media_paths = []
    
    try:
        doc_ref.set({'topic': topic, 'context': context, 'source_type': 'idea', 'duration': duration, 'status': 'processing', 'aspect_ratio': aspect_ratio, 'created_at': firestore.SERVER_TIMESTAMP})
        
        original_script = generate_video_script_from_idea(topic, context, duration)
        voices = ['en-US-Chirp3-HD-Iapetus', 'en-US-Chirp3-HD-Leda']
        generate_podcast_audio(original_script, audio_filepath, voices)
        
        visual_plan = generate_visual_plan(original_script)
        
        for i, item in enumerate(visual_plan):
            filename_base = f"{job_id}_media_{i}"
            try:
                if item['type'] == 'video':
                    filename = f"{filename_base}.mp4"
                    video_path = find_and_download_stock_video(item['keywords'], filename)
                    if video_path:
                        media_paths.append(video_path)
                    else:
                        print(f"Fallback: No video found for '{item['keywords']}'. Generating image instead.")
                        filename = f"{filename_base}.png"
                        generate_image(item['keywords'], filename, aspect_ratio)
                        media_paths.append(filename)
                elif item['type'] == 'image':
                    filename = f"{filename_base}.png"
                    generate_image(item['prompt'], filename, aspect_ratio)
                    media_paths.append(filename)
            except Exception as e:
                print(f"Warning: Could not process visual item {i}. Reason: {e}. Skipping.")
                continue

        assemble_video_with_ffmpeg(media_paths, audio_filepath, video_filepath)
        return _finalize_job(job_id, 'videos', video_filepath, f"videos/{video_filepath}", generated_script=original_script, visual_plan=visual_plan)

    except Exception as e:
        print(f"ERROR in video task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        return {"status": "Failed", "error": str(e)}
    finally:
        # Clean up temporary files
        if os.path.exists(audio_filepath): os.remove(audio_filepath)
        for path in media_paths:
            if os.path.exists(path): os.remove(path)

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
    aspect_ratio = data.get('aspect_ratio', '9:16') # Default to portrait
    generate_video_from_idea_task.delay(job_id, data['topic'], data['context'], data.get('duration', '1 minute'), aspect_ratio)
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


