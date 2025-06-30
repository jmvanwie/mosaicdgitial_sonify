# All eventlet and environment variable workarounds have been removed,
# as we will now use the 'solo' pool for the Celery worker on Windows.

import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS 
from celery import Celery
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PyPDF2 import PdfReader # Added for PDF processing
from google.cloud import texttospeech # Added for real audio generation
from google.oauth2 import service_account # Added for explicit credentials
import google.generativeai as genai # Added for AI script generation

# --- App & Celery Configuration ---
app = Flask(__name__)
# Initialize CORS, allowing requests from your Wix website
origins = ["http://localhost:8000", "http://localhost:8888", "https://www.mosaicdigital.ai"]
CORS(app, resources={r"/*": {"origins": origins}})

# Using standard localhost, as eventlet is no longer a factor.
app.config.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0'
)

# --- Firebase & Google AI Initialization (lazy loading) ---
db = None
bucket = None
# It's good practice to get the API key from environment variables in a real app
# For simplicity, we'll assume it's available or configured elsewhere.
# genai.configure(api_key="YOUR_GEMINI_API_KEY") 

def initialize_firebase():
    """Initializes the Firebase app if not already done."""
    global db, bucket
    if not firebase_admin._apps:
        try:
            # Construct an absolute path to the credentials file to avoid pathing issues
            basedir = os.path.abspath(os.path.dirname(__file__))
            key_path = os.path.join(basedir, "serviceAccountKey.json")
            print(f"Attempting to load credentials from: {key_path}")

            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'mosaic-tess-1.firebasestorage.app', # Corrected bucket name
                'projectId': 'mosaic-tess-1',
            })
            print("Successfully connected to Firebase.")
            db = firestore.client()
            bucket = storage.bucket()
        except Exception as e:
            print(f"Could not connect to Firebase: {e}")
            # Propagate the exception to stop the app if Firebase connection fails
            raise e

# Celery factory function to ensure tasks run within the Flask app context
def make_celery(app):
    celery = Celery(
        'app', # Explicitly name the celery app to avoid naming conflicts
        backend=app.config['result_backend'],
        broker=app.config['broker_url']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                # Ensure Firebase is initialized within the task context
                initialize_firebase()
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

# --- Podcast Generation Logic ---
def generate_script_from_idea(topic, context, duration):
    """
    Generates a podcast script using the Google Gemini API.
    """
    print(f"Generating AI script for topic: {topic}")
    try:
        # For using `gemini-pro`, it's recommended to configure the API key.
        # This can be done by setting the GOOGLE_API_KEY environment variable.
        # For this example, we'll assume it's configured.
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = (
            f"You are a professional podcast scriptwriter. Your task is to write a compelling and engaging podcast script. "
            f"The script should be approximately {duration} in length. "
            f"The topic of the podcast is: '{topic}'. "
            f"Here is some additional context: '{context}'. "
            f"Please provide only the script content, without any introductory or concluding remarks about the script itself. "
            f"Just write the words to be spoken."
        )

        response = model.generate_content(prompt)
        script = response.text
        print("AI script generated successfully.")
        return script
    except Exception as e:
        print(f"Error during Gemini script generation: {e}")
        return None

def generate_real_podcast_file(text_content, output_filepath):
    """
    Generates a real podcast audio file from text using Google Cloud TTS.
    """
    try:
        print(f"Generating real audio for text (first 100 chars): {text_content[:100]}...")
        
        # --- Explicitly load credentials to avoid environment conflicts ---
        basedir = os.path.abspath(os.path.dirname(__file__))
        key_path = os.path.join(basedir, "serviceAccountKey.json")
        tts_credentials = service_account.Credentials.from_service_account_file(key_path)
        
        # Instantiates a client, passing the correct credentials
        tts_client = texttospeech.TextToSpeechClient(credentials=tts_credentials)
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
    """
    Opens a PDF file and extracts all text content.
    """
    print(f"Extracting text from {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        print("Text extraction from PDF successful.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


# --- Celery Task Definitions ---
@celery.task
def generate_podcast_from_url_task(job_id, source_url, duration):
    """
    Background job for processing a URL. Now includes duration.
    """
    print(f"Celery worker picked up URL job {job_id} for URL: {source_url}")
    doc_ref = db.collection('podcasts').document(job_id)
    basedir = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(basedir, 'temp_audio')
    os.makedirs(temp_dir, exist_ok=True)
    output_filepath = os.path.join(temp_dir, f"{job_id}.mp3")

    try:
        doc_ref.set({
            'source_url': source_url,
            'source_type': 'url',
            'duration': duration,
            'status': 'processing', 
            'created_at': firestore.SERVER_TIMESTAMP
        })
        # TODO: Replace with actual web scraping logic to get real text
        placeholder_text = f"This is a test podcast generated from the URL: {source_url}"
        if not generate_real_podcast_file(placeholder_text, output_filepath):
             raise Exception("Failed to generate audio file.")

        storage_path = f"podcasts/{job_id}.mp3"
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(output_filepath)
        blob.make_public() 
        podcast_url = blob.public_url
        os.remove(output_filepath)
        doc_ref.update({
            'status': 'complete', 
            'podcast_url': podcast_url, 
            'completed_at': firestore.SERVER_TIMESTAMP
        })
        print(f"Celery worker finished URL job {job_id}")
        return {"status": "Complete", "podcast_url": podcast_url}
    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        return {"status": "Failed", "error": str(e)}

@celery.task
def generate_podcast_from_pdf_task(job_id, temp_pdf_path, original_filename, duration):
    """
    Background job for processing a PDF file. Now includes duration.
    """
    print(f"Celery worker picked up PDF job {job_id} for file: {original_filename}")
    doc_ref = db.collection('podcasts').document(job_id)
    basedir = os.path.abspath(os.path.dirname(__file__))
    temp_audio_dir = os.path.join(basedir, 'temp_audio')
    os.makedirs(temp_audio_dir, exist_ok=True)
    output_filepath = os.path.join(temp_audio_dir, f"{job_id}.mp3")

    try:
        doc_ref.set({
            'original_filename': original_filename,
            'source_type': 'pdf',
            'duration': duration,
            'status': 'processing', 
            'created_at': firestore.SERVER_TIMESTAMP
        })
        extracted_text = extract_text_from_pdf(temp_pdf_path)
        print(f"Task {job_id}: Extracted {len(extracted_text)} characters.")
        os.remove(temp_pdf_path)
        
        # Use the extracted text to generate the real audio
        if not generate_real_podcast_file(extracted_text, output_filepath):
             raise Exception("Failed to generate audio file.")

        storage_path = f"podcasts/{job_id}.mp3"
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(output_filepath)
        blob.make_public()
        podcast_url = blob.public_url
        os.remove(output_filepath)
        doc_ref.update({
            'status': 'complete', 
            'podcast_url': podcast_url, 
            'completed_at': firestore.SERVER_TIMESTAMP
        })
        print(f"Celery worker finished PDF job {job_id}")
        return {"status": "Complete", "podcast_url": podcast_url}
    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        return {"status": "Failed", "error": str(e)}

@celery.task
def generate_podcast_from_idea_task(job_id, topic, context, duration):
    """
    Background job for generating a podcast from an idea.
    """
    print(f"Celery worker picked up IDEA job {job_id} for topic: {topic}")
    doc_ref = db.collection('podcasts').document(job_id)
    basedir = os.path.abspath(os.path.dirname(__file__))
    temp_audio_dir = os.path.join(basedir, 'temp_audio')
    os.makedirs(temp_audio_dir, exist_ok=True)
    output_filepath = os.path.join(temp_audio_dir, f"{job_id}.mp3")

    try:
        doc_ref.set({
            'topic': topic,
            'context': context,
            'source_type': 'idea',
            'duration': duration,
            'status': 'processing', 
            'created_at': firestore.SERVER_TIMESTAMP
        })

        # --- Generate the script using the Gemini API ---
        podcast_script = generate_script_from_idea(topic, context, duration)
        if not podcast_script:
            raise Exception("Failed to generate podcast script from Gemini API.")
        
        # Pass the generated 'podcast_script' to the TTS engine.
        if not generate_real_podcast_file(podcast_script, output_filepath):
             raise Exception("Failed to generate audio file.")

        storage_path = f"podcasts/{job_id}.mp3"
        blob = bucket.blob(storage_path)
        blob.upload_from_filename(output_filepath)
        blob.make_public()
        podcast_url = blob.public_url
        os.remove(output_filepath)
        doc_ref.update({
            'status': 'complete', 
            'podcast_url': podcast_url,
            'generated_script': podcast_script, # Save the real script
            'completed_at': firestore.SERVER_TIMESTAMP
        })
        print(f"Celery worker finished IDEA job {job_id}")
        return {"status": "Complete", "podcast_url": podcast_url}
    except Exception as e:
        print(f"ERROR in Celery task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})
        return {"status": "Failed", "error": str(e)}


# --- API Endpoints ---
@app.before_request
def before_request_func():
    """Ensure Firebase is initialized before each web request."""
    initialize_firebase()

@app.route("/generate-from-url", methods=["POST"])
def handle_url_generation():
    print("\nReceived request for /generate-from-url")
    data = request.get_json()
    source_url = data.get('url')
    duration = data.get('duration', 'not specified') # Default duration
    if not source_url:
        return jsonify({"error": "URL is required"}), 400
    job_id = str(uuid.uuid4())
    generate_podcast_from_url_task.delay(job_id, source_url, duration)
    return jsonify({"message": "Podcast generation from URL has been queued!", "job_id": job_id}), 202 

@app.route("/generate-from-pdf", methods=["POST"])
def handle_pdf_generation():
    print("\nReceived request for /generate-from-pdf")
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    duration = request.form.get('duration', 'not specified') # Get duration from form data
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({"error": "A selected PDF file is required"}), 400
    
    job_id = str(uuid.uuid4())
    original_filename = file.filename
    basedir = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(basedir, 'temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, f"{job_id}.pdf")
    file.save(temp_filepath)
    generate_podcast_from_pdf_task.delay(job_id, temp_filepath, original_filename, duration)
    return jsonify({"message": "Podcast generation from PDF has been queued!", "job_id": job_id}), 202

@app.route("/generate-from-idea", methods=["POST"])
def handle_idea_generation():
    print("\nReceived request for /generate-from-idea")
    data = request.get_json()
    topic = data.get('topic')
    context = data.get('context')
    duration = data.get('duration', 'not specified')
    if not all([topic, context, duration]):
        return jsonify({"error": "Topic, context, and duration are required"}), 400
    job_id = str(uuid.uuid4())
    generate_podcast_from_idea_task.delay(job_id, topic, context, duration)
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
