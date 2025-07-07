# visify_routes.py

import uuid
from flask import Blueprint, request, jsonify
from firebase_admin import firestore

# Get the main app's Celery instance and service clients
from app import celery, db

# Create a Blueprint
visify_bp = Blueprint('visify_bp', __name__)


# --- Visify Celery Task (Phase 1) ---
@celery.task
def generate_video_task(job_id, idea, context):
    """
    Celery task for the entire video generation pipeline.
    Phase 1: Create a placeholder document in Firestore.
    """
    print(f"WORKER: Starting Visify job {job_id} for idea: {idea}")
    doc_ref = db.collection('videos').document(job_id)
    
    try:
        # Step 1: Create the placeholder record
        doc_ref.set({
            'idea': idea,
            'context': context,
            'status': 'pending',
            'created_at': firestore.SERVER_TIMESTAMP
        })
        print(f"Successfully created placeholder in Firestore for video job: {job_id}")

        # --- FUTURE PHASES WILL GO HERE ---
        # Phase 2: Generate visual plan
        # doc_ref.update({'status': 'planning'})
        # visual_plan = generate_visual_plan(...) # from video_generator.py
        # doc_ref.update({'visual_plan': visual_plan})
        
        # Phase 3: Generate assets
        # doc_ref.update({'status': 'generating_assets'})
        # ...

        # Phase 4: Assemble video
        # doc_ref.update({'status': 'assembling'})
        # ...

        # For now, we'll just mark it as complete for testing purposes.
        # REMOVE THIS LINE in future phases.
        doc_ref.update({'status': 'complete', 'video_url': 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4'})


    except Exception as e:
        print(f"ERROR in Visify task {job_id}: {e}")
        doc_ref.update({'status': 'failed', 'error_message': str(e)})


# --- Visify API Endpoints (Phase 1) ---
@visify_bp.route("/generate-video", methods=["POST"])
def handle_video_generation():
    data = request.get_json()
    if not data or 'idea' not in data:
        return jsonify({"error": "An 'idea' is required."}), 400
    
    job_id = str(uuid.uuid4())
    
    generate_video_task.delay(
        job_id,
        data['idea'],
        data.get('context', '') # context is optional
    )
    
    return jsonify({"message": "Video generation has been queued!", "job_id": job_id}), 202


@visify_bp.route("/video-status/<job_id>", methods=["GET"])
def get_video_status(job_id):
    try:
        doc_ref = db.collection('videos').document(job_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(doc.to_dict()), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500