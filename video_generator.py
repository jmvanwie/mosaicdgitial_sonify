# video_generator.py
import requests
from vertexai.preview.vision_models import ImageGenerationModel
# This file will hold the core logic for generating the video.
# You will build out these functions in Phases 2, 3, and 4.
from app import pixabay_api_key, pexels_api_key
from app import genai_model  # Example of getting a shared client

def generate_visual_plan(script):
    """
    Phase 2: Takes a script and asks Gemini to return a JSON array of visual instructions.
    """
    print("Generating visual plan...")
    # Your logic here...
    pass

def generate_image(prompt, filename):
    """
    Generates a single image from a prompt using Vertex AI.
    """
    print(f"Generating image for prompt: {prompt}")
    # The model is initialized on-the-fly using your project's default credentials
    model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    response = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="9:16" # Portrait for social media
    )
    response.images[0].save(location=filename, include_generation_parameters=False)
    print(f"Saved image to {filename}")
    return filename

def find_and_download_stock_video(keywords, filename):
    """
    Finds and downloads a stock video from Pexels (or Pixabay as a fallback).
    """
    print(f"Searching for stock video with keywords: {keywords}")
    
    # --- Try Pexels First ---
    if pexels_api_key:
        headers = {'Authorization': pexels_api_key}
        params = {'query': keywords, 'per_page': 1, 'orientation': 'portrait'}
        response = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=30)
        
        if response.ok and response.json().get('videos'):
            video_files = response.json()['videos'][0]['video_files']
            # Find a good quality link (e.g., HD)
            video_url = next((item['link'] for item in video_files if item.get('height') == 1920), video_files[0]['link'])
            
            print(f"Downloading video from Pexels: {video_url}")
            video_data = requests.get(video_url, timeout=60).content
            with open(filename, 'wb') as handler:
                handler.write(video_data)
            return filename
        
    if pixabay_api_key:
        params = {'key': pixabay_api_key, 'q': keywords, 'per_page': 3, 'orientation': 'vertical'}
        response = requests.get("https://pixabay.com/api/videos/", params=params, timeout=30)

        if response.ok and response.json().get('hits'):
            video_info = response.json()['hits'][0]
            video_url = video_info['videos']['large']['url']

            print(f"Downloading video from Pixabay: {video_url}")
            video_data = requests.get(video_url, timeout=60).content
            with open(filename, 'wb') as handler:
                handler.write(video_data)
            return filename
        
    print(f"No stock video found for keywords: {keywords}")
    return None

def assemble_video_with_ffmpeg(assets, audio_path, output_path):
    """
    Phase 4: Stitches all generated assets and voice-over into a final MP4 video.
    """
    print("Assembling video with FFmpeg...")
    # Your logic here...
    pass