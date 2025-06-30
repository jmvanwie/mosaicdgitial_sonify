# app.py - DIAGNOSTIC VERSION
# This file's only purpose is to check environment variables.

import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Allow requests from anywhere for this simple test
CORS(app)

@app.route("/")
def index():
    """A simple welcome message to confirm the server is running."""
    return "Diagnostic Server is running. Please go to /debug-env to see variables."

@app.route("/debug-env")
def debug_env():
    """
    Reads all the environment variables we care about and displays them.
    This will PROVE if the variables are being set correctly.
    """
    # List of all the keys we expect to be in the environment
    keys_to_check = [
        'GEMINI_API_KEY',
        'CELERY_BROKER_URL',
        'FIREBASE_STORAGE_BUCKET',
        'FIREBASE_PROJECT_ID',
        'GOOGLE_APPLICATION_CREDENTIALS' # This is set by Render for the secret file
    ]
    
    # Create a dictionary to hold the results
    env_vars = {}
    
    for key in keys_to_check:
        # Get the value from the environment. If it's not found, record that.
        value = os.environ.get(key)
        
        if key == 'GEMINI_API_KEY' and value:
            # For security, only show the first and last few characters of the key
            env_vars[key] = f"{value[:4]}...{value[-4:]}"
        elif value:
            env_vars[key] = "Found and Set"
        else:
            env_vars[key] = "!!! NOT FOUND !!!"
            
    return jsonify(env_vars)

if __name__ == '__main__':
    # Gunicorn will use this to run the app on Render
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))

