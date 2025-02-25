#!/usr/bin/env python3

"""
Model watcher script for the Frontend container.
This script will watch for changes to the model file and reload it if necessary.
It also starts the Flask application.
"""

import os
import sys
import time
import json
import logging
import hashlib
import threading
import subprocess
import signal
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
SHARED_DIR = os.environ.get('SHARED_DIR', '/shared')
MODEL_DIR = os.path.join(SHARED_DIR, 'models')
MODEL_FILE = os.environ.get('MODEL_FILE', os.path.join(MODEL_DIR, 'song_recommender.pkl'))
INFO_FILE = os.environ.get('INFO_FILE', os.path.join(MODEL_DIR, 'model_info.json'))
CHECK_INTERVAL = int(os.environ.get('CHECK_INTERVAL', '10'))  # Check every 10 seconds
FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.environ.get('FLASK_PORT', '5000'))

# Flask process
flask_process = None
last_model_hash = None

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    if not os.path.exists(file_path):
        return None
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def get_model_info():
    """Get model information from the info file."""
    if not os.path.exists(INFO_FILE):
        return None
    
    try:
        with open(INFO_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading model info file: {e}")
        return None

def start_flask():
    """Start the Flask application."""
    global flask_process
    
    # Stop existing process if running
    if flask_process is not None:
        logger.info("Stopping existing Flask process...")
        try:
            os.killpg(os.getpgid(flask_process.pid), signal.SIGTERM)
            flask_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping Flask process: {e}")
            try:
                os.killpg(os.getpgid(flask_process.pid), signal.SIGKILL)
            except:
                pass
    
    # Start new process
    logger.info(f"Starting Flask application on {FLASK_HOST}:{FLASK_PORT}...")
    cmd = [
        "gunicorn", 
        "--bind", f"{FLASK_HOST}:{FLASK_PORT}", 
        "--workers", "4", 
        "--worker-class", "gthread",
        "app:app"
    ]
    
    flask_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    logger.info(f"Flask application started with PID {flask_process.pid}")
    return flask_process

def check_model_update(last_hash=None, last_mtime=None):
    """
    Check if the model file has been updated.
    
    Args:
        last_hash: Last known hash of the model file
        last_mtime: Last known modification time of the model file
        
    Returns:
        Tuple of (has_changed, new_hash, new_mtime)
    """
    # Check if model file exists
    if not os.path.exists(MODEL_FILE):
        return False, None, None
    
    # Get current hash and mtime
    current_hash = calculate_file_hash(MODEL_FILE)
    current_mtime = os.path.getmtime(MODEL_FILE)
    
    # Check if changed
    has_changed = (
        (last_hash is None or current_hash != last_hash) or
        (last_mtime is None or current_mtime > last_mtime)
    )
    
    return has_changed, current_hash, current_mtime

def main():
    """Main function to watch for model updates and restart Flask app."""
    logger.info("Starting model watcher...")
    
    # Start Flask app
    start_flask()
    
    # Initial model check
    logger.info(f"Initial model load: {MODEL_FILE}")
    model_info = get_model_info()
    if model_info:
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
    
    # Get initial hash and mtime
    _, last_hash, last_mtime = check_model_update()
    
    # Main watch loop
    while True:
        try:
            # Check if model has changed
            has_changed, current_hash, current_mtime = check_model_update(last_hash, last_mtime)
            
            if has_changed:
                logger.info("Model file was modified. Reloading...")
                
                # Update last hash and mtime
                last_hash = current_hash
                last_mtime = current_mtime
                
                # Restart Flask app
                start_flask()
                
                # Log model info
                model_info = get_model_info()
                if model_info:
                    logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
            
            # Check if Flask process is still running
            if flask_process.poll() is not None:
                logger.warning("Flask process has stopped. Restarting...")
                start_flask()
            
            # Sleep for the check interval
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down...")
            if flask_process is not None:
                os.killpg(os.getpgid(flask_process.pid), signal.SIGTERM)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in model watcher: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main() 