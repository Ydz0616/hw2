from flask import Flask, request, jsonify, render_template
import pickle
import os
import datetime
import time

from song_recommender import SongRecommender

app = Flask(__name__)

# Version of the code
VERSION = "1.0.0"

# Path to the model file
MODEL_PATH = os.environ.get('MODEL_FILE', '/shared/models/song_recommender.pkl')

# Global variable to store the model last modified date
model_date = None

def load_model():
    """Load the song recommendation model from disk."""
    global model_date
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model file {MODEL_PATH} not found. Make sure to train the model first.")
            return None
        
        # Get model modification time
        mod_time = os.path.getmtime(MODEL_PATH)
        model_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Load the model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded successfully from {MODEL_PATH} (last modified: {model_date})")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Load the model on startup
app.model = load_model()


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "version": VERSION})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    API endpoint to get song recommendations.
    
    Expected JSON input: 
    {
        "songs": ["Song 1", "Song 2", ...]
    }
    
    Returns JSON:
    {
        "songs": ["Recommended Song 1", "Recommended Song 2", ...],
        "rule_songs": ["Rule-based Recommendation 1", ...],
        "similarity_songs": ["Similarity-based Recommendation 1", ...],
        "version": "1.0.0",
        "model_date": "2023-02-25 08:30:45"
    }
    """
    try:
        # Get the JSON data from the request
        data = request.get_json(force=True)
        
        # Validate input
        if not data or "songs" not in data:
            return jsonify({
                "error": "Invalid request. Please provide a 'songs' field."
            }), 400
        
        input_songs = data["songs"]
        
        # Validate songs list
        if not isinstance(input_songs, list) or len(input_songs) == 0:
            return jsonify({
                "error": "The 'songs' field must be a non-empty list."
            }), 400
        
        # Check if model is loaded
        if app.model is None:
            return jsonify({
                "error": "Model not loaded. Please train the model first."
            }), 500
        
        # Check if the model was modified and reload if necessary
        if os.path.exists(MODEL_PATH):
            if os.path.getmtime(MODEL_PATH) > os.path.getmtime(__file__):
                # Reload the model if it was modified after the server started
                print("Model file was modified. Reloading...")
                app.model = load_model()
        
        # Get number of recommendations from request or default to 5
        num_recommendations = int(request.args.get('count', 5))
        
        # Cap at 10
        num_recommendations = min(num_recommendations, 10)
        
        # Get rule-based recommendations
        rule_recommendations = app.model._recommend_from_rules(input_songs, num_recommendations)
        
        # Get similarity-based recommendations
        similarity_recommendations = app.model._find_similar_song_names(input_songs, num_recommendations)
        
        # Get combined recommendations
        recommendations = app.model.recommend_songs(input_songs, num_recommendations)
        
        # Prepare response
        response = {
            "songs": recommendations,
            "rule_songs": rule_recommendations,
            "similarity_songs": similarity_recommendations,
            "version": VERSION,
            "model_date": model_date
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Note: In production, use gunicorn or another WSGI server instead
    # This is just for testing
    print(f"Starting server with model from {MODEL_PATH}")
    app.run(host="0.0.0.0", port=5000, debug=True) 