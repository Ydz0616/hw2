FROM python:3.9-slim-bullseye

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py model_watcher.py ./
COPY song_recommender.py ./
COPY static/ ./static/
COPY templates/ ./templates/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV SHARED_DIR=/shared
ENV MODEL_FILE=/shared/models/song_recommender.pkl
ENV CHECK_INTERVAL=5

# Create directory for shared volume
RUN mkdir -p /shared/models && chmod -R 777 /shared

# Expose the port the app runs on
EXPOSE 5000

# Run the application with model watching
CMD ["python", "model_watcher.py"] 