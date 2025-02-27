FROM python:3.9-slim-bullseye

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary code files
COPY train_model.py .
COPY song_recommender.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV SHARED_DIR=/shared
ENV MIN_SUPPORT=0.05
ENV MIN_CONFIDENCE=0.1

# Run the model training script
CMD ["python", "train_model.py"] 


