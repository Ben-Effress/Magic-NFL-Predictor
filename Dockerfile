# Use the official Python image as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
COPY PredictionData /app/PredictionData

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable for the Flask app
ENV FLASK_APP=website/app.py

# Expose the port that the Flask app will listen on
EXPOSE $PORT

# Run the command to start the Flask app
CMD flask run --host=0.0.0.0 --port=$PORT
