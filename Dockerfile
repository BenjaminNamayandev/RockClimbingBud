FROM python:3.9

# Update and install system dependencies for PyAudio and espeak for pyttsx3
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    portaudio19-dev \
    python3-pyaudio \
    espeak \ 
    && rm -rf /var/lib/apt/lists/*

# Add your Python script
ADD main.py .

# Install Python dependencies
RUN pip install opencv-python mediapipe numpy pyttsx3 pyaudio

CMD ["python", "./main.py"]
