

import os
import numpy as np
import moviepy.editor as mp
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import nltk
import spacy
import re

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def check_file_accessibility(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    if not os.access(file_path, os.R_OK):
        print(f"File is not readable (permission denied): {file_path}")
        return False
    return True

def create_text_image(text, font_path='arial.ttf', fontsize=24, color='white', bg_color=None):
    font = ImageFont.truetype(font_path, fontsize)
    dummy_img = Image.new('RGB', (1, 1), bg_color)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new('RGBA', (text_width, text_height), bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=color)
    return img

video_path = r"C:\Users\pudhin\Videos\your_video.mp4\WhatsApp Video 2024-06-21 at 14.27.10_86ecac81.mp4"

if not check_file_accessibility(video_path):
    print("Please check the file path and permissions.")
    exit(1)

try:
    video = mp.VideoFileClip(video_path)
except OSError as e:
    print(f"Error loading video file: {e}")
    print("Check if the file path is correct and if the file is accessible.")
    exit(1)

# Extract the entire audio from the video
audio_path = 'full_audio.wav'
video.audio.write_audiofile(audio_path)

# Transcribe the full audio
recognizer = sr.Recognizer()
audio_clip = sr.AudioFile(audio_path)
with audio_clip as source:
    audio = recognizer.record(source)
try:
    transcript = recognizer.recognize_google(audio, show_all=True)
    if 'alternative' in transcript and len(transcript['alternative']) > 0:
        full_text = transcript['alternative'][0]['transcript']
        print(f"Full Transcript: {full_text}")

        # Load the spacy model
        nlp = spacy.load("en_core_web_sm")

        # Process the transcript text
        doc = nlp(full_text)

        # Identify entities like greetings
        greetings = []
        for entity in doc.ents:
            if entity.label_ == "GREETING":
                greetings.append(entity.text)

        # Tokenize the transcript
        filtered_words = []
        for word in full_text.split():
            if word not in stopwords.words('english'):
                filtered_words.append(word)

        # Calculate the start and end times for each word
        word_times = []
        for i, word in enumerate(filtered_words):
            start_time = video.duration * transcript['alternative'][0]['transcript'].index(word) / len(transcript['alternative'][0]['transcript'])
            end_time = video.duration * (transcript['alternative'][0]['transcript'].index(word) + len(word)) / len(transcript['alternative'][0]['transcript'])
            word_times.append((start_time, end_time))

        # Create a GIF for each individual word or phrase
        for i, word in enumerate(filtered_words):
            start_time, end_time = word_times[i]
            segment_clip = video.subclip(start_time, end_time)
            text_img = create_text_image(word)
            txt_clip = ImageClip(np.array(text_img)).set_duration(end_time - start_time).set_position(('center', 'bottom'))
            video_with_text = CompositeVideoClip([segment_clip, txt_clip.set_duration(end_time - start_time)])
            gif_path = f'gif_{i + 1}.gif'
            video_with_text.write_gif(gif_path, fps=10, program='ffmpeg')
            print(f'GIF saved to {gif_path}')

    else:
        print("Audio not clear")
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print(f"Request error: {e}")
except Exception as e:
    print(f"Other error: {e}")

# Clean up temporary files
os.remove(audio_path)

# Exit the script
exit(0)
