#@title Setup CorentinJ/Real-Time-Voice-Cloning

#@markdown * clone the project
#@markdown * download pretrained models
#@markdown * initialize the voice cloning models

# %tensorflow_version 1.x
import os
from os.path import exists, join, basename, splitext

# git_repo_url = 'https://github.com/CorentinJ/Real-Time-Voice-Cloning.git'
# project_name = splitext(basename(git_repo_url))[0]
# print(project_name)
# if not exists(project_name):
#     print("Downloading Files")
#     # clone and install
#     !git clone -q --recursive {git_repo_url}
#     # install dependencies
#     !cd {project_name} && pip install -q -r requirements.txt
#     !pip install -q gdown
#     !apt-get install -qq libportaudio2
#     !pip install -q https://github.com/tugstugi/dl-colab-notebooks/archive/colab_utils.zip

#   # download pretrained model
#     !cd {project_name} && gdown https://drive.google.com/uc?id=1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc && unzip pretrained.zip

import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from voice_clone.synthesizer.inference import Synthesizer
from voice_clone.encoder import inference as encoder
from voice_clone.vocoder import inference as vocoder

BASE_PATH_VOICE_CLONE "./voice_clone/"
SAMPLE_RATE = 22050
embedding = None


# loading Models
encoder.load_model(BASE_PATH_VOICE_CLONE / Path("encoder/saved_models/pretrained.pt"))
synthesizer = Synthesizer(BASE_PATH_VOICE_CLONE / Path("synthesizer/saved_models/logs-pretrained/taco_pretrained"))
vocoder.load_model(BASE_PATH_VOICE_CLONE / Path("vocoder/saved_models/pretrained/pretrained.pt"))
print("All models Load Sucessfully")




def _compute_embedding(audio):
    '''
    Description 
        Loading Embedding from the audio file to clone
        
    Input:
        audio: Audio File 
        
    Output
        Embeddings
    
    '''
    display(Audio(audio, rate=SAMPLE_RATE, autoplay=True))
    global embedding
    embedding = None
    embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, SAMPLE_RATE))

def read_audio_file(path):
    clear_output()
    fs, data = wavfile.read(path)
    _compute_embedding(data)

audio_file_path = "/home/sohaib/Downloads/WhatsApp Ptt 2020-08-29 at 1.28.35 PM.wav"
read_audio_file(audio_file_path)
print("Embedding Loads Sucessfully")



def clone_voice(text):
    
    def synthesize(embed, text):
        print("Synthesizing new audio...")
        #with io.capture_output() as captured:
        specs = synthesizer.synthesize_spectrograms([text], [embed])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        clear_output()
        display(Audio(generated_wav, rate=synthesizer.sample_rate, autoplay=True))

    if embedding is None:
        print("first record a voice or upload a voice file!")
    else:
        synthesize(embedding, text)
        print("Voice Clonned Sucessfully")
        
text = "I am bhola record I am here to see you in the middle of the earth hello there" #@param {type:"string"}
clone_voice(text)