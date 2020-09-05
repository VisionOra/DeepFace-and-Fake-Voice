#!/usr/bin/env python
# coding: utf-8

# # Real-Time Voice Cloning
# 
# This is a colab demo notebook using the open source project [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
# to clone a voice.
# 
# For other deep-learning Colab notebooks, visit [tugstugi/dl-colab-notebooks](https://github.com/tugstugi/dl-colab-notebooks).
# 
# 
# Original issue: https://github.com/tugstugi/dl-colab-notebooks/issues/18
# 
# ## Setup CorentinJ/Real-Time-Voice-Cloning

# In[1]:


get_ipython().system('pip install -r requirements_deepFake.txt')
# !pip install multiprocess
# !pip install librosa
# !pip install llvmlite
# !pip install numba
# !pip install llvmlite 0.33
# !pip uninstall -y numba
# !pip uninstall -y llvmlite

# !pip --use-feature=2020-resolver install numba==0.43.0
# !pip --use-feature=2020-resolver install llvmlite==0.31.0
# !pip install llvm
# !pip install fastparquet
# !pip install --upgrade pip
# !python --version
# !sudo apt install python3-llvmlite
import sys
sys.path.append("./Deepfake/Wav2Lip-master/")
sys.path
get_ipython().system('pip install librosa')


# In[13]:


get_ipython().system('pip freeze > requirements_deepFake.txt')


# In[3]:


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
BASE_PATH_VOICE_CLONE = "./voice_clone/"
import sys
sys.path.append(BASE_PATH_VOICE_CLONE)
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder


SAMPLE_RATE = 22050
embedding = None


# loading Models
encoder.load_model(BASE_PATH_VOICE_CLONE / Path("encoder/saved_models/pretrained.pt"))
synthesizer = Synthesizer(BASE_PATH_VOICE_CLONE / Path("synthesizer/saved_models/logs-pretrained/taco_pretrained"))
vocoder.load_model(BASE_PATH_VOICE_CLONE / Path("vocoder/saved_models/pretrained/pretrained.pt"))
print("All models Load Sucessfully")


# In[4]:


import librosa

def _compute_embedding(audio):
    '''
    Description 
        Loading Embedding from the audio file to clone
        
    Input:
        audio: Audio File 
        
    Output
        Embeddings
    
    '''
    global embedding
    embedding = None
    embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, SAMPLE_RATE))

def read_audio_file(path):
    samples, sample_rate = librosa.load(path)
    _compute_embedding(samples)

audio_file_path = "./voices/sundarPichai.wav"
read_audio_file(audio_file_path)
print("Embedding Loads Sucessfully")


# In[8]:



def clone_voice(text):
    
    def synthesize(embed, text):
        print("Synthesizing new audio...")
        #with io.capture_output() as captured:
        specs = synthesizer.synthesize_spectrograms([text], [embed])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        print(type(generated_wav))
        return 
#         display(Audio(generated_wav, rate=synthesizer.sample_rate, autoplay=True))

    if embedding is None:
        print("first record a voice or upload a voice file!")
    else:
        synthesize(embedding, text)
        print("Voice Clonned Sucessfully")
        
text = "I am bhola record I am here to see you in the middle of the earth hello there" #@param {type:"string"}
clone_voice(text)


# In[ ]:





# In[4]:


from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import os
import sys
sys.path.append("./Deepfake/Wav2Lip-master/")
checkpoint_path = "./Deepfake/Wav2Lip-master/weights/wav2lip_gan.pth"
video_path = "./Deepfake/Wav2Lip-master/samples/WhatsApp Video 2020-08-30 at 3.52.46 PM (1).mp4"
audio_path = "./Deepfake/Wav2Lip-master/samples/sundarPichai.wav"
infrence_path = "./Deepfake/Wav2Lip-master/inference.py"
img_size = 96
static = False
fps = 25
pads = [0, 10, 0, 0]
face_det_batch_size = 16
wav2lip_batch_size = 128
resize_factor = 1
outfile = "./Deepfake/Wav2Lip-master/results/result_voice.mp4"



if os.path.isfile(video_path) and video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
	static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = pads
	for rect, image in zip(predictions, images):
		if rect is None:
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = get_smoothened_boxes(np.array(results), T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if not static:
		face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
	else:
		face_det_results = face_detect([frames[0]])

	for i, m in enumerate(mels):
		idx = 0 if static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (img_size, img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

        
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
print(f"Device Available {device}")
def _load(checkpoint_path):
	checkpoint = torch.load(checkpoint_path)
# 	if device == 'cuda':
# 		print("Model Load in GPU")
# 		checkpoint = torch.load(checkpoint_path)
# 	else:
# 		print("Model Load in CPU")
# 		checkpoint = torch.load(checkpoint_path,
# 								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def main(audio_path):
	if not os.path.isfile(video_path):
		fnames = list(glob(os.path.join(video_path, '*.jpg')))
		sorted_fnames = sorted(fnames, key=lambda f: int(os.path.basename(f).split('.')[0]))
		full_frames = [cv2.imread(f) for f in sorted_fnames]

	elif video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(video_path)]
		fps = fps

	else:
		video_stream = cv2.VideoCapture(video_path)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	if not audio_path.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		audio_path = 'temp/temp.wav'

	wav = audio.load_wav(audio_path, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi', 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', outfile)
	subprocess.call(command, shell=True)


main(audio_path)


# In[6]:


pip freeze >> requirements.txt


# In[ ]:




