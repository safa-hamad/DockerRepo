
from flask import Flask, request, jsonify,render_template
import requests
import os
import pickle
import joblib

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

import io


app = Flask(__name__,static_url_path='/static')

def extract_features(audio_file,sampling_rates):
     x, sr = librosa.load(audio_file,sr=sampling_rates)
     onset_env = librosa.onset.onset_strength(y=x, sr=sr)

     tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)


     features = {
    "chroma_stft_mean": float(librosa.feature.chroma_stft(y=x, sr=sr).mean()),
    "chroma_stft_var": float(librosa.feature.chroma_stft(y=x, sr=sr).var()),
    "rms_mean": float(librosa.feature.rms(y=x).mean()),
    "rms_var": float(librosa.feature.rms(y=x).var()),
    "spectral_centroid_mean": float(librosa.feature.spectral_centroid(y=x, sr=sr).mean()),
    "spectral_centroid_var": float(librosa.feature.spectral_centroid(y=x, sr=sr).var()),
    "spectral_bandwidth_mean":float( librosa.feature.spectral_bandwidth(y=x, sr=sr).mean()),
    "spectral_bandwidth_var": float(librosa.feature.spectral_bandwidth(y=x, sr=sr).var()),
    "rolloff_mean": float(librosa.feature.spectral_rolloff(y=x, sr=sr).mean()),
    "rolloff_var": float(librosa.feature.spectral_rolloff(y=x, sr=sr).var()),
    "zero_crossing_rate_mean": float(librosa.feature.zero_crossing_rate(y=x).mean()),
    "zero_crossing_rate_var": float(librosa.feature.zero_crossing_rate(y=x).var()),
    "harmony_mean": float(librosa.effects.harmonic(y=x).mean()),
    "harmony_var": float(librosa.effects.harmonic(y=x).var()),
    "tempo":float(tempo[0])
     }
  # Extract the MFCC features.
     mfccs = librosa.feature.mfcc(y=x, sr=sr)
     for i in range(1, 21):
          features["mfcc{}_mean".format(i)] =float( mfccs[i - 1].mean())
          features["mfcc{}_var".format(i)] = float(mfccs[i - 1].var())

     print("Feature Values:")
     for value in features.values():
         print(value)


     return [list(features.values())]

@app.route('/')
def home():
     return render_template('form.html')

@app.route('/classify', methods=['POST'])
def classify_audio(): 
     if request.method == 'POST':
          audio_file=request.files['file']
          audio_data = audio_file.read()
          audio_io = io.BytesIO(audio_data)
          data_samples=extract_features(audio_io,None)
          selected_value = request.form['model_selected']
          
          if selected_value == "none" :
               return render_template('form.html')

          if selected_value == "svm" :
               svm_response = requests.post("http://svm_service:80/svm",json=data_samples)
               return jsonify(svm_response.text)  

          if selected_value == "vgg" :
               svm_response = requests.post("http://vgg19_service:80/vgg",json=data_samples)
               return jsonify(svm_response.text)  
          
          

        
       
if __name__ == '__main__':
          app.run(debug=True)
