import pickle
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib

# # Get the current working directory
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # File name
# file_name = 'SVM_RBF_85_ACCURACY.pkl'

# # Create the full file path
# file_path = os.path.join(current_directory, file_name)

class AudioClassifier:
    def __init__(self, file_path):
        # Load the pre-trained model 
        self.model = joblib.load(file_path)
        print("Constructor called !")

        


    def extract_features(self, audio_file):
        y, sr = librosa.load(audio_file)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = self.scaler.transform(mfccs.T)

        return mfccs

    def predict(self, audio_file):
        features = self.extract_features(audio_file)
        predicted_label = self.model.predict(features)

        return predicted_label
