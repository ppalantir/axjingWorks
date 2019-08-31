import os
import glob
import zipfile
import functools

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import pandas as pd 
from PIL import Image

import tensorflow as tf 
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
import kaggle

#Upload the API token
def get_kaggle_credentials():
    token_dir = os.path.join(os.path.expanduser("~"), "kaggle")
    token_file = os.path.join(token_dir, "kaggle.json")
    if not os.path.isdir(token_dir):
        os.mkdir(token_dir)
    try:
        with open(token_file, 'r') as f:
            pass
    except IOError as files:
        try:
            from google.colab import files
        except ImportError:
            raise

        uploaded = files.upload()
        if "kaggle.json" not in uploaded:
            raise ValueError("You need an API key! see:" "https://github.com/Kaggle/kaggle-api#api-credentials")
        with open(token_file, "wb") as f:
            f.write(uploaded["kaggle.json"])
        os.chmod(token_file, 600)
        
#get_kaggle_credentials()

competition_name = 'carvana-image-masking-challenge'

# Download data from Kaggle and unzip teh files of interest
def load_data_from_zip(competition, file):
    with zipfile.ZipFile(os.path.join(competition, file), "r") as zip_ref:
        unzipped_file = zip_ref.namelist()[0]
        zip_ref.extractall(competition)

def get_data(competition):
    kaggle.api.competition_download_files(competition, competition)
    load_data_from_zip(competition, 'train.zip')
    load_data_from_zip(competition, 'train_masks.zip')
    load_data_from_zip(competition, 'train_masks.csv.zip')

get_data(competition_name)    