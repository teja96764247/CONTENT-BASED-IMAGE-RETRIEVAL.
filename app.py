import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image,ImageOps
from numpy.linalg import norm
import tensorflow
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
import pickle

feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.markdown("<h2 style='text-align: center;'>CBIR PROJECT</h2>", unsafe_allow_html=True)







file = st.file_uploader("Please upload a fashion image", type=['jpg', 'png'])

if file is None:
    st.text('Please upload an image file')
else:
    size=(224,224)
    img=Image.open(file)
    st.image(img,use_column_width=True)
    img_path=ImageOps.fit(img,size,Image.ANTIALIAS)
    img_arr=np.array(img_path)
    extend_img=np.expand_dims(img_arr,axis=0)
    pre_process_img=preprocess_input(extend_img)
    result=model.predict(pre_process_img).flatten()
    normalized=result/norm(result)
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([normalized])
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(filenames[indices[0][0]])
    with col2:
        st.image(filenames[indices[0][1]])
    with col3:
        st.image(filenames[indices[0][2]])
    with col4:
        st.image(filenames[indices[0][3]])
    with col5:
        st.image(filenames[indices[0][4]])



