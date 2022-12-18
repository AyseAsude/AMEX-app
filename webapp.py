import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gsheetsdb import connect
import tensorflow as tf
import pickle

conn = connect()

def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

sheet_url = st.secrets["public_gsheets_url"]
rows = run_query(f'SELECT * FROM "{sheet_url}"')
cnn_df = pd.DataFrame(rows)


file = open("tab2imgconverter.pkl", "rb")
unpickler = pickle.Unpickler(file)
converter = unpickler.load()
file.close()

cnn_model =  tf.keras.models.load_model("cnn_model.h5")

def user_input_features():
    B_1_mean = st.sidebar.slider('B_1_mean', -1.0, 1.0, 0.5)
    B_1_min = st.sidebar.slider('B_1_min', 0.0, 1.0, 0.6)
    B_10_mean = st.sidebar.slider('B_10_mean', -1.5, 5.0, 0.2)
    D_107_mean = st.sidebar.slider('D_107_mean', 0.0, 5.0, 0.2)
    D_133_max = st.sidebar.slider('D_133_max', 0.0, 1.5, 0.05)
    return B_1_mean, B_1_min, B_10_mean, D_107_mean, D_133_max


B_1_mean, B_1_min , B_10_mean, D_107_mean, D_133_max = user_input_features()
cnn_df["B_1_mean"] = B_1_mean
cnn_df["B_1_min"] = B_1_min
cnn_df["B_10_mean"] = B_10_mean
cnn_df["D_107_mean"] = D_107_mean
cnn_df["D_133_max"] = D_133_max

st.write(cnn_df)

st.header("Image")
image = converter.transform(cnn_df.values).reshape(22,22,1)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(image, cmap = "gray")
st.pyplot(fig)

input = np.array(image)
input = input.reshape(1, 22, 22)
st.header('Prediction')
prediction = cnn_model.predict(input)
st.write(prediction)