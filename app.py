import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Загрузка обученной модели
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Загрузка масштабировщика (если использовался при обучении)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("🎵 Music Genre Predictor")
st.write("Введите аудиофичи трека для предсказания жанра.")

# Пример фичей: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                 'instrumentalness', 'liveness', 'valence', 'tempo']

# Создание формы для ввода фичей
with st.form("feature_form"):
    features = []
    for feature in feature_names:
        value = st.number_input(f"{feature.capitalize()}:", min_value=0.0, max_value=1.0, value=0.5)
        features.append(value)
    submit = st.form_submit_button("Предсказать жанр")

if submit:
    # Преобразование введенных данных в массив и масштабирование
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    # Предсказание жанра
    prediction = model.predict(input_scaled)
    st.success(f"Предсказанный жанр: **{prediction[0]}**")
