import streamlit as st
import os
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from groq import Groq
import time


# Descargas necesarias de NLTK
if not os.path.exists(os.path.join(nltk.data.path[0], 'corpora/stopwords.zip')):
    nltk.download('stopwords')
if not os.path.exists(os.path.join(nltk.data.path[0], 'corpora/wordnet.zip')):
    nltk.download('wordnet')
if not os.path.exists(os.path.join(nltk.data.path[0], 'tokenizers/punkt')):
    nltk.download('punkt')
if not os.path.exists(os.path.join(nltk.data.path[0], 'corpora/omw-1.4.zip')):
    nltk.download('omw-1.4')

# Cargar el Dataset
def cargar_dataset(url):
    """Carga el dataset desde una URL"""
    return pd.read_excel(url)

df_supplements = cargar_dataset('https://github.com/Dantelarroy/suplements-chatbot/raw/main/Dataset.xlsx')

# Preprocesamiento de Texto
lemmatizer = WordNetLemmatizer()

def clean(texto):
    """Limpia y preprocesa el texto"""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúüñ0-9\s]+', '', texto)  # Eliminar puntuación y caracteres especiales
    tokens = [word for word in texto.split() if word not in stopwords.words('spanish')]  # Eliminar stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lematización
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Aplicar la función de limpieza al DataFrame
df_supplements['Texto_Limpio'] = df_supplements['Texto'].apply(clean)

# Feature Engineering - Embedding Models
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

def get_embedding(text):
    """Obtiene el embedding de un texto dado"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

# Aplicar la función de embeddings al DataFrame
df_supplements['Embedding'] = df_supplements['Texto_Limpio'].apply(get_embedding)

# Visualización 2D de los embeddings usando TSNE
embeddings = np.vstack(df_supplements['Embedding'])
label_encoder = LabelEncoder()
df_supplements['Sentimiento'] = label_encoder.fit_transform(df_supplements['Sentimiento'])

# Preparo los datos para el modelo
X = np.vstack(df_supplements['Embedding'].values)
y = df_supplements[["Sentimiento"]].values.ravel()  # Convertir a vector unidimensional

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de clasificación (Regresión logística)
model_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
model_lr.fit(X_train, y_train)

# Evaluar el modelo
y_pred_lr = model_lr.predict(X_test)

# K-Means clustering para identificación de temas
n_clusters = 5  # Número de clusters a identificar
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(embeddings)
df_supplements['Cluster'] = labels_kmeans

# Asignar tema al cluster
def get_topic(cluster_id):
    """Asigna un tema a un cluster dado"""
    topics = {
        0: "Atención al Cliente",
        1: "Sabor y Textura",
        2: "Calidad del Producto",
        3: "Precio del Producto",
        4: "Envío y Logística",
    }
    return topics.get(cluster_id, "Tema desconocido")

df_supplements['Tema'] = df_supplements['Cluster'].apply(get_topic)



# Maqueta de función principal para análisis crítico de reseñas
# Función de análisis de crítica adaptada para Gradio
def analisis_critica(reseña):
    ''' Esta función permite realizar el análisis de la reseña ingresada y devolver una respuesta acorde'''

    if not reseña:
        return "No se ha ingresado una reseña. Intentarlo nuevamente"
    
    # Inicializar el cliente de Groq
    client = Groq(api_key='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')

    # Mapear el sentimiento
    sentimiento_mapeo = {
        0: "Negativo",
        1: "Neutral",
        2: "Positivo"
    }

    try:
        # Limpieza del texto
        texto_limpio = clean(reseña)

        # Transformación a embeddings
        embeddings_final = get_embedding(texto_limpio).reshape(1, -1)

        # Predicción del sentimiento
        sentimiento_num = model_lr.predict(embeddings_final)[0]
        sentimiento = sentimiento_mapeo.get(sentimiento_num, "Desconocido")

        # Asignar un cluster
        cluster_id = kmeans.predict(embeddings_final)[0]

        # Obtener el tema del cluster
        tema = get_topic(cluster_id)

        # Generar respuesta personalizada con LLM
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": f"""
Eres un empleado de atención al cliente especializado en gestión de reseñas para una empresa de suplementos nutricionales.
Tu función es responder las reseñas según el {sentimiento} y {tema} detectados:
Responde de manera breve, profesional y empática:
- Si el {sentimiento} es **Positivo**: Agradece y celebra la experiencia positiva. Ejemplo: Gracias por sus comentarios. Nos alegra que haya sido una excelente experiencia.
- Si el {sentimiento} es **Negativo**: Muestra empatía y ofrece resolver el problema. Ejemplo: Gracias por sus comentarios. Lamentamos que su experiencia no fuera la mejor. Nos comunicaremos con usted para ofrecerle una respuesta personalizada.
- Si el {sentimiento} es **Neutral**: Agradece de forma clara e invita al cliente a comprar nuevamente. Ejemplo: Gracias por sus comentarios. Lo esperamos nuevamente cuando desee.

No incluyas preguntas, comentarios fuera de contexto, ni frases adicionales.
"""
            }, {
                "role": "user",
                "content": reseña
            }], 
            model="llama-3.2-1b-preview",
            temperature=0.1
        )

        # Retornar respuesta generada
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error al procesar la reseña: {str(e)}"


# Configurar la página con estilo y visualización
st.set_page_config(page_title="Supplements Reviews", page_icon="💬", layout="centered")

# Función para mostrar los mensajes como burbujas con animaciones
def display_message(message, is_user=False):
    """Función para mostrar los mensajes como burbujas con animaciones"""
    if is_user:
        st.markdown(f'''
            <div style="text-align: right; background: linear-gradient(145deg, #A1C4FD, #C2E9FB); 
            padding: 12px; border-radius: 25px; max-width: 60%; margin-left: auto; 
            margin-bottom: 12px; box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.1); animation: slide-right 0.5s;">
            {message}</div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div style="text-align: left; background: linear-gradient(145deg, #E0E4E9, #F2F3F5); 
            padding: 12px; border-radius: 25px; max-width: 60%; margin-right: auto; 
            margin-bottom: 12px; box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.1); animation: slide-left 0.5s;">
            {message}</div>
        ''', unsafe_allow_html=True)

# Estilos adicionales para animaciones y efectos visuales
st.markdown(""" 
    <style>
    @keyframes slide-right {
        0% { opacity: 0; transform: translateX(10px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slide-left {
        0% { opacity: 0; transform: translateX(-10px); }
        100% { opacity: 1; transform: translateX(0); }
    }

    h1 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        color: #333;
        margin-bottom: 10px;
    }

    .stProgress > div {
        border-radius: 10px;
        background-color: #76c7c0;
        height: 20px;
    }

    .stButton > button {
        background-color: #6c63ff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 30px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #5a53e5;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado del chatbot
st.title("🤖 Chatbot ")
st.markdown(""" 
    <h3 style="font-size: 18px; color: #555;">Envianos tu reseña y recibe una respuesta personalizada!🙌</h3>
""", unsafe_allow_html=True)

# Inicializar historial de chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Entrada de texto para el usuario con campo estilizado
with st.form(key='message_form', clear_on_submit=True):
    user_message = st.text_area("Escribe tu crítica:", key='user_message', height=100, placeholder="Escribe aquí tu crítica...", max_chars=500, label_visibility="collapsed")
    submit_button = st.form_submit_button(label='Enviar')

    if submit_button and user_message:
        # Mostrar mensaje del usuario
        st.session_state.messages.append({"message": user_message, "is_user": True})

        # Mostrar barra de progreso
        with st.spinner("Procesando tu reseña..."):
            time.sleep(1)

        # Obtener respuesta del chatbot (Simulación con ejemplo estático)
        response = analisis_critica(user_message)
        
        # Mostrar la respuesta completa al finalizar la simulación de tipeo
        st.session_state.messages.append({"message": response, "is_user": False})

# Mostrar el historial de chat
for msg in st.session_state.messages:
    display_message(msg['message'], msg['is_user'])