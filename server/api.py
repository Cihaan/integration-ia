from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from loguru import logger
import uvicorn
import numpy as np
import pandas as pd

# Initialisation de FastAPI
app = FastAPI()

class NotePredictionData(BaseModel):
    ville: str
    surface: float
    price: float

# Définition d'un modèle pour les données d'entrée
class PredictionData(BaseModel):
    surface: float


# Définition d'un modèle pour les données d'entrée
class PredictionDataAppartement(BaseModel):
    surface: float
    nbRooms: float
    nbWindows: float
    price: float

# Initialisation du modèle de régression linéaire
model = LinearRegression()


# Initialisation du modèle de régression linéaire
modelSecond = LogisticRegression(max_iter=200)


# Initialisation du modèle de régression linéaire
modelThird = KNeighborsClassifier(n_neighbors=5)

# Utilisation cohérente de l'encodeur : L'encodeur label_encoder que vous avez utilisé pour transformer les catégories lors de l'entraînement doit être réutilisé pour inverser cette transformation lors de la prédiction.
label_encoder = LabelEncoder()

note_model = LinearRegression()
ville_encoder = None
ville_columns = None

# Variable pour vérifier si le modèle est entraîné
is_model_trained = False

# Endpoint pour entraîner le modèle

@app.post("/train")
async def train():
    global is_model_trained, model, modelSecond, modelThird, label_encoder

    # Lire le fichier CSV
    df = pd.read_csv('appartements.csv')

    # ... (le reste de votre code d'entraînement existant)

    # Ajout du nouveau modèle pour prédire la note
    X = df[['ville', 'surface', 'price']]
    y = df['note']

    # Encodage one-hot pour la variable catégorielle 'ville'
    X = pd.get_dummies(X, columns=['ville'])

    # Création et entraînement du modèle
    note_model = LinearRegression()
    note_model.fit(X, y)

    # Sauvegarder l'encodeur et les colonnes pour une utilisation ultérieure
    global ville_encoder, ville_columns
    ville_encoder = pd.get_dummies(df['ville'])
    ville_columns = ville_encoder.columns

    is_model_trained = True
    logger.info("Tous les modèles ont été entraînés avec succès.")

    return {"message": "Modèles entraînés avec succès."}

@app.post("/predict-note")
async def predict_note(data: NotePredictionData):
    global is_model_trained, note_model, ville_encoder, ville_columns

    if not is_model_trained:
        raise HTTPException(status_code=400, detail="Les modèles ne sont pas encore entraînés. Veuillez entraîner les modèles d'abord.")

    # Préparer les données d'entrée
    input_data = pd.DataFrame([[data.ville, data.surface, data.price]], columns=['ville', 'surface', 'price'])
    
    # Encoder la ville
    ville_encoded = pd.get_dummies(input_data['ville'])
    
    # S'assurer que toutes les colonnes de ville sont présentes
    for col in ville_columns:
        if col not in ville_encoded.columns:
            ville_encoded[col] = 0

    # Réorganiser les colonnes pour correspondre à l'ordre d'entraînement
    ville_encoded = ville_encoded[ville_columns]

    # Combiner les données encodées
    X_new = pd.concat([ville_encoded, input_data[['surface', 'price']]], axis=1)

    # Prédire la note
    predicted_note = note_model.predict(X_new)[0]

    logger.info(f"Prédiction de note faite pour ville: {data.ville}, surface: {data.surface}, prix: {data.price}")
    logger.info(f"Note prédite: {predicted_note}")

    return {"predicted_note": predicted_note}
