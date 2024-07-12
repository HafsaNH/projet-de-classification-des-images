#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request, jsonify
import os
import cv2
import pickle
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Charger le modèle SVM et le scaler
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Fonction pour prétraiter une image
def preprocess_image(image_path, scaler):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (128, 64))
    feature = hog(image_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    feature_scaled = scaler.transform([feature])
    return feature_scaled

# Route pour la page d'accueil
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Vérifier si un fichier d'image est soumis
        if 'image' not in request.files:
            return render_template('index.html', message='Aucune image sélectionnée')
        
        file = request.files['image']

        # Vérifier si le fichier est vide
        if file.filename == '':
            return render_template('index.html', message='Aucune image sélectionnée')

        # Vérifier si le fichier est une image
        if file and file.filename.endswith(('.png', '.jpg', '.jpeg')):
            # Sauvegarder temporairement l'image soumise
            image_path = os.path.join(os.getcwd(), 'static', 'img', file.filename)
            file.save(image_path)

            # Prétraiter l'image
            preprocessed_features = preprocess_image(image_path, scaler)

            # Faire la prédiction avec le modèle SVM
            prediction = svm_model.predict(preprocessed_features)

            # Supprimer l'image temporaire après la prédiction
            os.remove(image_path)

            return render_template('index.html', prediction=str(prediction[0]))
        else:
            return render_template('index.html', message='Format d\'image non supporté')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




