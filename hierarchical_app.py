"""
Hierarchical Animal Classification Web Application
Save this as: hierarchical_app.py
Predicts BOTH species (30 classes) AND category (6 classes)
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json
import os
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model and mappings
MODEL_PATH = 'hierarchical_animal_classifier.keras'
MAPPINGS_PATH = 'class_mappings.json'

print("Loading hierarchical model...")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss={
            'species_output': 'categorical_crossentropy',
            'category_output': 'categorical_crossentropy'
        },
        metrics=['accuracy']
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load mappings
print("Loading class mappings...")
with open(MAPPINGS_PATH, 'r', encoding='utf-8') as f:
    mappings = json.load(f)

id_to_species = mappings['id_to_species']
id_to_category = mappings['id_to_category']
species_to_category = mappings['species_to_category']

print(f"✓ Loaded {len(id_to_species)} species and {len(id_to_category)} categories")

IMG_SIZE = (224, 224)

# Animal info with emojis and descriptions
ANIMAL_INFO = {
    'Chat': {'emoji': '🐱', 'description': 'Mammifère domestique, compagnon populaire'},
    'Chien': {'emoji': '🐕', 'description': 'Meilleur ami de l\'homme, fidèle compagnon'},
    'Lion': {'emoji': '🦁', 'description': 'Roi des animaux, grand félin d\'Afrique'},
    'Tigre': {'emoji': '🐯', 'description': 'Plus grand félin rayé d\'Asie'},
    'Éléphant': {'emoji': '🐘', 'description': 'Plus grand mammifère terrestre'},
    'Girafe': {'emoji': '🦒', 'description': 'Animal le plus grand du monde'},
    'Zèbre': {'emoji': '🦓', 'description': 'Équidé rayé d\'Afrique'},
    'Singe': {'emoji': '🐵', 'description': 'Primate intelligent et agile'},
    'Ours': {'emoji': '🐻', 'description': 'Grand mammifère omnivore'},
    'Cheval': {'emoji': '🐴', 'description': 'Équidé domestique, rapide et puissant'},
    'Aigle': {'emoji': '🦅', 'description': 'Rapace majestueux, excellent chasseur'},
    'Perroquet': {'emoji': '🦜', 'description': 'Oiseau coloré capable d\'imiter des sons'},
    'Hibou': {'emoji': '🦉', 'description': 'Rapace nocturne aux grands yeux'},
    'Flamant rose': {'emoji': '🦩', 'description': 'Oiseau rose élégant des zones humides'},
    'Manchot': {'emoji': '🐧', 'description': 'Oiseau marin vivant dans l\'Antarctique'},
    'Canard': {'emoji': '🦆', 'description': 'Oiseau aquatique au bec plat'},
    'Requin': {'emoji': '🦈', 'description': 'Prédateur marin redoutable'},
    'Poisson-clown': {'emoji': '🐠', 'description': 'Petit poisson orange et blanc des récifs'},
    'Saumon': {'emoji': '🐟', 'description': 'Poisson migrateur apprécié en cuisine'},
    'Poisson rouge': {'emoji': '🐡', 'description': 'Poisson d\'ornement populaire'},
    'Crocodile': {'emoji': '🐊', 'description': 'Grand reptile aquatique carnivore'},
    'Serpent': {'emoji': '🐍', 'description': 'Reptile sans pattes, peut être venimeux'},
    'Tortue': {'emoji': '🐢', 'description': 'Reptile à carapace, symbole de longévité'},
    'Lézard': {'emoji': '🦎', 'description': 'Petit reptile agile à quatre pattes'},
    'Grenouille': {'emoji': '🐸', 'description': 'Amphibien sauteur vivant près de l\'eau'},
    'Salamandre': {'emoji': '🦎', 'description': 'Amphibien ressemblant à un lézard'},
    'Abeille': {'emoji': '🐝', 'description': 'Insecte pollinisateur produisant du miel'},
    'Papillon': {'emoji': '🦋', 'description': 'Insecte aux ailes colorées'},
    'Coccinelle': {'emoji': '🐞', 'description': 'Petit insecte rouge à points noirs'},
    'Fourmi': {'emoji': '🐜', 'description': 'Insecte social travaillant en colonie'}
}

def preprocess_image(image):
    """Preprocess image for model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(image):
    """Make hierarchical prediction"""
    processed_img = preprocess_image(image)
    
    # Get predictions
    predictions = model.predict(processed_img, verbose=0)
    species_pred = predictions['species_output'][0]
    category_pred = predictions['category_output'][0]
    
    # Get top 5 species predictions
    top_species_indices = np.argsort(species_pred)[-5:][::-1]
    
    results = []
    for idx in top_species_indices:
        species_name = id_to_species.get(str(idx), "Unknown")
        species_confidence = float(species_pred[idx])
        
        # Get category for this species
        category_name = species_to_category.get(species_name, "Unknown")
        category_id = str(mappings['category_to_id'].get(category_name, 0))
        category_confidence = float(category_pred[int(category_id)])
        
        # Get animal info
        info = ANIMAL_INFO.get(species_name, {'emoji': '🐾', 'description': 'Information non disponible'})
        
        results.append({
            'species': species_name,
            'species_confidence': species_confidence,
            'species_confidence_percent': f"{species_confidence * 100:.1f}%",
            'category': category_name,
            'category_confidence': category_confidence,
            'category_confidence_percent': f"{category_confidence * 100:.1f}%",
            'emoji': info['emoji'],
            'description': info['description']
        })
    
    return results

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Predict
        results = predict_image(image)
        
        # Convert to base64
        img_str = base64.b64encode(image_bytes).decode()
        
        return jsonify({
            'success': True,
            'predictions': results,
            'image': f"data:image/jpeg;base64,{img_str}"
        })
    
    except Exception as e:
        import traceback
        print("Error during prediction:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Model info"""
    return jsonify({
        'total_species': len(id_to_species),
        'total_categories': len(id_to_category),
        'model_type': 'Hierarchical Classification',
        'categories': list(mappings['category_to_id'].keys())
    })

if __name__ == '__main__':
    # Create folders
    try:
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
    except:
        pass
    
    print("\n" + "="*60)
    print("🦁 Hierarchical Animal Classifier Web App")
    print("="*60)
    print(f"Species classes: {len(id_to_species)}")
    print(f"Category classes: {len(id_to_category)}")
    print("\nStarting server at http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
