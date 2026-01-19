"""
Enhanced Hierarchical Animal Classification Web Application
With Authentication, Webcam Capture, and Analytics Dashboard
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session as flask_session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json
import os
import base64
from datetime import datetime, timedelta
from models import db, User, Activity, UserSession, SystemConfig
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///animal_classifier.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'

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
    print("⚠ Running without model - please train the model first")
    model = None

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

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Role-based access decorators
def teacher_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_teacher():
            flash('Accès réservé aux enseignants.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin():
            flash('Accès réservé aux administrateurs.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def preprocess_image(image):
    """Preprocess image for model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(image):
    """Make hierarchical prediction"""
    if model is None:
        raise Exception("Model not loaded. Please train the model first.")

    start_time = datetime.now()
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

    duration = (datetime.now() - start_time).total_seconds() * 1000
    return results, duration

def log_activity(user_id, predictions, image_source='upload', session_id=None, duration_ms=0):
    """Log classification activity"""
    if not predictions:
        return None

    top_prediction = predictions[0]
    activity = Activity(
        user_id=user_id,
        predicted_species=top_prediction['species'],
        predicted_category=top_prediction['category'],
        confidence=top_prediction['species_confidence'],
        category_confidence=top_prediction['category_confidence'],
        image_source=image_source,
        session_id=session_id,
        duration_ms=int(duration_ms),
        all_predictions=json.dumps(predictions)
    )
    db.session.add(activity)
    db.session.commit()
    return activity

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password) and user.is_active:
            login_user(user, remember=data.get('remember', False))
            user.last_login = datetime.utcnow()
            db.session.commit()

            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Nom d\'utilisateur ou mot de passe incorrect.', 'error')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name')

        # Validation
        if User.query.filter_by(username=username).first():
            flash('Ce nom d\'utilisateur existe déjà.', 'error')
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash('Cet email est déjà utilisé.', 'error')
            return redirect(url_for('signup'))

        # Create new user (default role: student)
        user = User(username=username, email=email, full_name=full_name, role='student')
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Compte créé avec succès! Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Vous avez été déconnecté.', 'success')
    return redirect(url_for('login'))

# Main routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard - role-based redirect"""
    if current_user.is_teacher():
        return redirect(url_for('teacher_dashboard'))
    return render_template('student_dashboard.html', user=current_user)

@app.route('/classifier')
@login_required
def classifier():
    """Classifier page with upload and webcam options"""
    return render_template('classifier.html', user=current_user)

# Prediction routes
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image prediction from upload"""
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
        results, duration = predict_image(image)

        # Log activity
        log_activity(current_user.id, results, image_source='upload', duration_ms=duration)

        # Convert to base64
        img_str = base64.b64encode(image_bytes).decode()

        return jsonify({
            'success': True,
            'predictions': results,
            'image': f"data:image/jpeg;base64,{img_str}",
            'duration_ms': duration
        })

    except Exception as e:
        import traceback
        print("Error during prediction:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict_webcam', methods=['POST'])
@login_required
def predict_webcam():
    """Handle real-time webcam frame prediction"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Predict
        results, duration = predict_image(image)

        # Log activity (optional - might log too many frames)
        should_log = data.get('log', False)
        if should_log:
            log_activity(current_user.id, results, image_source='webcam', duration_ms=duration)

        return jsonify({
            'success': True,
            'predictions': results,
            'duration_ms': duration
        })

    except Exception as e:
        import traceback
        print("Error during webcam prediction:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Teacher Dashboard Routes
@app.route('/teacher/dashboard')
@teacher_required
def teacher_dashboard():
    """Teacher dashboard with student activity monitoring"""
    # Get statistics
    total_students = User.query.filter_by(role='student').count()
    total_activities = Activity.query.count()

    # Recent activities (last 50)
    recent_activities = Activity.query.order_by(Activity.timestamp.desc()).limit(50).all()

    # Get all students with their stats
    students = User.query.filter_by(role='student').all()
    student_stats = []
    for student in students:
        stats = {
            'id': student.id,
            'username': student.username,
            'full_name': student.full_name,
            'email': student.email,
            'activity_count': student.get_activity_count(),
            'accuracy_rate': student.get_accuracy_rate(),
            'last_login': student.last_login
        }
        student_stats.append(stats)

    return render_template('teacher_dashboard.html',
                         user=current_user,
                         total_students=total_students,
                         total_activities=total_activities,
                         recent_activities=recent_activities,
                         students=student_stats)

@app.route('/teacher/student/<int:student_id>')
@teacher_required
def student_detail(student_id):
    """View detailed student activity"""
    student = User.query.get_or_404(student_id)
    if student.role != 'student':
        flash('Utilisateur invalide.', 'error')
        return redirect(url_for('teacher_dashboard'))

    activities = student.activities.order_by(Activity.timestamp.desc()).all()

    return render_template('student_detail.html',
                         user=current_user,
                         student=student,
                         activities=activities)

# Admin Panel Routes
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin panel for system management"""
    total_users = User.query.count()
    total_activities = Activity.query.count()
    total_sessions = UserSession.query.count()

    users = User.query.order_by(User.created_at.desc()).all()

    return render_template('admin_dashboard.html',
                         user=current_user,
                         total_users=total_users,
                         total_activities=total_activities,
                         total_sessions=total_sessions,
                         users=users)

@app.route('/admin/user/<int:user_id>/toggle_active', methods=['POST'])
@admin_required
def toggle_user_active(user_id):
    """Toggle user active status"""
    user = User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    db.session.commit()

    status = "activé" if user.is_active else "désactivé"
    flash(f'Utilisateur {user.username} {status}.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/user/<int:user_id>/change_role', methods=['POST'])
@admin_required
def change_user_role(user_id):
    """Change user role"""
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')

    if new_role in ['student', 'teacher', 'admin']:
        user.role = new_role
        db.session.commit()
        flash(f'Rôle de {user.username} changé en {new_role}.', 'success')
    else:
        flash('Rôle invalide.', 'error')

    return redirect(url_for('admin_dashboard'))

# API Routes for Analytics
@app.route('/api/analytics/overview')
@login_required
def analytics_overview():
    """Get overview analytics for current user"""
    user_id = current_user.id

    # Get activity count by category
    activities = Activity.query.filter_by(user_id=user_id).all()

    category_counts = {}
    species_counts = {}

    for activity in activities:
        category_counts[activity.predicted_category] = category_counts.get(activity.predicted_category, 0) + 1
        species_counts[activity.predicted_species] = species_counts.get(activity.predicted_species, 0) + 1

    # Get recent activity (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_count = Activity.query.filter(
        Activity.user_id == user_id,
        Activity.timestamp >= seven_days_ago
    ).count()

    return jsonify({
        'total_classifications': len(activities),
        'recent_classifications': recent_count,
        'average_confidence': current_user.get_accuracy_rate(),
        'category_distribution': category_counts,
        'species_distribution': species_counts,
        'favorite_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
    })

@app.route('/api/analytics/history')
@login_required
def analytics_history():
    """Get classification history"""
    user_id = current_user.id
    limit = request.args.get('limit', 100, type=int)

    activities = Activity.query.filter_by(user_id=user_id)\
        .order_by(Activity.timestamp.desc())\
        .limit(limit)\
        .all()

    return jsonify({
        'activities': [a.to_dict() for a in activities]
    })

@app.route('/info')
def info():
    """Model info"""
    return jsonify({
        'total_species': len(id_to_species),
        'total_categories': len(id_to_category),
        'model_type': 'Hierarchical Classification',
        'categories': list(mappings['category_to_id'].keys())
    })

# Database initialization
def init_db():
    """Initialize database with default admin user"""
    with app.app_context():
        db.create_all()

        # Create default admin if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@animal-classifier.com',
                full_name='Administrator',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("✓ Default admin user created (username: admin, password: admin123)")

if __name__ == '__main__':
    # Initialize database
    init_db()

    # Create folders
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    print("\n" + "="*60)
    print("🦁 Enhanced Animal Classifier Web App")
    print("="*60)
    print(f"Species classes: {len(id_to_species)}")
    print(f"Category classes: {len(id_to_category)}")
    print("\n Features:")
    print("  ✓ User Authentication (Students, Teachers, Admin)")
    print("  ✓ Webcam Real-time Classification")
    print("  ✓ Activity Logging & Progress Tracking")
    print("  ✓ Teacher Dashboard with Student Monitoring")
    print("  ✓ Admin Panel for User Management")
    print("\nStarting server at http://127.0.0.1:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
