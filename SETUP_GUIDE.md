# 🦁 Enhanced Animal Classifier - Setup Guide

## Overview

This enhanced version includes:
- ✅ User authentication (Students, Teachers, Admin)
- ✅ Real-time webcam classification
- ✅ Activity logging and progress tracking
- ✅ Teacher dashboard for student monitoring
- ✅ Admin panel for user management
- ✅ Analytics and visualizations

## Prerequisites

- Python 3.8 or higher
- Webcam (for real-time classification feature)
- Trained model file: `hierarchical_animal_classifier.keras`
- SQLite (included with Python)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Athenwine/Pi-animal-classifier.git
cd Pi-animal-classifier
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train or Download the Model

**Option A: Train the model (recommended)**
1. Open `V2.ipynb` in Google Colab
2. Download the dataset from the link in README
3. Run all cells to train the model
4. Download `hierarchical_animal_classifier.keras`
5. Place it in the project root directory

**Option B: Use existing model**
- If you have the model file, place it in the project root

### 5. Initialize Database

The database will be automatically created when you first run the app.

```bash
python app.py
```

On first run, a default admin account is created:
- **Username:** `admin`
- **Password:** `admin123`

**⚠️ IMPORTANT:** Change the admin password after first login!

### 6. Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## User Roles

### Student (Default)
- Upload images for classification
- Use webcam for real-time classification
- View personal statistics and activity history
- Track progress and accuracy

### Teacher
- All student features
- View all students' activity
- Monitor student progress
- Access detailed student analytics
- View recent activity feed

### Admin
- All teacher features
- Manage user accounts
- Change user roles
- Activate/deactivate users
- View system-wide statistics

## Creating User Accounts

### Via Web Interface (Recommended)
1. Go to `http://127.0.0.1:5000/signup`
2. Fill in the registration form
3. New users are created as "students" by default
4. Admins can change roles via the admin panel

### Via Python Console (Advanced)

```python
from app import app, db
from models import User

with app.app_context():
    # Create a teacher account
    teacher = User(
        username='teacher1',
        email='teacher@example.com',
        full_name='Teacher Name',
        role='teacher'
    )
    teacher.set_password('password123')
    db.session.add(teacher)
    db.session.commit()
```

## Features Guide

### 1. Image Upload Classification
- Navigate to "Classifier" from dashboard
- Click "Upload Image" mode
- Drag & drop or select an image
- Click "Classifier l'animal"
- View top 5 predictions with confidence scores

### 2. Webcam Real-Time Classification
- Navigate to "Classifier" from dashboard
- Click "Webcam en direct" mode
- Click "Démarrer la caméra"
- Grant camera permissions
- Real-time predictions appear as overlay
- View FPS and accuracy statistics

### 3. Student Dashboard
- View total classifications
- See average confidence
- Track weekly activity
- View category distribution chart
- Browse recent activity history

### 4. Teacher Dashboard
- View all students with statistics
- Monitor student accuracy and activity count
- View recent classifications from all students
- Click "Voir détails" to see individual student history
- Track student engagement

### 5. Admin Panel
- Manage all users
- Change user roles (Student/Teacher/Admin)
- Activate/deactivate accounts
- View system-wide statistics
- Monitor total activities and sessions

## Database Schema

### Users Table
- id, username, email, password_hash
- role (student/teacher/admin)
- full_name, created_at, last_login
- is_active (boolean)

### Activities Table
- id, user_id, timestamp
- predicted_species, predicted_category
- confidence, category_confidence
- image_source (upload/webcam)
- duration_ms, all_predictions (JSON)

### UserSessions Table
- id, user_id, start_time, end_time
- session_type, ip_address, user_agent

## API Endpoints

### Authentication
- `GET/POST /login` - User login
- `GET/POST /signup` - User registration
- `GET /logout` - User logout

### Main Routes
- `GET /` - Landing page
- `GET /dashboard` - Role-based dashboard redirect
- `GET /classifier` - Classifier page (login required)

### Classification
- `POST /predict` - Upload image classification
- `POST /predict_webcam` - Real-time webcam frame classification

### Teacher Routes
- `GET /teacher/dashboard` - Teacher dashboard
- `GET /teacher/student/<id>` - Student detail view

### Admin Routes
- `GET /admin/dashboard` - Admin panel
- `POST /admin/user/<id>/toggle_active` - Toggle user active status
- `POST /admin/user/<id>/change_role` - Change user role

### Analytics API
- `GET /api/analytics/overview` - User analytics overview
- `GET /api/analytics/history` - Classification history
- `GET /info` - Model information

## Configuration

### Environment Variables

Create a `.env` file (optional):

```env
SECRET_KEY=your-secret-key-here
DATABASE_URI=sqlite:///animal_classifier.db
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
```

### Database Location

By default, the SQLite database is stored as:
```
animal_classifier.db
```

## Troubleshooting

### Model not found
**Error:** "Model not loaded. Please train the model first."

**Solution:** Ensure `hierarchical_animal_classifier.keras` is in the project root directory.

### Webcam not working
**Error:** Camera access denied

**Solution:**
1. Check browser permissions
2. Use HTTPS or localhost
3. Ensure no other app is using the webcam

### Database locked
**Error:** "database is locked"

**Solution:**
1. Close any other connections to the database
2. Restart the application
3. Delete `animal_classifier.db` to recreate (WARNING: loses all data)

### Import errors
**Error:** "No module named 'flask_sqlalchemy'"

**Solution:**
```bash
pip install -r requirements.txt
```

## Development

### Running in Debug Mode

The app runs in debug mode by default:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Running in Production

For production, use a proper WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Database Migrations

For schema changes:

```python
from app import app, db

with app.app_context():
    db.create_all()  # Creates new tables
    # Use Alembic for complex migrations
```

## Security Considerations

1. **Change Default Admin Password** immediately after first login
2. **Use HTTPS** in production
3. **Set Strong SECRET_KEY** in production
4. **Regular Backups** of the database
5. **Limit File Upload Size** (default: 16MB)
6. **Sanitize User Input** (already implemented)
7. **Use Environment Variables** for sensitive data

## Performance Tips

1. **Webcam Frame Rate**: Adjust processing interval in `classifier.html` (default: 500ms)
2. **Activity Logging**: Webcam logs every 10th frame by default
3. **Database Indexing**: User_id and timestamp are indexed
4. **Chart.js**: Analytics use Chart.js for performance

## Backup and Restore

### Backup Database

```bash
# Simple backup
cp animal_classifier.db animal_classifier_backup.db

# With timestamp
cp animal_classifier.db "backup_$(date +%Y%m%d_%H%M%S).db"
```

### Restore Database

```bash
cp animal_classifier_backup.db animal_classifier.db
```

## Credits

**Original Project:** ESPRIT School of Business Team
- Aws Ourari - Lead Developer & ML Engineer
- Nairi Najla - Data Scientist
- Ameni Amina - UI/UX Designer
- Ines Jaziri - Data Collection

**Enhanced Features:** Added authentication, webcam, analytics, and multi-user support

## License

Educational project for ESPRIT School of Business

## Support

For issues or questions:
- Check the troubleshooting section
- Review the README.md
- Contact: awsourari123@gmail.com
