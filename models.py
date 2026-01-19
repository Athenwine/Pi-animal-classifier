"""
Database Models for Animal Classifier with Authentication and Activity Tracking
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model with role-based access"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')  # student, teacher, admin
    full_name = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    activities = db.relationship('Activity', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    sessions = db.relationship('UserSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)

    def is_teacher(self):
        """Check if user is a teacher"""
        return self.role in ['teacher', 'admin']

    def is_admin(self):
        """Check if user is an admin"""
        return self.role == 'admin'

    def get_activity_count(self):
        """Get total number of classifications"""
        return self.activities.count()

    def get_accuracy_rate(self):
        """Calculate average confidence across all activities"""
        activities = self.activities.all()
        if not activities:
            return 0.0
        total_confidence = sum(a.confidence for a in activities if a.confidence)
        return total_confidence / len(activities) if activities else 0.0

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'activity_count': self.get_activity_count(),
            'accuracy_rate': self.get_accuracy_rate()
        }


class Activity(db.Model):
    """Activity log for each classification"""
    __tablename__ = 'activities'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Classification results
    predicted_species = db.Column(db.String(50), nullable=False)
    predicted_category = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float)  # Species confidence
    category_confidence = db.Column(db.Float)

    # Image metadata
    image_source = db.Column(db.String(20))  # 'upload' or 'webcam'
    image_path = db.Column(db.String(255))  # Optional: store image path

    # Session info
    session_id = db.Column(db.Integer, db.ForeignKey('user_sessions.id'), index=True)
    duration_ms = db.Column(db.Integer)  # Processing time

    # Additional predictions (JSON string of top 5)
    all_predictions = db.Column(db.Text)  # JSON string

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'predicted_species': self.predicted_species,
            'predicted_category': self.predicted_category,
            'confidence': self.confidence,
            'category_confidence': self.category_confidence,
            'image_source': self.image_source,
            'duration_ms': self.duration_ms
        }


class UserSession(db.Model):
    """User session tracking for analytics"""
    __tablename__ = 'user_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    session_type = db.Column(db.String(20))  # 'upload', 'webcam', 'dashboard'
    ip_address = db.Column(db.String(50))
    user_agent = db.Column(db.String(255))

    # Relationships
    activities = db.relationship('Activity', backref='session', lazy='dynamic')

    def get_duration(self):
        """Get session duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'session_type': self.session_type,
            'duration': self.get_duration(),
            'activity_count': self.activities.count()
        }


class SystemConfig(db.Model):
    """System configuration and settings"""
    __tablename__ = 'system_config'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False, index=True)
    value = db.Column(db.Text)
    description = db.Column(db.String(255))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @staticmethod
    def get_value(key, default=None):
        """Get configuration value"""
        config = SystemConfig.query.filter_by(key=key).first()
        return config.value if config else default

    @staticmethod
    def set_value(key, value, description=None):
        """Set configuration value"""
        config = SystemConfig.query.filter_by(key=key).first()
        if config:
            config.value = value
            if description:
                config.description = description
        else:
            config = SystemConfig(key=key, value=value, description=description)
            db.session.add(config)
        db.session.commit()
