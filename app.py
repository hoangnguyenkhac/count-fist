from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
import os
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime
import bcrypt
from functools import wraps
import base64
from bson.objectid import ObjectId

app = Flask(__name__)
app.secret_key = "fish_counter_secret_key"

# Configure upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLO model
model = YOLO("models/best.pt")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["fishcounting"]
collection = db["historicalanalysis"]
users_collection = db["users"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    if 'user_id' in session:
        return redirect(url_for("analysis"))
    return render_template("home.html")

@app.route("/system")
def system():
    return render_template("system.html")

@app.route("/trial")
def trial():
    return render_template("trial.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # Process contact form submission
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        subject = request.form.get("subject", "")
        message = request.form.get("message", "")
        
        # Store in database or send email (placeholder for now)
        # You could add actual email sending functionality here
        
        flash("Thank you for your message! We'll get back to you soon.", "success")
        return redirect(url_for("contact"))
        
    return render_template("contact.html")

@app.route("/analysis")
@login_required
def analysis():
    user_id = session["user_id"]
    
    # Get statistics for the dashboard
    analyses = list(collection.find({"user_id": user_id}).sort("timestamp", -1))
    
    # Calculate statistics
    analyses_count = len(analyses)
    total_live_fish = sum(analysis.get('live_fish_count', 0) for analysis in analyses)
    total_dead_fish = sum(analysis.get('dead_fish_count', 0) for analysis in analyses)
    
    # Format date for display
    last_analysis_date = "No analyses yet"
    if analyses:
        last_analysis = analyses[0]
        last_analysis_date = last_analysis["timestamp"].strftime("%B %d, %Y %H:%M")
    
    # Get most recent analyses for dashboard display (limited to 5)
    recent_analyses = analyses[:5] if analyses else []
    
    return render_template(
        "index.html",
        analyses_count=analyses_count,
        total_live_fish=total_live_fish,
        total_dead_fish=total_dead_fish,
        last_analysis_date=last_analysis_date,
        recent_analyses=recent_analyses
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        # Validate form data
        if not username or not email or not password or not confirm_password:
            flash("All fields are required")
            return render_template("register.html")
        
        if password != confirm_password:
            flash("Passwords do not match")
            return render_template("register.html")
        
        # Check if username or email already exists
        if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            flash("Username or email already exists")
            return render_template("register.html")
        
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create a new user
        user = {
            "username": username,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now()
        }
        
        users_collection.insert_one(user)
        flash("Registration successful. Please log in.")
        return redirect(url_for("login"))
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = users_collection.find_one({"username": username})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            # Store user info in session
            session["user_id"] = str(user["_id"])
            session["username"] = user["username"]
            flash("Login successful")
            return redirect(url_for("analysis"))
        else:
            flash("Invalid username or password")
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out")
    return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
def upload_file():
    # Check if the image is from camera or file upload
    if "file" in request.files:
        # File upload
        file = request.files["file"]
        
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + "." + secure_filename(file.filename).rsplit(".", 1)[1].lower()
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Process the saved image
            return process_image(filepath, filename)
    
    elif "file" in request.form:
        # Camera capture (data URL)
        image_data = request.form.get("file")
        if image_data and image_data.startswith('data:image'):
            try:
                # Extract the base64 encoded image
                image_data = image_data.split(',')[1]
                
                # Decode the base64 image
                image_bytes = base64.b64decode(image_data)
                
                # Generate a unique filename
                filename = str(uuid.uuid4()) + ".jpg"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                
                # Save the image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                # Check if file was saved properly
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    flash("Error saving camera image. Please try again.")
                    return redirect(url_for('index'))
                
                # Process the saved image
                return process_image(filepath, filename)
            except Exception as e:
                app.logger.error(f"Error processing camera image: {str(e)}")
                flash(f"Error processing camera image: {str(e)}")
                return redirect(url_for('index'))
        else:
            flash("Invalid image data from camera. Please try again.")
            return redirect(url_for('index'))
    
    flash("Invalid file type. Please upload an image (png, jpg, jpeg).")
    return redirect(url_for('index'))

def process_image(filepath, filename):
    """Process an image file and detect fish"""
    # Process image and detect fish
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference with the model
    results = model(image, conf=0.25)  # Set confidence threshold
    
    # Get detection results
    result = results[0]
    detections = result.boxes
    
    # Extract detection details for visualization and summary
    detection_details = []
    live_fish_count = 0
    dead_fish_count = 0
    
    for box in detections:
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Class 0 is live fish, Class 1 is dead fish
        if cls == 0:
            live_fish_count += 1
            class_name = "Live Fish"
        elif cls == 1:
            dead_fish_count += 1
            class_name = "Dead Fish"
        else:
            continue
            
        # Get coordinates (xywh format)
        x, y, w, h = box.xywh[0]
        
        detection_details.append({
            'class': cls,
            'class_name': class_name,
            'confidence': round(conf * 100, 1),
            'box': [float(x), float(y), float(w), float(h)]
        })
    
    # Custom visualization with enhanced colors and labels
    annotated_image = result.plot(
        conf=True,             # Show confidence
        line_width=2,          # Line width of bounding boxes
        font_size=0.8,         # Font size
        labels=True,           # Show labels
        boxes=True,            # Show boxes
        font='Arial.ttf',      # Font
    )
    
    # Convert back to BGR for OpenCV
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Save the annotated image
    result_filename = "result_" + filename
    result_filepath = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
    cv2.imwrite(result_filepath, annotated_image)
    
    # Get image dimensions for UI
    height, width = image.shape[:2]
    
    # Save data to MongoDB
    analysis_record = {
        "timestamp": datetime.now(),
        "original_image": filename,
        "result_image": result_filename,
        "live_fish_count": live_fish_count,
        "dead_fish_count": dead_fish_count,
        "total_fish_count": live_fish_count + dead_fish_count,
        "detections": detection_details,
        "image_width": width,
        "image_height": height
    }
    
    # If user is logged in, associate the analysis with their account
    if 'user_id' in session:
        analysis_record["user_id"] = session["user_id"]
    
    collection.insert_one(analysis_record)
    
    return render_template(
        "result.html", 
        original_image=filename, 
        result_image=result_filename,
        live_fish_count=live_fish_count,
        dead_fish_count=dead_fish_count,
        total_fish_count=live_fish_count + dead_fish_count,
        detections=detection_details,
        image_width=width,
        image_height=height
    )

@app.route("/history")
@login_required
def history():
    # Get user's historical analyses from MongoDB
    user_id = session["user_id"]
    analyses = list(collection.find({"user_id": user_id}).sort("timestamp", -1))
    
    # Convert ObjectId to string for JSON serialization
    for analysis in analyses:
        analysis["_id"] = str(analysis["_id"])
    
    return render_template("history.html", analyses=analyses)

@app.route("/profile")
@login_required
def profile():
    user_id = session["user_id"]
    
    # Convert string ID to ObjectId
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    
    if not user:
        flash("User not found")
        return redirect(url_for("index"))
    
    # Get user's analyses to show statistics
    analyses = list(collection.find({"user_id": user_id}).sort("timestamp", -1))
    user['analyses'] = analyses
    
    return render_template("profile.html", user=user)

# Route for real-time camera fish counting (placeholder for beta feature)
@app.route("/realtime")
@login_required
def realtime():
    return render_template("realtime.html")

@app.route("/change-password", methods=["POST"])
@login_required
def change_password():
    user_id = session["user_id"]
    current_password = request.form.get("current_password")
    new_password = request.form.get("new_password")
    confirm_new_password = request.form.get("confirm_new_password")
    
    # Validate inputs
    if not current_password or not new_password or not confirm_new_password:
        flash("All fields are required")
        return redirect(url_for("profile"))
    
    if new_password != confirm_new_password:
        flash("New passwords do not match")
        return redirect(url_for("profile"))
    
    # Get user from database
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    
    if not user:
        flash("User not found")
        return redirect(url_for("profile"))
    
    # Verify current password
    if not bcrypt.checkpw(current_password.encode('utf-8'), user["password"]):
        flash("Current password is incorrect")
        return redirect(url_for("profile"))
    
    # Hash new password
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    
    # Update password in database
    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password": hashed_password}}
    )
    
    flash("Password changed successfully")
    return redirect(url_for("profile"))

if __name__ == "__main__":
    app.run(debug=True) 