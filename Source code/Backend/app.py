from flask import Flask, render_template, request, redirect, url_for, session
import os
import time
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

def compute_image_quality(img_2d: np.ndarray):
    """Simple heuristic quality check for grayscale image in range [0,1].
    Returns a quality label ('Good'|'Poor') and metrics dict.
    Metrics: std (contrast), grad_mean (average gradient magnitude).
    """
    # Ensure float
    img = img_2d.astype(float)
    std = float(np.std(img))

    # Gradient-based metric (edge strength)
    gx, gy = np.gradient(img)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(np.mean(grad))

    # Laplacian variance (simple blur detector) using 3x3 laplacian kernel
    # kernel: [[0,1,0],[1,-4,1],[0,1,0]]
    padded = np.pad(img, ((1, 1), (1, 1)), mode='reflect')
    lap = (
        padded[0:-2, 1:-1] + padded[1:-1, 0:-2] + padded[1:-1, 2:] + padded[2:, 1:-1]
        - 4 * padded[1:-1, 1:-1]
    )
    lap_var = float(np.var(lap))

    # Heuristic thresholds (tuned for normalized 128x128 images)
    std_thresh = 0.05        # contrast
    grad_thresh = 0.02       # mean gradient magnitude
    lap_var_thresh = 0.0008  # laplacian variance (blur detection)

    # Decide quality: require reasonable contrast AND (edges OR laplacian variance)
    is_good = (std >= std_thresh) and ((grad_mean >= grad_thresh) or (lap_var >= lap_var_thresh))
    quality = "Good" if is_good else "Poor"
    return quality, {"std": round(std, 4), "grad_mean": round(grad_mean, 5), "lap_var": round(lap_var, 7)}

# ---------------------------
# Flask App Configuration
# ---------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session handling
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PERMANENT_SESSION_LIFETIME"] = 300  # seconds
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fingerprint_classifier.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
    print("✅ Model input shape:", model.input_shape)
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise

# Define your class labels (update as per your model)
CLASS_NAMES = ["Arch", "Left Loop", "Right Loop", "Whorl", "Tented Arch"]

# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html", active_page="home")

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # Create uploads directory if it doesn't exist
                os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
                
                # Save file with timestamp to avoid conflicts
                filename = secure_filename(file.filename)
                filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}{os.path.splitext(filename)[1]}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Load and preprocess the image
                image = Image.open(filepath).convert("L")
                image = image.resize((128, 128))
                img_2d = np.array(image) / 255.0

                # Compute a simple quality metric (contrast + gradient)
                quality_label, q_metrics = compute_image_quality(img_2d)

                # Prepare array for model prediction (1,128,128,1)
                img_array = np.expand_dims(img_2d, axis=(0, -1))

                # Predict the class
                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)
                result_text = CLASS_NAMES[class_index]
                max_prob = float(np.max(prediction))

                # Use model-probability based quality rule: Good if prob > 0.5 else Bad
                prob_based_quality = "Good" if max_prob > 0.5 else "Bad"
                # store a simple quality_metrics with probability
                prob_metrics = {"prob": round(max_prob, 4)}

                print("✅ Predicted result:", result_text)
                print("ℹ️ Probability:", max_prob, "=> quality:", prob_based_quality)

                # Store in session before redirecting
                session.permanent = True
                session["prediction"] = result_text
                session["uploaded_image"] = filename
                session["quality"] = prob_based_quality
                session["quality_metrics"] = prob_metrics

                return redirect(url_for("result_page"))
            except Exception as e:
                print(f"Error during image processing: {str(e)}")
                return render_template("upload.html", error="Error processing image. Please try again.")
        else:
            return render_template("upload.html", error="Invalid file type. Please upload an image file.")

    return render_template("upload.html", active_page="upload")


@app.route('/predict', methods=['POST'])
def predict():
    # This endpoint is used by the hero upload form on index.html
    if 'file' not in request.files:
        return render_template('index.html', active_page='home', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', active_page='home', error='No file selected')

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        return render_template('index.html', active_page='home', error='Invalid file type')

    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}{os.path.splitext(filename)[1]}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Load and preprocess image
        image = Image.open(filepath).convert('L')
        image = image.resize((128, 128))
        img_2d = np.array(image) / 255.0

        # Compute quality
        quality_label, q_metrics = compute_image_quality(img_2d)

        img_array = np.expand_dims(img_2d, axis=(0, -1))

        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))
        result_text = CLASS_NAMES[class_index]
        max_prob = float(np.max(prediction))

        # Model-probability based quality decision
        prob_based_quality = "Good" if max_prob > 0.5 else "Bad"
        prob_metrics = {"prob": round(max_prob, 4)}

        session.permanent = True
        session['prediction'] = result_text
        session['uploaded_image'] = filename
        session['quality'] = prob_based_quality
        session['quality_metrics'] = prob_metrics

        return redirect(url_for('result_page'))
    except Exception as e:
        print(f"Error in /predict: {e}")
        return render_template('index.html', active_page='home', error='Error processing image')

@app.route("/result")
def result_page():
    prediction = session.get("prediction")
    uploaded_image = session.get("uploaded_image")
    quality = session.get("quality")
    quality_metrics = session.get("quality_metrics")

    if prediction and uploaded_image:
        image_path = f"uploads/{uploaded_image}"  # Relative path for template
        return render_template("result.html", result=prediction, image_path=image_path, quality=quality, quality_metrics=quality_metrics)
    return render_template("result.html", result=None, image_path=None)

@app.route("/about")
def about():
    return render_template("about.html", active_page="about")

@app.route("/overview")
def overview():
    return render_template("overview.html", active_page="overview")

@app.route("/contact")
def contact():
    return render_template("contact.html", active_page="contact")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Simulated login logic (replace with real user check as needed)
        data = request.get_json() or request.form
        email = data.get("email")
        password = data.get("password")
        if email == "user@example.com" and password == "password123":
            return {"success": True, "message": "Login successful!"}
        else:
            return {"success": False, "message": "Invalid credentials."}, 401
    return render_template("login.html", active_page="login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        # Here you would normally save the user to a database
        return {"success": True, "message": "Registration successful!"}
    return render_template("register.html", active_page="register")

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    print(f"✅ Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"✅ Model path: {MODEL_PATH}")
    print(f"✅ Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"  - {rule}")
    app.run(debug=True)