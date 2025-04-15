import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch # Use PyTorch
import torchvision.transforms.functional as F_trans # Use torchvision functional for transforms
from PIL import Image
import io
import psycopg2
from datetime import datetime
import os

# Import the ORIGINAL PyTorch model definition
from model import Net as PyTorchNet # Make sure model.py has the PyTorch Net class

# --- Database Configuration ---
# Read from environment variables, with fallbacks for local testing
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5433") # Keep 5433 as default for potential local runs
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "mysecretpassword")

# --- PyTorch Model Loading Function ---
@st.cache_resource # Cache the loaded model
def load_model():
    model = PyTorchNet()
    # Load weights from the original .pth file
    PATH = './mnist_cnn.pth'
    try:
         # Check if running on GPU is possible/needed (won't be in container unless specific setup)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load state dict, mapping location to CPU ensure compatibility in basic container
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.to(device) # Move model to appropriate device
        model.eval() # Set model to evaluation mode
        print(f"PyTorch Model loaded successfully onto {device}.")
    except FileNotFoundError:
         st.error(f"Error: Model weights file not found at {PATH}")
         st.stop()
    except Exception as e:
         st.error(f"Error loading PyTorch model: {e}")
         st.stop()
    return model, device

# --- Database Connection Function ---
@st.cache_resource # Re-enable caching
def get_db_connection():
    connection_string = f"dbname='{DB_NAME}' user='{DB_USER}' host='{DB_HOST}' password='{DB_PASSWORD}' port='{DB_PORT}'"
    # st.info(f"Attempting connection with string: dbname='{DB_NAME}' user='{DB_USER}' host='{DB_HOST}' port='{DB_PORT}' password='***'")
    try:
        conn = psycopg2.connect(connection_string)
        print("Database connection established.")
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Could not connect to database: {e}")
        st.error("Check docker compose logs for 'web' and 'db' services if running in Compose.")
        return None
    except Exception as e:
         st.error(f"An unexpected error occurred during DB connection: {e}")
         return None

# --- DB Helper Functions (init_db, log_prediction, get_history) ---
# (Keep these functions exactly as they were in the last working version)
# Function to initialize database table
def init_db(conn):
    if conn is None: return
    try:
        with conn.cursor() as cur: cur.execute("CREATE TABLE IF NOT EXISTS predictions (id SERIAL PRIMARY KEY, timestamp TIMESTAMP NOT NULL, predicted_digit INTEGER NOT NULL, true_label INTEGER, confidence FLOAT);"); conn.commit(); print("DB table initialized.")
    except psycopg2.Error as e: st.error(f"DB Init Error: {e}"); conn.rollback()

# Function to insert prediction data
def log_prediction(conn, timestamp, predicted_digit, true_label, confidence):
    if conn is None: st.warning("Cannot log: No DB conn."); return
    try:
        with conn.cursor() as cur: cur.execute("INSERT INTO predictions (timestamp, predicted_digit, true_label, confidence) VALUES (%s, %s, %s, %s)", (timestamp, predicted_digit, true_label, confidence)); conn.commit(); print(f"Logged: Pred={predicted_digit}, True={true_label}")
    except psycopg2.Error as e: st.error(f"DB Log Error: {e}"); conn.rollback()

# Function to fetch history
def get_history(conn):
    if conn is None: return []
    try:
        with conn.cursor() as cur: cur.execute("SELECT timestamp, predicted_digit, true_label, confidence FROM predictions ORDER BY timestamp DESC LIMIT 10;"); return cur.fetchall()
    except psycopg2.Error as e: st.error(f"DB History Error: {e}"); return []

# --- Image Preprocessing Function for PyTorch ---
def preprocess_image(image_data):
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert PIL image to PyTorch tensor
    # F_trans.to_tensor scales images from [0, 255] to [0, 1]
    img_tensor = F_trans.to_tensor(img) # Output shape (C=1, H=28, W=28)

    # Apply normalization: Normalize((0.5,), (0.5,)) -> (tensor - 0.5) / 0.5
    img_tensor = F_trans.normalize(img_tensor, mean=(0.5,), std=(0.5,))

    # Add batch dimension (B=1, C=1, H=28, W=28)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

# --- Main App Logic ---
st.title("PyTorch MNIST Digit Recognizer (Containerized)")

# Load model and connect to DB
pytorch_model, device = load_model() # Get model and device
conn = get_db_connection()

# Initialize DB table
if conn:
    init_db(conn)

st.write("Draw a digit (0-9) below and click Predict.")

# Canvas setup
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)", stroke_width=20, stroke_color="#FFFFFF",
    background_color="#000000", width=280, height=280,
    drawing_mode="freedraw", key="canvas",
)

# Prediction and Feedback Section
# (Layout and session state logic remains the same)
col1, col2 = st.columns(2)

with col1:
    predict_button_pressed = st.button("Predict", key="predict")

if predict_button_pressed:
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data
        st.write("Preprocessing drawing...")
        # Preprocess for PyTorch NCHW format
        img_tensor = preprocess_image(img_data).to(device) # Send tensor to device
        st.write("Image tensor shape after preprocessing:", img_tensor.shape)

        # Display preprocessed image
        display_img_tensor = img_tensor.squeeze().cpu() # Remove batch/channel, move to CPU
        display_img_tensor = (display_img_tensor * 0.5) + 0.5 # Denormalize
        display_img_np = np.clip(display_img_tensor.numpy() * 255, 0, 255).astype(np.uint8)
        st.image(display_img_np, caption="Preprocessed Image (28x28 Grayscale)", width=140)

        # Make prediction using PyTorch model
        st.write("Running prediction...")
        with torch.no_grad(): # Disable gradient calculation
            output = pytorch_model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

            prediction = prediction.item() # Get scalar value
            confidence = confidence.item() # Get scalar value

        # Display prediction
        st.metric("Prediction", f"{prediction}", f"{confidence:.1%}")

        # Store prediction in session state
        st.session_state.prediction = prediction
        st.session_state.confidence = confidence

    else:
        st.warning("Please draw a digit first!")

# Feedback form (remains the same)
if 'prediction' in st.session_state:
    with col2:
        with st.form(key="feedback_form"):
            true_label = st.number_input("Enter True Label:", min_value=0, max_value=9, step=1, key="true_label_input")
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                if conn:
                     log_prediction(conn, datetime.now(), st.session_state.prediction, true_label, st.session_state.confidence)
                     st.success("Feedback logged to database!")
                     del st.session_state.prediction
                     if 'confidence' in st.session_state: del st.session_state.confidence
                     st.rerun()
                else:
                    st.error("Cannot log feedback: No database connection.")

# History Display (remains the same)
st.subheader("Prediction History")
if conn:
    history = get_history(conn)
    if history:
        cols = st.columns([2, 1, 1, 1]); cols[0].write("**Timestamp**"); cols[1].write("**Pred**"); cols[2].write("**True**"); cols[3].write("**Conf**")
        for row in history: timestamp, pred_digit, true_lbl, conf = row; cols = st.columns([2, 1, 1, 1]); cols[0].text(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"); cols[1].text(f"{pred_digit}"); cols[2].text(f"{true_lbl if true_lbl is not None else '-'}") ; cols[3].text(f"{conf:.1%}" if conf is not None else '-')
    else:
        st.write("No prediction history yet.")
else:
    st.warning("Cannot display history: No database connection.")
