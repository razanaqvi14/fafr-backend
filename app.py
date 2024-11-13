from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from datetime import datetime
import os
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
import io
from pymongo import MongoClient


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUD_API_KEY"),
    api_secret=os.getenv("CLOUD_API_SECRET"),
    secure=True,
)

client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGO_DB")]


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "server is live"}), 200


# FEEDBACK FORM


@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    name = data.get("name")
    feedback = data.get("feedback")

    if feedback:
        time_added = datetime.now()
        feedback_entry = {"name": name, "feedback": feedback, "time_added": time_added}
        db.feedbacks.insert_one(feedback_entry)
        return (
            jsonify(
                {
                    "message": "Thank you for your valuable feedback! I appreciate your input and will use it to improve my service."
                }
            ),
            201,
        )
    else:
        return jsonify({"error": "There was an error submitting your feedback"}), 400


@app.route("/feedbacks", methods=["GET"])
def get_feedbacks():
    feedbacks = db.feedbacks.find()
    return jsonify(
        [
            {
                "name": feedback["name"],
                "feedback": feedback["feedback"],
                "time_added": feedback["time_added"],
            }
            for feedback in feedbacks
        ]
    )


# SAVE PREDICTIONS FORM


@app.route("/api/save_predictions_form", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    predictions = request.form.get("selectedOption")
    football_athlete_name = request.form.get("footballAthleteName")

    try:
        response = cloudinary.uploader.upload(file)
        image_url = response["secure_url"]
    except Exception as e:
        return jsonify({"error": "Cloudinary upload failed", "details": str(e)}), 500

    time_added = datetime.now()
    prediction_entry = {
        "uploaded_image_url": image_url,
        "predicted_football_athlete_name": football_athlete_name,
        "get_expected_prediction": predictions,
        "time_added": time_added,
    }
    db.predictionsinfo.insert_one(prediction_entry)

    return jsonify({"url": image_url}), 200


@app.route("/databasepredictions", methods=["GET"])
def get_database_predictions():
    predictions_info = db.predictionsinfo.find()
    return jsonify(
        [
            {
                "uploaded_image_url": prediction["uploaded_image_url"],
                "predicted_football_athlete_name": prediction[
                    "predicted_football_athlete_name"
                ],
                "get_expected_prediction": prediction["get_expected_prediction"],
                "time_added": prediction["time_added"],
            }
            for prediction in predictions_info
        ]
    )


# SAVE NO PREDICTIONS FORM


@app.route("/api/save_no_predictions_form", methods=["POST"])
def no_predictions_upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    predictions = request.form.get("selectedOption")

    try:
        response = cloudinary.uploader.upload(file)
        image_url = response["secure_url"]
    except Exception as e:
        return jsonify({"error": "Cloudinary upload failed", "details": str(e)}), 500

    time_added = datetime.now()
    no_prediction_entry = {
        "uploaded_image_url": image_url,
        "get_prediction": predictions,
        "time_added": time_added,
    }
    db.nopredictionsinfo.insert_one(no_prediction_entry)

    return jsonify({"url": image_url}), 200


@app.route("/databasenopredictions", methods=["GET"])
def get_database_no_predictions():
    no_predictions_info = db.nopredictionsinfo.find()
    return jsonify(
        [
            {
                "uploaded_image_url": no_prediction["uploaded_image_url"],
                "get_prediction": no_prediction["get_prediction"],
                "time_added": no_prediction["time_added"],
            }
            for no_prediction in no_predictions_info
        ]
    )


# PREDICTION

classes = [
    "Cristiano Ronaldo",
    "Erling Haaland",
    "Kylian Mbappe",
    "Lionel Messi",
    "Neymar Jr",
]


def load_model(weights_path):
    model = models.resnet18(weights="IMAGENET1K_V1")
    size_of_last_layer = model.fc.in_features
    model.fc = nn.Linear(size_of_last_layer, len(classes))
    model.load_state_dict(
        torch.load(
            weights_path,
        )
    )
    model.eval()
    return model


model = load_model(
    os.path.join(
        os.path.dirname(__file__),
        "Model",
        "TL_CNN_model_weights.pth",
    )
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

softmax = nn.Softmax(dim=1)


def crop_photo(image):
    faces = face_cascade.detectMultiScale(image)
    for x, y, w, h in faces:
        roi_color = image[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        if len(eyes) >= 2:
            return roi_color
    return None


def predict(cropped_image):
    transformed_image = image_transform(cropped_image).unsqueeze(0)
    with torch.no_grad():
        output = softmax(model(transformed_image))
        probabilities = {
            classes[i]: round(output[0, i].item(), 3) for i in range(len(classes))
        }
        _, prediction = torch.max(output, 1)
    return classes[prediction], probabilities


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cropped_image = crop_photo(image)

    if cropped_image is None:
        return jsonify(
            {
                "message": "Failed to process the image. This could be due to one of the following reasons: the image may not contain a clearly visible face with both eyes, or the image resolution might be too low for accurate face and eye detection. Please try again with a different image."
            }
        )

    prediction, probabilities = predict(cropped_image)
    return jsonify({"prediction": prediction, "probabilities": probabilities})


if __name__ == "__main__":
    app.run(debug=True)
