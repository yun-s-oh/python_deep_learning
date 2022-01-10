from io import BytesIO
import os
from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf


app = Flask(__name__)
api = Api(app)

GENDER_CAT = ["Female", "Male", "Infant"]
AGE_CAT = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

# Enable GPU dynamic memory allocation

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

object_detection_model = tf.saved_model.load(os.path.join("models", "faster_rcnn"))
detect_fn = object_detection_model.signatures["serving_default"]
classification_model = tf.keras.models.load_model(os.path.join("models", "facenet"))


def run_object_detection(image_np):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop("num_detections"))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections_index = sum(detections["detection_scores"] > 0.7)
    detections = {k: v[:detections_index] for k, v in detections.items()}
    detections["num_detections"] = detections_index
    # detection_classes should be int.
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    return detections


def run_classification(img, detections):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    faces = []
    face_locations = []
    if detections["num_detections"]:
        for img_box in detections["detection_boxes"]:
            (ymin, xmin, ymax, xmax) = (
                img_box[0] * height,
                img_box[1] * width,
                img_box[2] * height,
                img_box[3] * width,
            )
            face_locations.append((max(0, xmin - 3), max(0, ymin - 15)))
            draw.rectangle(
                (
                    (max(xmin - 3, 0), max(ymin - 3, 0)),
                    (min(xmax + 3, width), min(ymax + 3, height)),
                ),
                outline="green",
                width=2,
            )
            crop_img = (
                np.asarray(
                    img.crop(
                        (
                            max(int(xmin) - 3, 0),
                            max(int(ymin) - 3, 0),
                            min(int(xmax) + 3, width),
                            min(int(ymax) + 3, height),
                        )
                    ).resize((160, 160))
                )
                / 255.0
            )
            faces.append(crop_img)
        face_np = np.array(faces)
        prediction = classification_model.predict(face_np)
        age_pred = [AGE_CAT[x] for x in np.argmax(prediction[0], axis=-1)]
        gender_pred = [GENDER_CAT[x] for x in np.argmax(prediction[1], axis=-1)]
        for i in range(detections["num_detections"]):
            font = ImageFont.load_default()
            text = f"{gender_pred[i]} ({age_pred[i]})"
            text_size = font.getsize(text)
            button_size = (text_size[0] + 2, text_size[1])
            button_img = Image.new("RGBA", button_size, "green")
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((0, 0), text, font=font, align="left", fill="white")
            img.paste(
                button_img, (int(face_locations[i][0]), int(face_locations[i][1]))
            )
    return img


def img_to_byte(img):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


class FaceRecognition(Resource):
    def get(self):
        return {"about": "Hello world"}

    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument("file", type=FileStorage, location="files")
        args = parse.parse_args()
        image_file = args["file"]
        if not image_file:
            return "No file attached", 400
        if image_file.mimetype.lower() not in ("image/jpeg", "image/png"):
            return "File not supported", 400
        img = Image.open(BytesIO(image_file.read()))
        detections = run_object_detection(np.asarray(img))
        img_processed = run_classification(img, detections)
        return send_file(img_to_byte(img_processed), mimetype="image/jpeg")


api.add_resource(FaceRecognition, "/")

if __name__ == "__main__":
    app.run()
