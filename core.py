import io
import os
import cv2
import numpy as np
from PIL import Image
from keras import backend
from base64 import b64encode
from keras.models import Model
import matplotlib.pyplot as plt
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


class Core:
    def __init__(self, model_filename: str = "/Trained-Model/drone-detection-v5.h5") -> None:
        self.current_path = os.getcwd()
        self.model_path = self.current_path + model_filename
        self.labels_to_names = {0: 'drone', 1: 'dummy'}
        self.model = None

    def get_model(self) -> Model:
        print(self.model_path)
        return models.load_model(self.model_path, backbone_name='resnet50')

    def set_model(self, model: Model) -> 'Core':
        self.model = model
        return self

    @staticmethod
    def load_image_by_path(filename: str) -> np.ndarray:
        return read_image_bgr(filename)

    @staticmethod
    def load_image_by_memory(file: bytes) -> np.ndarray:
        image = np.asarray(Image.open(io.BytesIO(file)).convert('RGB'))
        return image[:, :, ::-1].copy()

    @staticmethod
    def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        return image[:, :, ::-1].copy()

    @staticmethod
    def pre_process_image(image: np.ndarray) -> tuple:
        pre_processed_image = preprocess_image(image)
        resized_image, scale = resize_image(pre_processed_image)
        return resized_image, scale

    @staticmethod
    def predict(model: Model, image: np.ndarray, scale: float) -> tuple:
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        return boxes, scores, labels

    def predict_with_graph_loaded_model(self, image: np.ndarray, scale: float) -> tuple:
        with backend.get_session().as_default():
            with backend.get_session().graph.as_default():
                return self.predict(self.model, image, scale)

    def predict_with_graph(self, model: Model, image: np.ndarray, scale: float) -> tuple:
        with backend.get_session().graph.as_default() as g:
            return self.predict(model, image, scale)

    @staticmethod
    def clear_graph_session() -> None:
        backend.clear_session()
        return None

    @staticmethod
    def get_drawing_image(image: np.ndarray) -> np.ndarray:
        # copy to draw on
        drawing_image = image.copy()
        drawing_image = cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB)
        return drawing_image

    def draw_boxes_in_image(self, drawing_image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                            threshold: float = 0.3) -> list:
        detections = []
        for box, score in zip(boxes[0], scores[0]):
            # scores are sorted so we can break
            if len(detections) > 0:
                threshold = 0.5
            if score < threshold:
                break

            detections.append({"box": [int(coord) for coord in box], "score": int(score * 100)})
            color = label_color(0)
            b = box.astype(int)
            draw_box(drawing_image, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[0], score)
            draw_caption(drawing_image, b, caption)
        return detections

    @staticmethod
    def visualize(drawing_image: np.ndarray) -> None:
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(drawing_image)
        plt.show()

    @staticmethod
    def convert_numpy_array_to_base64(image: np.ndarray, extension: str = ".png") -> bytes:
        data = cv2.imencode(extension, image)[1].tostring()
        return b64encode(data)
