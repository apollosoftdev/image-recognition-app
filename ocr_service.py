"""
OCR Service Module
Handles image text recognition using both keras-ocr and docTR (TensorFlow-based)
"""

import keras_ocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


class KerasOCRService:
    """Service class for performing OCR using keras-ocr."""

    def __init__(self):
        """Initialize the keras-ocr pipeline with pre-trained models."""
        self.pipeline = keras_ocr.pipeline.Pipeline()

    def extract_text(self, image_path: str) -> str:
        """Extract text from an image using keras-ocr."""
        image = keras_ocr.tools.read(image_path)
        prediction_groups = self.pipeline.recognize([image])

        if not prediction_groups or not prediction_groups[0]:
            return ""

        predictions = prediction_groups[0]
        sorted_predictions = sorted(
            predictions,
            key=lambda p: (self._get_y_center(p[1]), self._get_x_center(p[1]))
        )

        lines = self._group_into_lines(sorted_predictions)
        text_lines = []
        for line in lines:
            line_sorted = sorted(line, key=lambda p: self._get_x_center(p[1]))
            line_text = " ".join(word for word, _ in line_sorted)
            text_lines.append(line_text)

        return "\n".join(text_lines)

    def _get_x_center(self, box) -> float:
        return sum(point[0] for point in box) / 4

    def _get_y_center(self, box) -> float:
        return sum(point[1] for point in box) / 4

    def _get_box_height(self, box) -> float:
        y_coords = [point[1] for point in box]
        return max(y_coords) - min(y_coords)

    def _group_into_lines(self, predictions, threshold_factor=0.5):
        if not predictions:
            return []

        lines = []
        current_line = [predictions[0]]
        current_y = self._get_y_center(predictions[0][1])

        for prediction in predictions[1:]:
            y_center = self._get_y_center(prediction[1])
            box_height = self._get_box_height(prediction[1])
            threshold = box_height * threshold_factor

            if abs(y_center - current_y) <= threshold:
                current_line.append(prediction)
            else:
                lines.append(current_line)
                current_line = [prediction]
                current_y = y_center

        if current_line:
            lines.append(current_line)

        return lines


class DocTRService:
    """Service class for performing OCR using docTR."""

    def __init__(self):
        """Initialize the docTR OCR predictor with pre-trained models."""
        self.predictor = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True
        )

    def extract_text(self, image_path: str) -> str:
        """Extract text from an image using docTR."""
        doc = DocumentFile.from_images(image_path)
        result = self.predictor(doc)

        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join(word.value for word in line.words)
                    if line_text.strip():
                        lines.append(line_text)

        return "\n".join(lines)


# Singleton instances
_keras_ocr_service = None
_doctr_service = None


def get_keras_ocr_service() -> KerasOCRService:
    """Get or create the keras-ocr service singleton."""
    global _keras_ocr_service
    if _keras_ocr_service is None:
        _keras_ocr_service = KerasOCRService()
    return _keras_ocr_service


def get_doctr_service() -> DocTRService:
    """Get or create the docTR service singleton."""
    global _doctr_service
    if _doctr_service is None:
        _doctr_service = DocTRService()
    return _doctr_service
