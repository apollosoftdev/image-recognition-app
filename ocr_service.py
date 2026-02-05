"""
OCR Service Module
Handles image text recognition using both keras-ocr and docTR (TensorFlow-based)
"""

import time
import keras_ocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image


class KerasOCRService:
    """Service class for performing OCR using keras-ocr."""

    def __init__(self):
        """Initialize the keras-ocr pipeline with pre-trained models."""
        self.pipeline = keras_ocr.pipeline.Pipeline()

    def extract_text(self, image_path: str) -> dict:
        """Extract text from an image using keras-ocr with detailed analysis."""
        start_time = time.time()

        image = keras_ocr.tools.read(image_path)
        prediction_groups = self.pipeline.recognize([image])

        processing_time = round((time.time() - start_time) * 1000)  # ms

        if not prediction_groups or not prediction_groups[0]:
            return {
                "text": "",
                "word_count": 0,
                "line_count": 0,
                "char_count": 0,
                "regions_detected": 0,
                "processing_time_ms": processing_time,
                "has_uppercase": False,
                "has_numbers": False,
                "has_symbols": False
            }

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

        full_text = "\n".join(text_lines)

        return {
            "text": full_text,
            "word_count": len(predictions),
            "line_count": len(lines),
            "char_count": len(full_text.replace(" ", "").replace("\n", "")),
            "regions_detected": len(predictions),
            "processing_time_ms": processing_time,
            "has_uppercase": any(c.isupper() for c in full_text),
            "has_numbers": any(c.isdigit() for c in full_text),
            "has_symbols": any(not c.isalnum() and c not in ' \n' for c in full_text)
        }

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

    def extract_text(self, image_path: str) -> dict:
        """Extract text from an image using docTR with detailed analysis."""
        start_time = time.time()

        doc = DocumentFile.from_images(image_path)
        result = self.predictor(doc)

        processing_time = round((time.time() - start_time) * 1000)  # ms

        lines = []
        word_count = 0
        total_confidence = 0
        confidence_count = 0

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words_in_line = []
                    for word in line.words:
                        words_in_line.append(word.value)
                        word_count += 1
                        if hasattr(word, 'confidence'):
                            total_confidence += word.confidence
                            confidence_count += 1
                    line_text = " ".join(words_in_line)
                    if line_text.strip():
                        lines.append(line_text)

        full_text = "\n".join(lines)
        avg_confidence = round((total_confidence / confidence_count * 100), 1) if confidence_count > 0 else 0

        return {
            "text": full_text,
            "word_count": word_count,
            "line_count": len(lines),
            "char_count": len(full_text.replace(" ", "").replace("\n", "")),
            "regions_detected": word_count,
            "processing_time_ms": processing_time,
            "avg_confidence": avg_confidence,
            "has_uppercase": any(c.isupper() for c in full_text),
            "has_numbers": any(c.isdigit() for c in full_text),
            "has_symbols": any(not c.isalnum() and c not in ' \n' for c in full_text)
        }


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
