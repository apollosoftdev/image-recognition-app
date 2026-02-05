"""
Flask Web Application for Image Text Recognition
Presentation Demo: Comparing keras-ocr vs docTR
"""

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from ocr_service import (
    get_keras_ocr_service,
    get_doctr_service,
    get_tesseract_service,
    get_paddleocr_service
)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page with upload form."""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the about/technology page."""
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and perform OCR with both engines."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get all OCR services
        keras_service = get_keras_ocr_service()
        doctr_service = get_doctr_service()
        tesseract_service = get_tesseract_service()
        paddleocr_service = get_paddleocr_service()

        # Extract text using all engines (returns dict with analysis)
        keras_result = keras_service.extract_text(filepath)
        doctr_result = doctr_service.extract_text(filepath)
        tesseract_result = tesseract_service.extract_text(filepath)
        paddleocr_result = paddleocr_service.extract_text(filepath)

        # Clean up the uploaded file
        os.remove(filepath)

        return jsonify({
            'keras_ocr': {
                'text': keras_result['text'] if keras_result['text'].strip() else '(No text detected)',
                'word_count': keras_result['word_count'],
                'line_count': keras_result['line_count'],
                'char_count': keras_result['char_count'],
                'regions_detected': keras_result['regions_detected'],
                'processing_time_ms': keras_result['processing_time_ms'],
                'has_uppercase': keras_result['has_uppercase'],
                'has_numbers': keras_result['has_numbers'],
                'has_symbols': keras_result['has_symbols']
            },
            'doctr': {
                'text': doctr_result['text'] if doctr_result['text'].strip() else '(No text detected)',
                'word_count': doctr_result['word_count'],
                'line_count': doctr_result['line_count'],
                'char_count': doctr_result['char_count'],
                'regions_detected': doctr_result['regions_detected'],
                'processing_time_ms': doctr_result['processing_time_ms'],
                'avg_confidence': doctr_result['avg_confidence'],
                'has_uppercase': doctr_result['has_uppercase'],
                'has_numbers': doctr_result['has_numbers'],
                'has_symbols': doctr_result['has_symbols']
            },
            'tesseract': {
                'text': tesseract_result['text'] if tesseract_result['text'].strip() else '(No text detected)',
                'word_count': tesseract_result['word_count'],
                'line_count': tesseract_result['line_count'],
                'char_count': tesseract_result['char_count'],
                'regions_detected': tesseract_result['regions_detected'],
                'processing_time_ms': tesseract_result['processing_time_ms'],
                'avg_confidence': tesseract_result['avg_confidence'],
                'has_uppercase': tesseract_result['has_uppercase'],
                'has_numbers': tesseract_result['has_numbers'],
                'has_symbols': tesseract_result['has_symbols']
            },
            'paddleocr': {
                'text': paddleocr_result['text'] if paddleocr_result['text'].strip() else '(No text detected)',
                'word_count': paddleocr_result['word_count'],
                'line_count': paddleocr_result['line_count'],
                'char_count': paddleocr_result['char_count'],
                'regions_detected': paddleocr_result['regions_detected'],
                'processing_time_ms': paddleocr_result['processing_time_ms'],
                'avg_confidence': paddleocr_result['avg_confidence'],
                'has_uppercase': paddleocr_result['has_uppercase'],
                'has_numbers': paddleocr_result['has_numbers'],
                'has_symbols': paddleocr_result['has_symbols']
            }
        })

    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    print("Initializing OCR services (loading models)...")
    print("Loading keras-ocr models...")
    get_keras_ocr_service()
    print("Loading docTR models...")
    get_doctr_service()
    print("Loading Tesseract...")
    get_tesseract_service()
    print("Loading PaddleOCR models...")
    get_paddleocr_service()
    print("All OCR services ready!")

    app.run(debug=True, host='0.0.0.0', port=5000)
