"""
Flask Web Application for Image Text Recognition
Presentation Demo: Comparing keras-ocr vs docTR
"""

import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from ocr_service import get_keras_ocr_service, get_doctr_service

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

        # Get both OCR services
        keras_service = get_keras_ocr_service()
        doctr_service = get_doctr_service()

        # Extract text using both engines
        keras_text = keras_service.extract_text(filepath)
        doctr_text = doctr_service.extract_text(filepath)

        # Clean up the uploaded file
        os.remove(filepath)

        return jsonify({
            'keras_ocr': keras_text if keras_text.strip() else '(No text detected)',
            'doctr': doctr_text if doctr_text.strip() else '(No text detected)'
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
    print("All OCR services ready!")

    app.run(debug=True, host='0.0.0.0', port=5000)
