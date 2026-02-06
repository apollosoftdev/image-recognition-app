# Image Text Recognition Web App

A web application that extracts text from images using **4 different OCR engines** to compare their performance. Built with Python, TensorFlow, PaddlePaddle, and Tesseract for AI/ML demonstration purposes.

## Features

- **4 OCR Engines Comparison**: keras-ocr, docTR, Tesseract, PaddleOCR
- **Multi-language Support**: English, Chinese (Simplified & Traditional), Japanese
- **Drag-and-drop Upload**: Easy image uploading interface
- **Detailed Analysis**: Processing time, word count, confidence scores
- **Side-by-side Results**: Compare outputs from all 4 engines
- **Copy to Clipboard**: Easy text extraction
- **Educational About Page**: Learn about AI, ML, Deep Learning, and Neural Networks

## OCR Engines

| Engine | Framework | Best For | Languages |
|--------|-----------|----------|-----------|
| **keras-ocr** | TensorFlow | Scene text (photos, signs) | English |
| **docTR** | TensorFlow | Documents, structured text | English |
| **Tesseract** | Native C++ | General OCR, fast processing | 100+ languages |
| **PaddleOCR** | PaddlePaddle | Chinese text, high accuracy | 80+ languages |

## Technology Stack

### Backend
- **Python 3.10** - Programming language
- **Flask** - Web framework
- **Pillow** - Image processing

### Machine Learning Frameworks
- **TensorFlow 2.x** - Powers keras-ocr and docTR
- **PaddlePaddle 2.6** - Powers PaddleOCR
- **Tesseract** - Google's OCR engine

### OCR Libraries
- **keras-ocr 0.9.3** - CRAFT detector + CRNN recognizer
- **python-doctr 1.0** - DBNet detector + CRNN recognizer
- **pytesseract 0.3** - Python wrapper for Tesseract
- **paddleocr 2.8** - Baidu's PP-OCR system

### Infrastructure
- **Docker** - Containerization
- **Docker Compose** - Container orchestration

## Screenshots

![Demo Interface](ui-screenshots/1.png)
![OCR Results](ui-screenshots/2.png)
![About Page](ui-screenshots/3.png)

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 6GB RAM available for Docker

### Run the Application

```bash
# Build and start the container
docker compose up --build

# Or run in detached mode
docker compose up -d --build
```

**Note**: The first build downloads OCR models (~1GB total) and may take several minutes.

### Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

### Stop the Application

```bash
docker compose down
```

## Local Development Setup

### 1. Create a Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-jpn

# macOS
brew install tesseract tesseract-lang
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

## Usage

1. Open the web interface at `http://localhost:5000`
2. Upload an image by:
   - Dragging and dropping onto the upload area
   - Clicking "Browse Files" and selecting an image
3. Wait for all 4 OCR engines to process the image
4. Compare results in the 2x2 grid layout
5. Click "Copy" to copy text from any engine
6. Click "Clear & Try Another" to process another image

## Project Structure

```
image-recognition-app/
├── app.py                 # Main Flask application
├── ocr_service.py         # OCR processing logic (4 engines)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Files to exclude from Docker build
├── templates/
│   ├── index.html         # Main demo interface
│   └── about.html         # Educational about page
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── uploads/           # Temporary image storage
├── ui-screenshots/        # Screenshots for README
└── README.md              # This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main demo interface |
| `/about` | GET | Educational about page |
| `/upload` | POST | Upload image and get OCR results from all 4 engines |
| `/health` | GET | Health check endpoint |

### Upload Response Format

```json
{
  "keras_ocr": {
    "text": "extracted text",
    "word_count": 10,
    "line_count": 2,
    "char_count": 50,
    "processing_time_ms": 1234,
    "has_uppercase": false,
    "has_numbers": true,
    "has_symbols": false
  },
  "doctr": { ... },
  "tesseract": { ... },
  "paddleocr": { ... }
}
```

## Language Support

| Language | Tesseract | PaddleOCR | keras-ocr | docTR |
|----------|-----------|-----------|-----------|-------|
| English | Yes | Yes | Yes | Yes |
| Chinese (Simplified) | Yes | Yes | No | No |
| Chinese (Traditional) | Yes | Yes | No | No |
| Japanese | Yes | Yes* | No | No |

*PaddleOCR Chinese model supports Japanese Kanji

## Performance Tips

- **Scene Text**: Use keras-ocr for photos with text (signs, labels, banners)
- **Documents**: Use docTR for scanned documents, PDFs, structured text
- **Speed**: Use Tesseract for fastest processing
- **Chinese/Japanese**: Use PaddleOCR for best Asian language support
- **General**: Compare all 4 results and pick the best one

## Troubleshooting

### Container build fails
- Ensure you have sufficient disk space (at least 8GB)
- Check your internet connection for model downloads
- Try `docker compose build --no-cache` for a clean build

### Out of memory error
- Increase Docker memory limit to at least 6GB
- Close other applications to free up RAM

### Slow first request
- OCR models are loaded on startup
- First image processing may take longer
- Subsequent requests will be faster

### No text detected
- Ensure the image has clear, readable text
- Try with higher resolution images
- Check that the text is not too stylized or artistic
- Try a different OCR engine for better results

### PaddleOCR crashes
- Ensure Docker has at least 6GB memory allocated
- Check that `FLAGS_allocator_strategy=naive_best_fit` is set

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Browser)                        │
│  HTML5 + CSS3 + JavaScript + Fetch API                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Python)                          │
│  Flask + Werkzeug + Pillow                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Machine Learning Layer                       │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ keras-ocr   │ docTR       │ Tesseract   │ PaddleOCR        │
│ TensorFlow  │ TensorFlow  │ Native C++  │ PaddlePaddle     │
│ CRAFT+CRNN  │ DBNet+CRNN  │ LSTM        │ PP-OCRv4         │
└─────────────┴─────────────┴─────────────┴──────────────────┘
```

## Author

[@ApexSpire](https://github.com/apollosoftdev)

## License

MIT License
