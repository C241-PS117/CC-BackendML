# OCR API Documentation

This document outlines the available endpoints for the OCR API and provides instructions on how to use it effectively.

## Endpoints

### Predict

| Endpoint   | Method | Input                          | Description         | Status |
|------------|--------|--------------------------------|---------------------|--------|
| `/predict` | POST   | `jawaban`, `image` (FILE)      | Calculate Answer Score | OK     |

## Usage Instructions

### Prerequisites

- **Python Version**: Ensure you are using Python version 3.10.
- **Required Libraries**: Make sure you have the version of TensorFlow = 2.15.0

### Installation

Install the necessary libraries by running the following command:

```bash
pip pip install Flask SQLAlchemy PyMySQL tensorflow keras python-dotenv Flask-SQLAlchemy opencv-python
```

### Running the API

To start the API server, run the following command in your terminal:

```bash
python app.py
```

### Model Update

To update the `prediction_model.h5` file, simply replace the old file with the new one.

---
