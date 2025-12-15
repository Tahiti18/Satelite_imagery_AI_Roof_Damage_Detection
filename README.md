# AI Roof Damage Detection System

Automated roof damage detection from satellite imagery using YOLOv8-segmentation models. Input a US zipcode to fetch satellite images, detect roofs, and identify damage with visual annotations.

![Detection Example](https://github.com/aaliyanahmed1/Satelite_imagery_AI_Roof_Damage_Detection/raw/main/assets/images/detection_example.png)

## What It Does

This system analyzes satellite imagery to automatically detect roofs and identify damage. You provide a US zipcode, and it:

1. Fetches high-resolution satellite images from MapTiler API
2. Detects and segments roof boundaries using YOLOv8-segmentation
3. Identifies damage within detected roofs
4. Generates annotated visualizations and structured JSON/GeoJSON output

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Configure API key in .env
MAPTILER_API_KEY=your_key_here

# Download models
python scripts/download_mvp_models.py

# Run analysis
python test_zipcode.py
```

## API

```bash
python -m uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```
