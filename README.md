# AI Roof Damage Detection & Estimation System

Production-ready system for automated roof damage detection from satellite imagery using computer vision (YOLOv8-seg).

## Quick Start

### 1. Setup Environment

```bash
# Clone and enter directory
cd AI_Roof_Damage_Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file:

```bash
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

### 3. Run Analysis

**CLI Mode:**
```bash
python main.py analyze 75201 --output ./output
```

**API Mode:**
```bash
python main.py serve --port 8000
# API docs at: http://localhost:8000/docs
```

### 4. API Usage

```bash
# Analyze a zipcode
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"zipcode": "75201"}'
```

## Project Overview

End-to-end pipeline that processes satellite imagery to detect roofs and damage:

## Core Features

### 1. Image Ingestion Pipeline
- **Multi-source support**: Google, Bing, Mapbox, ESRI, Nearmap, SkyFi, drone uploads
- **Geospatial processing**: GDAL/Rasterio for orthorectification, georeferencing, reprojection
- **Quality checks**: Cloud cover detection, angle validation, resolution checks
- **Output**: Standardized, geospatial-ready mosaics

### 2. Roof Segmentation & Structure Extraction
- **SAM-based segmentation**: Initial roof mask generation
- **Post-processing**: OpenCV contours, morphological operations
- **Vector conversion**: Shapely polygons with simplification and smoothing
- **Pitch estimation**: Shadow geometry + solar angle metadata
- **Output**: JSON + GeoJSON of roof boundaries, planes, edges, pitch estimates

### 3. Damage Detection (Ensemble Approach)
- **YOLOv8/YOLOv9**: Custom-trained model for damage classes
  - Hail impact, missing shingles, ponding, membrane damage
  - Flashing anomalies, cracks, blisters, warping
- **OpenAI Vision**: Secondary verifier to reduce false positives
- **SAM masks**: Pixel-accurate damage isolation
- **Multi-temporal analysis**: Before/after event comparison
- **Confidence scoring**: Uncertainty quantification, auto-flagging for review
- **Output**: Severity-scored heatmaps with confidence scores

### 4. Measurement Engine
- **Area calculation**: Roof plane polygons + pitch values
- **Surface area**: Pitch-adjusted calculations
- **Perimeter & materials**: Multi-plane geometry processing
- **Output**: Per-plane measurements, total areas, waste factors, material breakdown

### 5. Cost Estimation Engine
- **Rule-based pricing**: Structured JSON pricing tables
- **Repair vs replacement**: Logic-based decision making
- **Per-plane differentiation**: Material-specific costs
- **Waste adjustment**: Labor scaling factors
- **PDF generation**: WeasyPrint with HTML templates
- **Output**: Structured JSON + downloadable PDF proposals

### 6. Property Owner Identification
- **County data extraction**: Automated appraisal district scraping/API
- **Data enrichment**: Dropcontact, Clay, Apollo integration
- **Decision-maker identification**: LinkedIn/Google Places lookup
- **Multi-source verification**: Cross-checking for accuracy
- **Fuzzy matching**: Name variations, LLCs, trusts, property management
- **Output**: Owner information, mailing addresses, property type, decision-maker roles

### 7. GoHighLevel Integration
- **API sync**: Contacts, damage reports, PDFs, images, metadata
- **Pipeline automation**: Stage updates, task creation, tagging
- **Webhook triggers**: Automated follow-up based on severity scores
- **Smart sequencing**: High-severity → immediate outreach, low-severity → nurture sequence
- **Multi-channel**: Email → phone → SMS → LinkedIn escalation

## Technical Stack

- **Languages**: Python 3.10+
- **ML Framework**: PyTorch, YOLOv8/YOLOv9, SAM (Segment Anything Model)
- **APIs**: OpenAI Vision API, GoHighLevel API
- **Geospatial**: GDAL, Rasterio, Shapely, GeoJSON
- **Image Processing**: OpenCV, PIL/Pillow
- **Web Framework**: FastAPI
- **Storage**: S3-compatible, PostgreSQL
- **PDF Generation**: WeasyPrint
- **Data Enrichment**: Dropcontact, Clay, Apollo, LinkedIn Sales Navigator

## Accuracy Targets

- **Damage Detection**: 85-92% precision, <8% false positives
- **Property Owner ID**: 75-85% accuracy
- **Decision-Maker Enrichment**: 60-70% accuracy (varies by county data quality)

## Implementation Plan

### Phase 1: Foundation (Weeks 1-2)
- [ ] Image ingestion pipeline (multi-source support)
- [ ] Geospatial processing (GDAL/Rasterio)
- [ ] Quality checks (cloud, angle, resolution)
- [ ] SAM integration for roof segmentation

### Phase 2: Damage Detection (Weeks 3-4)
- [ ] YOLOv8/YOLOv9 model training on damage dataset
- [ ] OpenAI Vision integration for verification
- [ ] Multi-temporal analysis pipeline
- [ ] Confidence scoring & uncertainty quantification
- [ ] Ensemble approach (YOLO + OpenAI agreement logic)

### Phase 3: Measurement & Estimation (Weeks 5-6)
- [ ] Roof measurement engine (Shapely geometry)
- [ ] Pitch-adjusted calculations
- [ ] Cost estimation logic (rule-based pricing)
- [ ] PDF proposal generation (WeasyPrint)

### Phase 4: Property Identification (Weeks 7-8)
- [ ] County data extraction (scraping/API)
- [ ] Data enrichment (Dropcontact, Clay, Apollo)
- [ ] Decision-maker identification (LinkedIn, Google Places)
- [ ] Multi-source verification & fuzzy matching

### Phase 5: Integration & Automation (Weeks 9-10)
- [ ] GoHighLevel API integration
- [ ] Webhook triggers for automated workflows
- [ ] Smart sequencing (severity-based outreach)
- [ ] Multi-channel enrichment pipeline

### Phase 6: Optimization & Learning (Weeks 11-12)
- [ ] Incremental learning pipeline
- [ ] Active learning (user corrections → model updates)
- [ ] A/B testing for enrichment sources
- [ ] Performance tracking & accuracy monitoring

## Key Improvements & Differentiators

1. **Multi-temporal Analysis**: Before/after event comparison reduces false positives
2. **Uncertainty Quantification**: Confidence scores flag low-confidence cases for review
3. **Incremental Learning**: System improves from user corrections
4. **Context-Aware Filtering**: Adjusts thresholds by roof material, spatial validation
5. **Active Learning**: Incorrect detections improve future predictions
6. **Smart Sequencing**: Automated workflow triggers based on severity
7. **Multi-source Verification**: Cross-checking improves owner identification accuracy

## Deliverables

- Production-ready Python codebase
- Trained YOLO model weights
- API endpoints (FastAPI)
- GoHighLevel integration module
- Documentation & deployment guides
- Test datasets & validation results

## Next Steps

1. Set up project structure
2. Initialize image ingestion pipeline
3. Prepare training dataset for YOLO model
4. Begin SAM integration for roof segmentation

