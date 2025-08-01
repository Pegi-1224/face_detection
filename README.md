# face_detection
Face recognition system that combines InsightFace embeddings with intelligent filtering and statistically optimized similarity thresholds. The system addresses common challenges in face detection by implementing multi-level quality filters, fallback strategies for consistent data representation.
# Project Overview
This project implements a robust face recognition pipeline that addresses common challenges in face detection systems:
- Quality-based face filtering to eliminate poor detections and non-face content
- Multi-level fallback strategies to ensure consistent data representation
- Optimized similarity thresholds using statistical validation (MCC optimization)\
  
## Face Embedding with Advanced Filtering
1. Detection Confidence Filtering: Only faces detected with ≥90% confidence (strict) or ≥50% confidence (fallback)
2. Size-based Filtering: Removes faces smaller than 80×80 pixels (strict) or 30×30 pixels (fallback)
3. Quality Prioritization: Selects top 3 largest faces per image, sorted by bounding box area
4. Fallback Strategy: Ensures every image contributes at least one embedding for consistent dataset representation

## Similarity Measurement
Uses cosine similarity between normalized embeddings
- Range: -1 to +1 (higher values indicate greater similarity)
- Threshold Optimization: Statistically validated using Matthews Correlation Coefficient (MCC)
- Optimal Threshold: around 0.3

# Quick Start
## Prerequisites
```bash
# Create conda environment
conda create -n insightface_env python=3.9
conda activate insightface_env

# Install dependencies
pip install opencv-python
pip install insightface
pip install scikit-learn
pip install tqdm
pip install numpy
```

## Usage

```python
from face_detection import FaceMatchingPipeline

# Initialize pipeline with optimized settings
pipeline = FaceMatchingPipeline(
    threshold=0.3122,           # Optimized similarity threshold
    min_box_size=80,            # Strict size filter
    det_thresh=0.9,             # Strict detection confidence
    fallback_det_thresh=0.5,    # Fallback detection confidence
    fallback_min_size=30,       # Fallback size filter
    top_n_faces=3,              # Max faces per image
    ctx_id=-1                   # CPU mode (-1) or GPU mode (0)
)

# Run face matching
matches = pipeline.run_pipeline(
    input_image_path="path/to/reference_image.jpg",
    folder_path="path/to/search_folder/",
    save_results=True
)

# Display results
for img_name, similarity in matches:
    print(f"{img_name}: {similarity:.4f}")
```
