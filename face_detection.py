import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional

class FaceMatchingPipeline:
    def __init__(self, 
                 threshold: float = 0.3,
                 min_box_size: int = 80,
                 top_n_faces: int = 3,
                 det_thresh: float = 0.9,
                 fallback_det_thresh: float = 0.5,
                 fallback_min_size: int = 30,
                 model_name: str = 'buffalo_l',
                 ctx_id: int = 0,
                 det_size: Tuple[int, int] = (640, 640)):
        """
        Initialize Face Matching Pipeline with improved detection filtering
        
        Args:
            threshold: Cosine similarity threshold for matching faces
            min_box_size: Minimum face box size in pixels (strict)
            top_n_faces: Maximum number of faces to keep per image
            det_thresh: Detection confidence threshold (strict)
            fallback_det_thresh: Fallback detection confidence threshold
            fallback_min_size: Fallback minimum box size in pixels
            model_name: InsightFace model name
            ctx_id: GPU context ID (0 for GPU, -1 for CPU)
            det_size: Detection input size
        """
        self.threshold = threshold
        self.min_box_size = min_box_size
        self.top_n_faces = top_n_faces
        self.det_thresh = det_thresh
        self.fallback_det_thresh = fallback_det_thresh
        self.fallback_min_size = fallback_min_size
        
        # Initialize InsightFace model
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        
        # Statistics tracking
        self.stats = {
            'total_images_processed': 0,
            'images_with_faces': 0,
            'images_needing_fallback': 0,
            'faces_detected': 0,
            'faces_strict_filter': 0,
            'faces_fallback': 0
        }
        
    def extract_face_embeddings(self, img_path: str, update_stats: bool = True) -> Optional[np.ndarray]:
        """
        Extract face embeddings from a single image with improved filtering
        
        Args:
            img_path: Path to the image file
            update_stats: Whether to update global statistics
            
        Returns:
            Array of face embeddings or None if no faces found at all
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            return None
        
        if update_stats:
            self.stats['total_images_processed'] += 1
            
        faces = self.app.get(img)
        if not faces:
            return None
            
        if update_stats:
            self.stats['images_with_faces'] += 1
            self.stats['faces_detected'] += len(faces)
        
        # Strategy 1: Apply strict filters (det_thresh + size)
        strict_filtered = []
        for f in faces:
            if f.det_score >= self.det_thresh:
                x1, y1, x2, y2 = map(int, f.bbox[:4])
                w, h = x2-x1, y2-y1
                if w >= self.min_box_size and h >= self.min_box_size:
                    strict_filtered.append((f, w*h))
        
        selected_faces = []
        used_fallback = False
        
        if strict_filtered:
            # Use strict criteria - sort by area and take top_n
            strict_filtered.sort(key=lambda tup: tup[1], reverse=True)
            selected_faces = [tup[0] for tup in strict_filtered[:self.top_n_faces]]
            if update_stats:
                self.stats['faces_strict_filter'] += len(selected_faces)
                
        else:
            # Fallback strategy: ensure every image contributes at least one face
            used_fallback = True
            
            # Priority 1: Lower detection threshold but keep size requirement
            fallback_candidates = []
            for f in faces:
                if f.det_score >= self.fallback_det_thresh:
                    x1, y1, x2, y2 = map(int, f.bbox[:4])
                    w, h = x2-x1, y2-y1
                    if w >= self.min_box_size and h >= self.min_box_size:
                        fallback_candidates.append((f, w*h))
            
            if fallback_candidates:
                # Use lower det_thresh but same size requirement
                fallback_candidates.sort(key=lambda tup: tup[1], reverse=True)
                selected_faces = [fallback_candidates[0][0]]  # Take only the largest
            else:
                # Priority 2: Lower both detection threshold and size requirement
                fallback_candidates = []
                for f in faces:
                    if f.det_score >= self.fallback_det_thresh:
                        x1, y1, x2, y2 = map(int, f.bbox[:4])
                        w, h = x2-x1, y2-y1
                        if w >= self.fallback_min_size and h >= self.fallback_min_size:
                            fallback_candidates.append((f, w*h, f.det_score))
                
                if fallback_candidates:
                    # Sort by detection score (confidence), then by size
                    fallback_candidates.sort(key=lambda tup: (tup[2], tup[1]), reverse=True)
                    selected_faces = [fallback_candidates[0][0]]  # Take highest confidence
                else:
                    # Last resort: take the most confident face regardless of size
                    best_face = max(faces, key=lambda f: f.det_score)
                    selected_faces = [best_face]
            
            if update_stats:
                self.stats['images_needing_fallback'] += 1
                self.stats['faces_fallback'] += len(selected_faces)
        
        if not selected_faces:
            return None
            
        # Extract embeddings
        embeddings = [face.normed_embedding for face in selected_faces]
        return np.stack(embeddings)
    
    def process_folder(self, folder_path: str) -> Dict[str, np.ndarray]:
        """
        Process all images in a folder and extract embeddings
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Dictionary mapping image names to their embeddings
        """
        folder_embeddings = {}
        
        # Reset statistics for this folder
        self.stats = {
            'total_images_processed': 0,
            'images_with_faces': 0,
            'images_needing_fallback': 0,
            'faces_detected': 0,
            'faces_strict_filter': 0,
            'faces_fallback': 0
        }
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        print(f"Processing {len(image_files)} images from folder...")
        print(f"Using detection threshold: {self.det_thresh} (fallback: {self.fallback_det_thresh})")
        print(f"Using size filter: {self.min_box_size}px (fallback: {self.fallback_min_size}px)")
        
        for img_name in tqdm(image_files, desc="Extracting embeddings"):
            img_path = os.path.join(folder_path, img_name)
            embeddings = self.extract_face_embeddings(img_path, update_stats=True)
            
            if embeddings is not None:
                folder_embeddings[img_name] = embeddings
        
        # Print statistics
        self._print_processing_stats()
        return folder_embeddings
    
    def _print_processing_stats(self):
        """Print processing statistics"""
        print(f"\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Images processed: {self.stats['total_images_processed']}")
        print(f"Images with faces: {self.stats['images_with_faces']}")
        print(f"Images using fallback: {self.stats['images_needing_fallback']} ({self.stats['images_needing_fallback']/max(1,self.stats['images_with_faces'])*100:.1f}%)")
        print(f"")
        print(f"Total faces detected: {self.stats['faces_detected']}")
        print(f"Faces passing strict filter: {self.stats['faces_strict_filter']}")
        print(f"Faces added via fallback: {self.stats['faces_fallback']}")
        print(f"Final embeddings: {self.stats['faces_strict_filter'] + self.stats['faces_fallback']}")
        print("="*60)
    
    def find_matching_images(self, 
                           input_image_path: str, 
                           folder_path: str) -> List[Tuple[str, float]]:
        """
        Find images in folder that match the input image
        
        Args:
            input_image_path: Path to the reference image
            folder_path: Path to folder containing candidate images
            
        Returns:
            List of tuples (image_name, max_similarity_score) for matching images
        """
        # Extract embeddings from input image
        print("Processing input image...")
        input_embeddings = self.extract_face_embeddings(input_image_path, update_stats=False)
        
        if input_embeddings is None:
            print("Error: No valid faces found in input image")
            return []
            
        print(f"Found {len(input_embeddings)} face(s) in input image")
        
        # Process folder images
        folder_embeddings = self.process_folder(folder_path)
        
        if not folder_embeddings:
            print("No valid faces found in folder images")
            return []
            
        # Find matches
        matches = []
        print("Comparing faces...")
        
        for img_name, img_embeddings in tqdm(folder_embeddings.items(), desc="Finding matches"):
            max_similarity = 0.0
            
            # Compare each face in input image with each face in candidate image
            for input_emb in input_embeddings:
                for img_emb in img_embeddings:
                    similarity = cosine_similarity([input_emb], [img_emb])[0][0]
                    max_similarity = max(max_similarity, similarity)
            
            # Check if similarity exceeds threshold
            if max_similarity >= self.threshold:
                matches.append((img_name, max_similarity))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def run_pipeline(self, 
                    input_image_path: str, 
                    folder_path: str, 
                    save_results: bool = True,
                    results_file: str = "face_matching_results.txt") -> List[Tuple[str, float]]:
        """
        Run the complete face matching pipeline
        
        Args:
            input_image_path: Path to the reference image
            folder_path: Path to folder containing candidate images
            save_results: Whether to save results to file
            results_file: Path to save results
            
        Returns:
            List of matching images with similarity scores
        """
        print("="*80)
        print("ENHANCED FACE MATCHING PIPELINE")
        print("="*80)
        print(f"Input image: {input_image_path}")
        print(f"Search folder: {folder_path}")
        print(f"Similarity threshold: {self.threshold}")
        print(f"Detection confidence: {self.det_thresh} (fallback: {self.fallback_det_thresh})")  
        print(f"Face size filter: {self.min_box_size}px (fallback: {self.fallback_min_size}px)")
        print("="*80)
        
        # Find matches
        matches = self.find_matching_images(input_image_path, folder_path)
        
        # Display results
        print(f"\nFound {len(matches)} matching images:")
        print("-" * 60)
        
        if matches:
            for i, (img_name, similarity) in enumerate(matches, 1):
                print(f"{i:2d}. {img_name:<35} (similarity: {similarity:.4f})")
        else:
            print("No matching faces found")
            
        # Save results to file if requested
        if save_results and matches:
            with open(results_file, 'w') as f:
                f.write(f"Enhanced Face Matching Results\n")
                f.write(f"Input Image: {input_image_path}\n")
                f.write(f"Search Folder: {folder_path}\n")
                f.write(f"Similarity Threshold: {self.threshold}\n")
                f.write(f"Detection Confidence: {self.det_thresh} (fallback: {self.fallback_det_thresh})\n")
                f.write(f"Size Filter: {self.min_box_size}px (fallback: {self.fallback_min_size}px)\n")
                f.write(f"Matches Found: {len(matches)}\n\n")
                
                for i, (img_name, similarity) in enumerate(matches, 1):
                    f.write(f"{i}. {img_name} (similarity: {similarity:.4f})\n")
                    
            print(f"\nResults saved to: {results_file}")
        
        return matches


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with your optimal settings
    pipeline = FaceMatchingPipeline(
        threshold=0.3122,           # Your optimal similarity threshold
        min_box_size=80,            # Strict size requirement
        top_n_faces=3,              # Max faces per image
        det_thresh=0.9,             # Strict detection confidence
        fallback_det_thresh=0.5,    # Fallback detection confidence
        fallback_min_size=30,       # Fallback size requirement
        ctx_id=0                    # Use GPU if available, -1 for CPU
    )
    
    # Example paths 
    input_image = "../testing/Chris_Evans_1.jpg"
    search_folder = "../testing/testing_images"
    
    # Run the pipeline
    matches = pipeline.run_pipeline(
        input_image_path=input_image,
        folder_path=search_folder,
        save_results=True,
        results_file="enhanced_matching_results.txt"
    )
    
    # Print summary
    print(f"\nSummary: Found {len(matches)} matching images")
    if matches:
        best_match = matches[0]
        print(f"Best match: {best_match[0]} (similarity: {best_match[1]:.4f})")