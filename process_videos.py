import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
from pathlib import Path
from PIL import Image
from lavis.models import load_model_and_preprocess
import huggingface_hub
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

class VideoProcessor:
    def __init__(self, base_path="SEED-DV"):
        self.base_path = Path(base_path)
        self.video_path = self.base_path / "Video"
        self.blip_path = self.video_path / "BLIP-caption"
        self.meta_path = self.video_path / "meta-info"
        
        # Set cache directory for models
        self.cache_dir = Path.home() / ".cache" / "blip_models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load BLIP-2 model
        print("Loading BLIP-2 model...")
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_opt",
            model_type="pretrain_opt2.7b",
            is_eval=True,
            device=self.device
        )
        print("BLIP-2 model loaded successfully")
        
        # Load sentence transformer for caption similarity
        print("Loading sentence transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer loaded successfully")
    
    def _load_metadata(self):
        """Load all metadata from .npy files"""
        metadata = {}
        for file in self.meta_path.glob("*.npy"):
            key = file.stem
            metadata[key] = np.load(file)
        return metadata
    
    def _get_video_clips(self, video_path):
        """Extract 2-second clips from video"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate number of 2-second clips
        frames_per_clip = int(fps * 2)
        num_clips = total_frames // frames_per_clip
        
        clips = []
        for i in range(num_clips):
            start_frame = i * frames_per_clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clips.append({
                    "start_frame": start_frame,
                    "end_frame": start_frame + frames_per_clip,
                    "frame": frame_rgb
                })
        
        cap.release()
        return clips
    
    def _load_blip_captions(self, video_name):
        """Load BLIP captions for a specific video"""
        caption_file = self.blip_path / f"{video_name}.txt"
        if not caption_file.exists():
            return []
        
        with open(caption_file, 'r') as f:
            captions = [line.strip() for line in f.readlines()]
        return captions
    
    def _is_valid_caption(self, caption, base_caption):
        """Check if a caption is valid based on length, character set, and relevance."""
        if not caption or len(caption.strip()) < 10:
            return False
        
        # Check for Chinese characters (rough check)
        if any(ord(c) > 0x4e00 and ord(c) < 0x9fff for c in caption):
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            "chinese",
            "japanese",
            "oriental",
            "script",
            "language",
            "characters",
            "written",
            "writing",
            "guadalajara",
            "mexico",
            "san francisco",
            "google street view",
            "virtual city"
        ]
        
        if any(pattern in caption.lower() for pattern in invalid_patterns):
            return False
        
        # Check for relevance to base caption
        base_words = set(base_caption.lower().split())
        caption_words = set(caption.lower().split())
        
        # Get important words from base caption
        important_words = {"city", "building", "buildings", "street", "night", "tall", "mountain", "river", "water"}
        base_important = base_words.intersection(important_words)
        caption_important = caption_words.intersection(important_words)
        
        # If the base caption has important words, the new caption should have at least one
        if base_important and not caption_important:
            return False
        
        return True
    
    def _get_theme(self, caption):
        """Extract the main theme from a caption."""
        themes = {
            "mountain": ["mountain", "peak", "hill", "snow", "rock"],
            "water": ["river", "lake", "ocean", "sea", "water"],
            "city": ["city", "building", "street", "road", "urban"],
            "nature": ["tree", "forest", "grass", "field", "plant"],
            "animal": ["animal", "bird", "dog", "cat", "horse"],
            "person": ["person", "man", "woman", "people", "human"],
            "vehicle": ["car", "boat", "ship", "vehicle", "transport"],
            "sky": ["sky", "cloud", "sun", "moon", "star"]
        }
        
        caption_lower = caption.lower()
        for theme, keywords in themes.items():
            if any(keyword in caption_lower for keyword in keywords):
                return theme
        
        return "unknown"
    
    def _refine_caption(self, caption, similar_captions, threshold=0.7):
        """Refine a caption using similar captions from other videos."""
        if not similar_captions:
            return caption
        
        # Get embeddings for all captions
        caption_embedding = self.sentence_model.encode([caption])[0]
        similar_embeddings = self.sentence_model.encode(similar_captions)
        
        # Calculate similarities
        similarities = cosine_similarity([caption_embedding], similar_embeddings)[0]
        
        # Get most similar captions
        similar_indices = np.where(similarities > threshold)[0]
        if len(similar_indices) == 0:
            return caption
        
        # Combine similar captions
        similar_captions = [similar_captions[i] for i in similar_indices]
        combined_caption = " ".join(similar_captions)
        
        # Use the original caption if refinement doesn't improve it
        if len(combined_caption.split()) < len(caption.split()):
            return caption
        
        return combined_caption
    
    def _generate_additional_captions(self, image, base_caption):
        """Generate additional captions for an image."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Process the image
            image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
            
            additional_captions = []
            print("Generating additional captions...")
            
            # Generate 3 different captions
            for i in range(3):
                max_retries = 5  # Maximum number of retries per caption
                retry_count = 0
                caption = ""
                
                while retry_count < max_retries:
                    try:
                        # Clear CUDA cache before each generation attempt
                        torch.cuda.empty_cache()
                        
                        # Move model to CPU temporarily to free up GPU memory
                        self.model.cpu()
                        torch.cuda.empty_cache()
                        
                        # Add a small delay to allow memory to be freed
                        time.sleep(1)
                        
                        # Move model back to GPU for generation
                        self.model.to(self.device)
                        
                        # Generate caption using the model's generate method with lower memory settings
                        caption = self.model.generate(
                            {"image": image},
                            use_nucleus_sampling=True,
                            num_beams=2,  # Further reduced from 3 to save memory
                            max_length=25,  # Reduced from 30
                            min_length=10,
                            repetition_penalty=1.5
                        )[0]
                        
                        # Move model back to CPU to free memory
                        self.model.cpu()
                        torch.cuda.empty_cache()
                        
                        # Validate the caption
                        if self._is_valid_caption(caption, base_caption):
                            additional_captions.append(caption)
                            print(f"  Generated caption {i+1}: {caption}")
                            break
                        else:
                            print(f"  Retrying caption {i+1} (attempt {retry_count + 1}/{max_retries})")
                            retry_count += 1
                            # Add a small delay between retries
                            time.sleep(0.5)
                            continue
                    
                    except Exception as e:
                        print(f"Error generating caption: {str(e)}")
                        retry_count += 1
                        # Add a longer delay after errors
                        time.sleep(2)
                        continue
                    
                    finally:
                        # Ensure model is back on GPU for next iteration
                        self.model.to(self.device)
                
                if retry_count >= max_retries:
                    print(f"  Failed to generate valid caption {i+1} after {max_retries} attempts")
                    additional_captions.append("")
                
                # Add a delay between caption generations
                time.sleep(1)
            
            # Clear CUDA cache after processing all captions
            torch.cuda.empty_cache()
            
            return additional_captions
        
        except Exception as e:
            print(f"Error in caption generation: {str(e)}")
            return ["", "", ""]
    
    def process_videos(self):
        """Process videos and generate caption mappings"""
        video_mappings = {}
        all_captions = []  # Store all captions for similarity comparison
        
        # Get all video files
        video_files = list(self.video_path.glob("*.mp4"))
        print(f"\nFound {len(video_files)} video files to process")
        
        # Process only the first video and first 5 clips
        video_file = video_files[0]
        video_name = video_file.stem
        print(f"\nProcessing video: {video_name}")
        
        # Get clips and captions
        print("Extracting video clips...")
        clips = self._get_video_clips(video_file)
        clips = clips[:5]  # Take only first 5 clips
        print(f"Extracted {len(clips)} clips from video")
        
        print("Loading base captions...")
        base_captions = self._load_blip_captions(video_name)
        base_captions = base_captions[:5]  # Take only first 5 captions
        print(f"Loaded {len(base_captions)} base captions")
        
        # Create mapping for this video
        video_mappings[video_name] = []
        
        for i, (clip, base_caption) in enumerate(zip(clips, base_captions)):
            print(f"\nProcessing clip {i+1}/{len(clips)}")
            print(f"Frame range: {clip['start_frame']} to {clip['end_frame']}")
            print(f"Base caption: {base_caption}")
            
            # Clear CUDA cache before processing each clip
            torch.cuda.empty_cache()
            
            # Generate additional captions
            print("Generating additional captions...")
            additional_captions = self._generate_additional_captions(clip["frame"], base_caption)
            
            # Refine captions using similar scenes
            refined_captions = []
            for caption in additional_captions:
                if caption:  # Only refine non-empty captions
                    # Find similar captions from other videos
                    similar_captions = [c for c in all_captions if c != caption]
                    refined_caption = self._refine_caption(caption, similar_captions)
                    refined_captions.append(refined_caption)
                else:
                    refined_captions.append("")
            
            print("Generated captions:")
            for j, caption in enumerate(refined_captions):
                print(f"  Caption {j+1}: {caption}")
            
            # Store captions for similarity comparison
            all_captions.extend([c for c in refined_captions if c])
            
            clip_data = {
                "clip_id": i,
                "start_frame": clip["start_frame"],
                "end_frame": clip["end_frame"],
                "captions": {
                    "base": base_caption,
                    "additional": refined_captions
                },
                "metadata": {
                    "color": self.metadata.get("All_video_color", [])[i].tolist() if i < len(self.metadata.get("All_video_color", [])) else None,
                    "face_detected": self.metadata.get("All_video_face_apperance", [])[i].tolist() if i < len(self.metadata.get("All_video_face_apperance", [])) else None,
                    "human_present": self.metadata.get("All_video_human_apperance", [])[i].tolist() if i < len(self.metadata.get("All_video_human_apperance", [])) else None,
                    "label": self.metadata.get("All_video_label", [])[i].tolist() if i < len(self.metadata.get("All_video_label", [])) else None,
                    "object_count": self.metadata.get("All_video_obj_number", [])[i].tolist() if i < len(self.metadata.get("All_video_obj_number", [])) else None,
                    "motion_score": self.metadata.get("All_video_optical_flow_score", [])[i].tolist() if i < len(self.metadata.get("All_video_optical_flow_score", [])) else None
                }
            }
            video_mappings[video_name].append(clip_data)
            print(f"Completed processing clip {i+1}")
        
        # Save mappings to JSON
        output_file = self.base_path / "enhanced_video_caption_mappings.json"
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(video_mappings, f, indent=2)
        
        print("\nProcessing complete!")

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_videos() 