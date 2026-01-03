#!/usr/bin/env python3
"""
VLM (Vision Language Model) Guidance for Gaussian Decoder Training.

Experimental module that uses local VLMs through LM Studio to provide
semantic understanding for image-to-3D reconstruction.

Features:
- Density guidance: identify regions needing more Gaussians
- Depth hints: relative depth ordering from VLM understanding
- Quality evaluation: compare rendered output to input

Requires LM Studio running with a VLM model (e.g., Qwen2-VL, Qwen3-VL).

Usage:
    from vlm_guidance import VLMGuidance

    vlm = VLMGuidance()
    density_map = vlm.get_density_guidance("image.png")
    depth_hints = vlm.get_depth_hints("image.png")
"""

import base64
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import background removal from preprocessing script
SCRIPT_DIR = Path(__file__).parent
try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from preprocess_training_data import remove_background, REMBG_AVAILABLE
except ImportError:
    REMBG_AVAILABLE = False
    remove_background = None


class VLMGuidance:
    """
    Vision Language Model guidance for Gaussian splatting.

    Connects to LM Studio's OpenAI-compatible API to get semantic
    understanding of images for improved 3D reconstruction.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:1234/v1/chat/completions",
        model: str = "qwen2-vl",
        timeout: int = 60,
    ):
        """
        Initialize VLM guidance.

        Args:
            api_url: LM Studio API endpoint
            model: Model name to use
            timeout: Request timeout in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library required. Run: pip install requests")

        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_pil_image(self, image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _query(self, image_b64: str, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """
        Query the VLM with an image and prompt.

        Args:
            image_b64: Base64 encoded image
            prompt: Text prompt
            max_tokens: Maximum response tokens

        Returns:
            VLM response text or None on error
        """
        try:
            # Detect image format from base64 header or default to PNG
            if image_b64.startswith("/9j/"):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_b64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,  # Low temperature for consistent outputs
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"VLM API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            print("VLM API connection failed. Is LM Studio running?")
            return None
        except Exception as e:
            print(f"VLM query error: {e}")
            return None

    def check_connection(self) -> bool:
        """Check if LM Studio is running and accessible."""
        try:
            # Try the models endpoint to check connection
            models_url = self.api_url.replace("/chat/completions", "/models")
            response = requests.get(models_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_density_guidance(
        self,
        image_path: str,
        grid_size: int = 4,
    ) -> Optional[np.ndarray]:
        """
        Get density guidance map from VLM.

        Asks the VLM to identify regions that need more detail/Gaussians.

        Args:
            image_path: Path to input image
            grid_size: Size of output grid (NxN)

        Returns:
            numpy array of shape (grid_size, grid_size) with importance weights [0-1]
            or None on error
        """
        image_b64 = self._encode_image(image_path)

        prompt = f"""Analyze this image and rate the visual importance of each region.
Divide the image into a {grid_size}x{grid_size} grid.

For each cell, rate its importance from 0.0 to 1.0 where:
- 1.0 = Critical detail (faces, eyes, text, fine textures)
- 0.8 = High importance (edges, distinctive features)
- 0.5 = Medium importance (general textures, patterns)
- 0.2 = Low importance (flat areas, backgrounds)
- 0.0 = Empty/uniform regions

Output ONLY a {grid_size}x{grid_size} grid of numbers, one row per line.
Example for a 4x4 grid:
0.3 0.5 0.8 0.4
0.4 1.0 1.0 0.5
0.3 0.8 0.9 0.4
0.2 0.3 0.4 0.2"""

        response = self._query(image_b64, prompt)
        if response is None:
            return None

        # Parse the grid from response
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```[\w]*\n?', '', response)
            response = response.strip()

            lines = response.split("\n")
            grid = []

            for line in lines:
                # Skip empty lines and lines that look like explanations
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('*'):
                    continue

                # Extract decimal numbers (more precise pattern)
                numbers = re.findall(r'\b0?\.\d+\b|\b1\.0\b|\b[01]\b', line)

                # Also try space/comma separated floats
                if len(numbers) < grid_size:
                    numbers = re.findall(r'[\d]+\.[\d]+|[01](?:\s|,|$)', line)

                # Last resort: any number-like strings
                if len(numbers) < grid_size:
                    numbers = [n for n in re.findall(r'[\d.]+', line)
                               if '.' in n or (len(n) == 1 and n in '01')]

                if len(numbers) >= grid_size:
                    try:
                        row = [min(1.0, max(0.0, float(n))) for n in numbers[:grid_size]]
                        grid.append(row)
                    except ValueError:
                        continue

                if len(grid) >= grid_size:
                    break

            if len(grid) >= grid_size:
                return np.array(grid[:grid_size], dtype=np.float32)
            else:
                # Fallback: generate uniform grid if parsing fails
                # This allows training to continue even with VLM parsing issues
                return None

        except Exception as e:
            print(f"Error parsing VLM density response: {e}")
            return None

    def get_depth_hints(
        self,
        image_path: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get relative depth ordering hints from VLM.

        Args:
            image_path: Path to input image

        Returns:
            List of objects with relative depth info, or None on error
        """
        image_b64 = self._encode_image(image_path)

        prompt = """Analyze the depth in this image. List the main objects/regions
from NEAREST to FARTHEST from the camera.

For each region, provide:
1. Name/description (brief)
2. Approximate screen position (top/bottom/left/right/center)
3. Relative depth (0.0=very close, 1.0=very far)

Output as JSON array:
[
  {"name": "object name", "position": "center", "depth": 0.2},
  {"name": "background", "position": "all", "depth": 0.9}
]

Only output the JSON array, nothing else."""

        response = self._query(image_b64, prompt, max_tokens=1024)
        if response is None:
            return None

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            print(f"Error parsing VLM depth response: {e}")
            return None

    def evaluate_quality(
        self,
        original_path: str,
        rendered_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare original image to rendered output.

        Args:
            original_path: Path to original input image
            rendered_path: Path to rendered/reconstructed image

        Returns:
            Quality evaluation dict or None on error
        """
        # Load both images and combine side by side
        original = Image.open(original_path).convert("RGB")
        rendered = Image.open(rendered_path).convert("RGB")

        # Resize to same size
        size = (256, 256)
        original = original.resize(size, Image.Resampling.LANCZOS)
        rendered = rendered.resize(size, Image.Resampling.LANCZOS)

        # Combine side by side
        combined = Image.new("RGB", (512, 256))
        combined.paste(original, (0, 0))
        combined.paste(rendered, (256, 0))

        image_b64 = self._encode_pil_image(combined)

        prompt = """Compare these two images (original on LEFT, reconstruction on RIGHT).

Rate the reconstruction quality on these aspects (0-10):
1. Overall similarity
2. Fine detail preservation
3. Color accuracy
4. Edge sharpness
5. Structural accuracy

Also identify:
- Best preserved regions
- Regions needing improvement

Output as JSON:
{
  "similarity": 7,
  "detail": 6,
  "color": 8,
  "sharpness": 5,
  "structure": 7,
  "overall": 6.6,
  "good_regions": ["face", "edges"],
  "needs_work": ["background texture", "fine details"]
}

Only output JSON, nothing else."""

        response = self._query(image_b64, prompt, max_tokens=512)
        if response is None:
            return None

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            print(f"Error parsing VLM quality response: {e}")
            return None

    def get_segmentation_hints(
        self,
        image_path: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get semantic segmentation hints from VLM.

        Useful for region-aware loss weighting during training.

        Args:
            image_path: Path to input image

        Returns:
            List of semantic regions with bounding boxes
        """
        image_b64 = self._encode_image(image_path)

        prompt = """Identify the main semantic regions in this image.

For each region provide:
1. Label (e.g., face, body, background, object)
2. Importance for 3D reconstruction (critical/high/medium/low)
3. Approximate bounding box as fractions [x_min, y_min, x_max, y_max] where 0,0 is top-left

Output as JSON array:
[
  {"label": "face", "importance": "critical", "bbox": [0.3, 0.1, 0.7, 0.5]},
  {"label": "background", "importance": "low", "bbox": [0.0, 0.0, 1.0, 1.0]}
]

Only output JSON array."""

        response = self._query(image_b64, prompt, max_tokens=1024)
        if response is None:
            return None

        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            print(f"Error parsing VLM segmentation response: {e}")
            return None

    def density_map_to_tensor(
        self,
        density_grid: np.ndarray,
        output_size: Tuple[int, int],
    ) -> "torch.Tensor":
        """
        Interpolate density grid to full resolution tensor.

        Args:
            density_grid: NxN numpy array of density weights
            output_size: (height, width) of output tensor

        Returns:
            torch.Tensor of shape (1, 1, H, W) with interpolated weights
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for tensor output")

        import torch
        import torch.nn.functional as F

        # Convert to tensor
        grid = torch.from_numpy(density_grid).float()
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Interpolate to output size
        output = F.interpolate(
            grid,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return output

    # ================================================================
    # Face-Specific Methods
    # ================================================================

    def detect_image_type(self, image_path: str) -> str:
        """
        Ask VLM what type of image this is.

        Returns: 'face', 'object', 'scene', or 'animal'
        """
        image_b64 = self._encode_image(image_path)

        prompt = "What is the main subject of this image? Reply with ONE word: face, object, scene, or animal"
        response = self._query(image_b64, prompt, max_tokens=10)

        if response:
            # Extract the keyword
            response_lower = response.strip().lower()
            for keyword in ['face', 'object', 'scene', 'animal']:
                if keyword in response_lower:
                    return keyword
        return 'object'  # default

    def get_face_density_guidance(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed face region importance with landmarks.

        Returns dict with landmarks: {name: [x, y, importance], ...}
        """
        image_b64 = self._encode_image(image_path)

        prompt = """This image contains a face. Identify these regions and their locations.

For each region, provide coordinates as fractions (0-1) where 0,0 is top-left:
- left_eye: [x_center, y_center, importance]
- right_eye: [x_center, y_center, importance]
- nose: [x_center, y_center, importance]
- mouth: [x_center, y_center, importance]
- face_outline: [x_center, y_center, importance]
- hair: [x_center, y_center, importance]

Importance is 0.0 to 1.0 (1.0 = most important for reconstruction).
Eyes should be 1.0, mouth 0.9, nose 0.8, face_outline 0.7, hair 0.5.

Output as JSON only:
{
  "left_eye": [0.35, 0.35, 1.0],
  "right_eye": [0.65, 0.35, 1.0],
  "nose": [0.5, 0.5, 0.8],
  "mouth": [0.5, 0.65, 0.9],
  "face_outline": [0.5, 0.45, 0.7],
  "hair": [0.5, 0.15, 0.5]
}"""

        response = self._query(image_b64, prompt, max_tokens=512)
        if response is None:
            return None

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            print(f"Error parsing face landmarks: {e}")
            return None

    def face_landmarks_to_density(
        self,
        landmarks: Dict[str, List[float]],
        size: int = 256
    ) -> np.ndarray:
        """
        Convert face landmarks to continuous density map.

        Args:
            landmarks: Dict of {name: [x, y, importance]}
            size: Output size (size x size)

        Returns:
            numpy array of shape (size, size) with density values
        """
        density = np.zeros((size, size), dtype=np.float32)

        # Sigma values for different regions (eyes need tighter focus)
        sigmas = {
            'left_eye': 15,
            'right_eye': 15,
            'nose': 25,
            'mouth': 20,
            'face_outline': 40,
            'hair': 50,
        }

        for name, values in landmarks.items():
            if len(values) < 3:
                continue

            x, y, importance = values[0], values[1], values[2]
            cx = int(x * size)
            cy = int(y * size)
            sigma = sigmas.get(name, 30)

            # Create coordinate grids
            yy, xx = np.ogrid[:size, :size]

            # Compute Gaussian
            dist_sq = (xx - cx)**2 + (yy - cy)**2
            gaussian = importance * np.exp(-dist_sq / (2 * sigma**2))

            density += gaussian

        # Normalize to [0, 1]
        if density.max() > 0:
            density = density / density.max()

        return density

    def get_smart_density_guidance(
        self,
        image_path: str,
        grid_size: int = 8,
    ) -> Optional[np.ndarray]:
        """
        Smart density guidance that detects image type and uses appropriate method.

        For faces: Uses landmark-based continuous density map
        For other: Uses grid-based density
        """
        # First detect image type
        image_type = self.detect_image_type(image_path)
        print(f"  Detected image type: {image_type}")

        if image_type == 'face':
            # Use face-specific landmarks
            landmarks = self.get_face_density_guidance(image_path)
            if landmarks:
                print(f"  Got face landmarks: {list(landmarks.keys())}")
                # Generate high-res density from landmarks
                density = self.face_landmarks_to_density(landmarks, size=256)
                # Downsample to grid_size for consistency
                from scipy.ndimage import zoom
                scale = grid_size / 256
                density_grid = zoom(density, scale, order=1)
                return density_grid

        # Fall back to grid-based density
        return self.get_density_guidance(image_path, grid_size)

    # ================================================================
    # Visualization Methods
    # ================================================================

    def visualize_density(
        self,
        image_path: str,
        density_grid: np.ndarray,
        output_path: str = None
    ) -> Image.Image:
        """
        Overlay density heatmap on image.

        Args:
            image_path: Path to original image
            density_grid: NxN density array
            output_path: Optional path to save result

        Returns:
            PIL Image with heatmap overlay
        """
        try:
            from scipy.ndimage import zoom
            import matplotlib.cm as cm
        except ImportError:
            print("scipy and matplotlib required for visualization")
            print("Run: pip install scipy matplotlib")
            return None

        # Load image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)

        # Interpolate density to image size
        h, w = img_np.shape[:2]
        scale_h = h / density_grid.shape[0]
        scale_w = w / density_grid.shape[1]
        density_full = zoom(density_grid, (scale_h, scale_w), order=1)

        # Clip to [0, 1]
        density_full = np.clip(density_full, 0, 1)

        # Create heatmap overlay using jet colormap
        heatmap = cm.jet(density_full)[:, :, :3]  # RGB from colormap
        heatmap = (heatmap * 255).astype(np.uint8)

        # Blend with original (40% heatmap)
        alpha = 0.4
        blended = (img_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)

        result = Image.fromarray(blended)

        if output_path:
            result.save(output_path)
            print(f"Saved density visualization to {output_path}")

        return result

    def visualize_segmentation(
        self,
        image_path: str,
        segments: List[Dict[str, Any]],
        output_path: str = None
    ) -> Image.Image:
        """
        Draw segmentation bboxes on image.

        Args:
            image_path: Path to original image
            segments: List of segment dicts with bbox and importance
            output_path: Optional path to save result

        Returns:
            PIL Image with bounding boxes drawn
        """
        from PIL import ImageDraw, ImageFont

        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        colors = {
            'critical': 'red',
            'high': 'orange',
            'medium': 'yellow',
            'low': 'gray'
        }

        w, h = img.size
        for seg in segments:
            bbox = seg.get('bbox', [0, 0, 1, 1])
            importance = seg.get('importance', 'medium')
            label = seg.get('label', '')

            # Convert fractional to pixel coords
            x0, y0, x1, y1 = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h

            color = colors.get(importance, 'white')
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

            # Draw label
            label_text = f"{label} ({importance})"
            draw.text((x0 + 2, y0 - 18), label_text, fill=color)

        if output_path:
            img.save(output_path)
            print(f"Saved segmentation visualization to {output_path}")

        return img

    def visualize_all(
        self,
        image_path: str,
        output_dir: str,
        grid_size: int = 8
    ) -> Dict[str, str]:
        """
        Run all analyses and save visualizations.

        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            grid_size: Grid size for density

        Returns:
            Dict of output file paths
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = Path(image_path).stem
        outputs = {}

        # Density visualization
        print("Getting density guidance...")
        density = self.get_smart_density_guidance(image_path, grid_size)
        if density is not None:
            # Save raw density as numpy
            np.save(output_dir / f"{name}_density.npy", density)
            outputs['density_npy'] = str(output_dir / f"{name}_density.npy")

            # Save visualization
            viz_path = str(output_dir / f"{name}_density_viz.png")
            self.visualize_density(image_path, density, viz_path)
            outputs['density_viz'] = viz_path

        # Segmentation visualization
        print("Getting segmentation hints...")
        segments = self.get_segmentation_hints(image_path)
        if segments:
            # Save raw segments as JSON
            with open(output_dir / f"{name}_segments.json", 'w') as f:
                json.dump(segments, f, indent=2)
            outputs['segments_json'] = str(output_dir / f"{name}_segments.json")

            # Save visualization
            viz_path = str(output_dir / f"{name}_segments_viz.png")
            self.visualize_segmentation(image_path, segments, viz_path)
            outputs['segments_viz'] = viz_path

        # Depth hints
        print("Getting depth hints...")
        depth = self.get_depth_hints(image_path)
        if depth:
            with open(output_dir / f"{name}_depth.json", 'w') as f:
                json.dump(depth, f, indent=2)
            outputs['depth_json'] = str(output_dir / f"{name}_depth.json")

        print(f"\nSaved {len(outputs)} outputs to {output_dir}")
        return outputs


def test_vlm_guidance():
    """Test VLM guidance functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="VLM Guidance for Gaussian Splatting")
    parser.add_argument('image', nargs='?', help='Path to input image')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate and save visualizations')
    parser.add_argument('--output', '-o', type=str, default='vlm_output',
                        help='Output directory for visualizations (default: vlm_output)')
    parser.add_argument('--grid_size', '-g', type=int, default=8,
                        help='Grid size for density map (default: 8)')
    parser.add_argument('--smart', '-s', action='store_true',
                        help='Use smart density (auto-detect faces)')
    parser.add_argument('--remove_background', '-r', action='store_true',
                        help='Remove background before VLM analysis (matches training preprocessing)')
    parser.add_argument('--url', type=str, default='http://localhost:1234/v1/chat/completions',
                        help='LM Studio API URL')

    args = parser.parse_args()

    print("=" * 60)
    print("VLM Guidance for Gaussian Splatting")
    print("=" * 60)

    vlm = VLMGuidance(api_url=args.url)

    # Check connection
    print("\nChecking LM Studio connection...")
    if vlm.check_connection():
        print("  Connected to LM Studio")
    else:
        print("  ERROR: Cannot connect to LM Studio")
        print("  Make sure LM Studio is running with a VLM model loaded")
        print(f"  URL: {args.url}")
        sys.exit(1)

    # Test with an image if provided
    if args.image:
        image_path = args.image
        print(f"\nProcessing image: {image_path}")

        # Handle background removal if requested
        temp_file = None
        if args.remove_background:
            if not REMBG_AVAILABLE or remove_background is None:
                print("  ERROR: rembg not available for background removal")
                print("  Run: pip install rembg[gpu]")
                sys.exit(1)

            print("  Removing background...")
            try:
                import rembg
                rembg_session = rembg.new_session('u2net')
                raw_img = Image.open(image_path)
                processed_img = remove_background(raw_img, rembg_session)

                # Save processed image to temp file for VLM
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                processed_img.save(temp_file.name)
                image_path = temp_file.name
                print(f"  Background removed, using processed image")

                # Also save to output dir if visualizing
                if args.visualize:
                    output_dir = Path(args.output)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    name = Path(args.image).stem
                    processed_path = output_dir / f"{name}_no_bg.png"
                    processed_img.save(processed_path)
                    print(f"  Saved processed image: {processed_path}")

            except Exception as e:
                print(f"  ERROR removing background: {e}")
                sys.exit(1)

        if args.visualize:
            # Full visualization mode
            print(f"\nGenerating visualizations (grid_size={args.grid_size})...")
            outputs = vlm.visualize_all(image_path, args.output, args.grid_size)

            print("\nGenerated files:")
            for key, path in outputs.items():
                print(f"  {key}: {path}")

        else:
            # Quick test mode
            if args.smart:
                print(f"\n1. Smart density guidance (grid_size={args.grid_size})...")
                density = vlm.get_smart_density_guidance(image_path, grid_size=args.grid_size)
            else:
                print(f"\n1. Density guidance (grid_size={args.grid_size})...")
                density = vlm.get_density_guidance(image_path, grid_size=args.grid_size)

            if density is not None:
                print("  Got density grid:")
                print(density)
            else:
                print("  Failed to get density guidance")

            print("\n2. Depth hints...")
            depth = vlm.get_depth_hints(image_path)
            if depth is not None:
                print("  Got depth hints:")
                for item in depth:
                    print(f"    {item}")
            else:
                print("  Failed to get depth hints")

            print("\n3. Segmentation hints...")
            segments = vlm.get_segmentation_hints(image_path)
            if segments is not None:
                print("  Got segmentation hints:")
                for item in segments:
                    print(f"    {item}")
            else:
                print("  Failed to get segmentation hints")

        # Cleanup temp file
        if temp_file:
            import os
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

    else:
        print("\nNo image provided. Usage:")
        print("  python vlm_guidance.py path/to/image.png")
        print("  python vlm_guidance.py path/to/image.png --visualize --output viz_output/")
        print("  python vlm_guidance.py path/to/image.png --smart --grid_size 16")
        print("  python vlm_guidance.py path/to/image.png --remove_background -v")
        print("\nOptions:")
        print("  --visualize, -v         Generate and save visualizations")
        print("  --output, -o            Output directory (default: vlm_output)")
        print("  --grid_size, -g         Grid size for density (default: 8)")
        print("  --smart, -s             Use smart density (auto-detect faces)")
        print("  --remove_background, -r Remove background before analysis")
        print("  --url                   LM Studio API URL")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    test_vlm_guidance()
