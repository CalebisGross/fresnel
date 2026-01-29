#!/usr/bin/env python3
"""
VLM-Powered Evaluation for Auto-Tuning.

Integrates LM Studio VLMs into the hyperparameter optimization loop
for semantic-aware quality assessment.

Features:
- Quality scoring alongside SSIM
- Failure diagnosis for bad trials
- Semantic-weighted evaluation
- Trial pattern analysis
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
from PIL import Image

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import VLMGuidance
try:
    import sys
    SCRIPT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(SCRIPT_DIR))
    from utils.vlm_guidance import VLMGuidance
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    VLMGuidance = None


class VLMEvaluator:
    """
    VLM-powered evaluation for auto-tuning.

    Provides semantic quality assessment that captures aspects
    SSIM and other pixel metrics miss.
    """

    def __init__(
        self,
        vlm_url: str = "http://localhost:1234/v1/chat/completions",
        model: str = "qwen2-vl",
        timeout: int = 60,
        verbose: bool = True,
    ):
        """
        Initialize VLM evaluator.

        Args:
            vlm_url: LM Studio API endpoint
            model: Model name to use
            timeout: Request timeout in seconds
            verbose: Print status messages
        """
        self.vlm_url = vlm_url
        self.model = model
        self.timeout = timeout
        self.verbose = verbose
        self.vlm = None

        # Try to connect to VLM
        if VLM_AVAILABLE and REQUESTS_AVAILABLE:
            if self._check_connection():
                self.vlm = VLMGuidance(api_url=vlm_url, model=model, timeout=timeout)
                if verbose:
                    print(f"VLM connected: {vlm_url}")
            else:
                if verbose:
                    print(f"VLM not available (LM Studio not running?)")
        else:
            if verbose:
                print("VLM dependencies not available")

    def _check_connection(self) -> bool:
        """Check if LM Studio is running."""
        try:
            # Try the models endpoint
            base_url = self.vlm_url.rsplit('/v1/', 1)[0]
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        """Check if VLM is available for use."""
        return self.vlm is not None

    def _tensor_to_pil(self, tensor) -> Image.Image:
        """Convert tensor (C, H, W) or (H, W, C) to PIL Image."""
        import torch
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            if tensor.dim() == 3:
                if tensor.shape[0] in [1, 3, 4]:  # (C, H, W)
                    tensor = tensor.permute(1, 2, 0)
            tensor = tensor.numpy()

        # Normalize to 0-255
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).astype(np.uint8)
        else:
            tensor = tensor.astype(np.uint8)

        # Handle grayscale
        if tensor.ndim == 2:
            return Image.fromarray(tensor, mode='L')
        elif tensor.shape[-1] == 1:
            return Image.fromarray(tensor.squeeze(-1), mode='L')
        elif tensor.shape[-1] == 3:
            return Image.fromarray(tensor, mode='RGB')
        elif tensor.shape[-1] == 4:
            return Image.fromarray(tensor, mode='RGBA')
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    def _create_comparison_image(
        self,
        pred_img,
        target_img,
        size: tuple = (256, 256),
    ) -> Image.Image:
        """Create side-by-side comparison image."""
        # Convert tensors to PIL if needed
        if not isinstance(pred_img, Image.Image):
            pred_img = self._tensor_to_pil(pred_img)
        if not isinstance(target_img, Image.Image):
            target_img = self._tensor_to_pil(target_img)

        # Resize
        pred_img = pred_img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        target_img = target_img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        # Combine: target on LEFT, prediction on RIGHT
        combined = Image.new("RGB", (size[0] * 2, size[1]))
        combined.paste(target_img, (0, 0))
        combined.paste(pred_img, (size[0], 0))

        return combined

    def evaluate_quality(
        self,
        pred_img,
        target_img,
    ) -> Dict[str, Any]:
        """
        Evaluate visual quality using VLM.

        Args:
            pred_img: Predicted/rendered image (tensor or PIL)
            target_img: Target/ground truth image (tensor or PIL)

        Returns:
            Dict with quality scores and regions
        """
        if not self.is_available:
            return {'vlm_available': False}

        # Create comparison image
        comparison = self._create_comparison_image(pred_img, target_img)

        # Save temporarily for VLM
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            comparison.save(f.name)
            comparison_path = f.name

        # Also save individual images for evaluate_quality (expects paths)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            if not isinstance(target_img, Image.Image):
                target_pil = self._tensor_to_pil(target_img)
            else:
                target_pil = target_img
            target_pil.save(f.name)
            target_path = f.name

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            if not isinstance(pred_img, Image.Image):
                pred_pil = self._tensor_to_pil(pred_img)
            else:
                pred_pil = pred_img
            pred_pil.save(f.name)
            pred_path = f.name

        try:
            # Use existing evaluate_quality method
            result = self.vlm.evaluate_quality(target_path, pred_path)

            if result:
                return {
                    'vlm_available': True,
                    'similarity': result.get('similarity', 5) / 10.0,
                    'detail': result.get('detail', 5) / 10.0,
                    'color': result.get('color', 5) / 10.0,
                    'sharpness': result.get('sharpness', 5) / 10.0,
                    'structure': result.get('structure', 5) / 10.0,
                    'overall': result.get('overall', 5) / 10.0,
                    'good_regions': result.get('good_regions', []),
                    'needs_work': result.get('needs_work', []),
                }
            else:
                return {'vlm_available': True, 'error': 'VLM returned no result'}

        except Exception as e:
            if self.verbose:
                print(f"VLM evaluation error: {e}")
            return {'vlm_available': True, 'error': str(e)}

        finally:
            # Cleanup temp files
            import os
            for path in [comparison_path, target_path, pred_path]:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def diagnose_failure(
        self,
        pred_img,
        target_img,
    ) -> str:
        """
        Diagnose why a reconstruction failed.

        Args:
            pred_img: Predicted/rendered image
            target_img: Target/ground truth image

        Returns:
            Text description of what went wrong
        """
        if not self.is_available:
            return "VLM not available"

        # Create comparison image
        comparison = self._create_comparison_image(pred_img, target_img)

        # Encode for API
        buffer = io.BytesIO()
        comparison.save(buffer, format='PNG')
        import base64
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        prompt = """Compare the target image (LEFT) with the reconstruction (RIGHT).

The reconstruction has quality issues. Identify the main problems:
1. Is it too blobby or blurry?
2. Are there missing parts?
3. Are colors wrong?
4. Is the shape incorrect?
5. Are there artifacts?

Provide a SHORT diagnosis (1-2 sentences) describing the main issue.
Be specific and actionable (e.g., "Missing fine edge details" not "bad quality")."""

        try:
            response = self.vlm._query(image_b64, prompt, max_tokens=256)
            return response if response else "Could not diagnose"
        except Exception as e:
            return f"Diagnosis error: {e}"

    def get_density_map(
        self,
        target_img,
        grid_size: int = 8,
    ) -> Optional[np.ndarray]:
        """
        Get semantic importance density map.

        Args:
            target_img: Target image to analyze
            grid_size: Output grid size

        Returns:
            numpy array (grid_size, grid_size) with importance weights [0-1]
        """
        if not self.is_available:
            return None

        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            if not isinstance(target_img, Image.Image):
                pil_img = self._tensor_to_pil(target_img)
            else:
                pil_img = target_img
            pil_img.save(f.name)
            img_path = f.name

        try:
            # Use smart density guidance
            density = self.vlm.get_smart_density_guidance(img_path, grid_size)
            return density
        except Exception as e:
            if self.verbose:
                print(f"Density map error: {e}")
            return None
        finally:
            import os
            try:
                os.unlink(img_path)
            except Exception:
                pass

    def analyze_trials(
        self,
        trials: List[Any],
        max_trials: int = 10,
    ) -> str:
        """
        Analyze hyperparameter trial patterns.

        Args:
            trials: List of Optuna trials
            max_trials: Maximum number of trials to include

        Returns:
            VLM analysis of patterns and suggestions
        """
        if not self.is_available:
            return "VLM not available for analysis"

        # Format trial data for LLM
        trial_data = []
        for trial in trials[-max_trials:]:
            if trial.state.name == 'COMPLETE':
                trial_data.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                })

        if not trial_data:
            return "No completed trials to analyze"

        # Sort by value (best first)
        trial_data.sort(key=lambda x: x['value'], reverse=True)

        prompt = f"""Analyze these hyperparameter tuning results for 3D Gaussian reconstruction:

{json.dumps(trial_data, indent=2)}

The objective is to maximize visual quality (SSIM-like metric, higher is better).

Identify:
1. What parameter values tend to work best?
2. Are there any patterns in successful vs unsuccessful trials?
3. What should we try next?

Be concise (3-4 sentences max)."""

        try:
            # Use text-only query (no image)
            response = requests.post(
                self.vlm_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 512,
                    "temperature": 0.3,
                },
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Analysis error: {response.status_code}"

        except Exception as e:
            return f"Analysis error: {e}"

    def full_evaluate(
        self,
        pred_img,
        target_img,
        trial=None,
        include_density: bool = False,
    ) -> Dict[str, Any]:
        """
        Full VLM evaluation suite.

        Args:
            pred_img: Predicted image
            target_img: Target image
            trial: Optional Optuna trial for storing attributes
            include_density: Whether to compute density map

        Returns:
            Dict with all evaluation results
        """
        results = {'vlm_available': self.is_available}

        if not self.is_available:
            return results

        # Quality assessment
        quality = self.evaluate_quality(pred_img, target_img)
        results.update(quality)

        # Failure diagnosis if quality is poor
        similarity = results.get('similarity', 0.5)
        if similarity < 0.5:
            diagnosis = self.diagnose_failure(pred_img, target_img)
            results['diagnosis'] = diagnosis

            # Store in trial if provided
            if trial is not None:
                try:
                    trial.set_user_attr("vlm_diagnosis", diagnosis)
                except Exception:
                    pass

        # Density map if requested
        if include_density:
            density = self.get_density_map(target_img)
            if density is not None:
                results['density_map'] = density

        return results


if __name__ == '__main__':
    """Test VLM evaluator."""
    print("Testing VLMEvaluator...")

    evaluator = VLMEvaluator()

    if evaluator.is_available:
        print("VLM is available!")

        # Create test images
        test_img = Image.new('RGB', (128, 128), color='red')
        pred_img = Image.new('RGB', (128, 128), color='orange')

        print("\nTesting quality evaluation...")
        result = evaluator.evaluate_quality(pred_img, test_img)
        print(f"Result: {result}")

        print("\nTesting failure diagnosis...")
        diagnosis = evaluator.diagnose_failure(pred_img, test_img)
        print(f"Diagnosis: {diagnosis}")

        print("\nTesting density map...")
        density = evaluator.get_density_map(test_img)
        print(f"Density shape: {density.shape if density is not None else 'None'}")

    else:
        print("VLM not available - start LM Studio with a VLM model")
