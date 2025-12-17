# ABOUTME: Vertex AI Nano Banana (Gemini Image) text-to-image generation node
# ABOUTME: Generates images from text prompts using Google's Nano Banana Pro or Nano Banana models

import io

import numpy as np
import torch
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Part
from PIL import Image


class SFVertexAINanaBananaPro:
    """
    Generates images from text prompts using Google Vertex AI Nano Banana models.

    - Nano Banana Pro (gemini-3-pro-image-preview): Up to 4096px (4K), best quality, preview
    - Nano Banana (gemini-2.5-flash-image): Up to 1024px (1K), faster, stable

    Supports up to 4 input images for editing/reference.
    """

    # Available models
    MODELS = [
        "gemini-2.5-flash-image",  # Nano Banana - stable, 1024px max
        "gemini-3-pro-image-preview",  # Nano Banana Pro - preview, 4096px max
    ]

    # Supported aspect ratios
    ASPECT_RATIOS = [
        "1:1",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "3:2",
        "2:3",
        "4:5",
        "5:4",
        "21:9",
    ]

    # Supported image sizes (Nano Banana Pro supports all, Nano Banana only 1K)
    IMAGE_SIZES = [
        "1K",
        "2K",
        "4K",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_id": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Google Cloud project ID",
                    },
                ),
                "location": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "global",
                        "tooltip": "Google Cloud region (use 'global' for Gemini/Nano Banana models)",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text description of the image to generate. Be specific and detailed for best results.",
                    },
                ),
                "model": (
                    cls.MODELS,
                    {
                        "default": "gemini-3-pro-image-preview",
                        "tooltip": "Nano Banana (2.5 flash, stable, 1K only) or Nano Banana Pro (3 pro, preview, up to 4K)",
                    },
                ),
                "aspect_ratio": (
                    cls.ASPECT_RATIOS,
                    {
                        "default": "1:1",
                        "tooltip": "Output image aspect ratio",
                    },
                ),
                "image_size": (
                    cls.IMAGE_SIZES,
                    {
                        "default": "1K",
                        "tooltip": "Output image size. Nano Banana only supports 1K. Nano Banana Pro supports 1K/2K/4K.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                        "tooltip": "Random seed for reproducible results (0 for random)",
                    },
                ),
            },
            "optional": {
                "image1": (
                    "IMAGE",
                    {"tooltip": "Optional reference/input image 1"},
                ),
                "image2": (
                    "IMAGE",
                    {"tooltip": "Optional reference/input image 2"},
                ),
                "image3": (
                    "IMAGE",
                    {"tooltip": "Optional reference/input image 3"},
                ),
                "image4": (
                    "IMAGE",
                    {"tooltip": "Optional reference/input image 4"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "text_response", "thoughts")
    FUNCTION = "generate"
    CATEGORY = "Stillfront/VertexAI"

    def _tensor_to_pil(self, tensor):
        """Convert a ComfyUI tensor to PIL Image."""
        if tensor is None:
            return None
        image_np = tensor.squeeze().cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        return Image.fromarray(image_np)

    def _pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor (RGB, not RGBA)."""
        pil_image = pil_image.convert("RGB")
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)[None,]

    def generate(
        self,
        project_id,
        location,
        prompt,
        model,
        aspect_ratio,
        image_size,
        seed,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
    ):
        if not project_id:
            raise ValueError(
                "project_id is required. Provide your Google Cloud project ID."
            )

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Validate image_size for Nano Banana (only supports 1K)
        if model == "gemini-2.5-flash-image" and image_size != "1K":
            print(
                f"[SF VertexAI Nano Banana Pro] Warning: Nano Banana (gemini-2.5-flash-image) only supports 1K. Falling back to 1K."
            )
            image_size = "1K"

        # Initialize the client
        client = genai.Client(vertexai=True, project=project_id, location=location)

        # Build contents - images first, then prompt
        contents = []
        images = [image1, image2, image3, image4]

        for img_tensor in images:
            if img_tensor is not None:
                pil_img = self._tensor_to_pil(img_tensor)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format="PNG")
                img_bytes = img_byte_arr.getvalue()
                contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # Add the text prompt
        contents.append(prompt)

        # Build generation configuration with image config as dict
        config = GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
            seed=seed if seed > 0 else None,
            image_config={
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
            },
        )

        # Call the Gemini API
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Nano Banana API call failed ({model}): {str(e)}")

        # Check finish reason for errors
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason != types.FinishReason.STOP:
                raise ValueError(
                    f"Generation stopped due to: {finish_reason}. Try rephrasing your prompt."
                )

        # Process response - extract images, text, and thoughts
        image_tensors = []
        text_parts = []
        thought_parts = []

        for candidate in response.candidates:
            for part in candidate.content.parts:
                # Check if this is a thought
                if hasattr(part, "thought") and part.thought:
                    if part.text:
                        thought_parts.append(part.text)
                    continue

                if part.text:
                    text_parts.append(part.text)
                elif part.inline_data:
                    try:
                        pil_image = Image.open(io.BytesIO(part.inline_data.data))
                        image_tensors.append(self._pil_to_tensor(pil_image))
                    except (ValueError, AttributeError, OSError) as e:
                        print(
                            f"[SF VertexAI Nano Banana Pro] Skipping image that could not be decoded: {e}"
                        )
                        continue

        # Combine text responses
        text_response = "\n".join(text_parts) if text_parts else ""
        thoughts = "\n".join(thought_parts) if thought_parts else ""

        if not image_tensors:
            raise ValueError(
                "No valid images were returned by the API. Your request was likely blocked by safety filters "
                "or the model chose to respond with text only. Try being more explicit about wanting an image."
            )

        # Stack all images into a batch tensor
        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor, text_response, thoughts)


# Node registration
NODE_CLASS_MAPPINGS = {"SFVertexAINanaBananaPro": SFVertexAINanaBananaPro}

NODE_DISPLAY_NAME_MAPPINGS = {"SFVertexAINanaBananaPro": "SF VertexAI Nano Banana Pro"}
