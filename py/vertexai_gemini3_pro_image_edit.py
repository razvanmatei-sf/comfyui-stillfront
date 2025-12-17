# ABOUTME: Vertex AI Nano Banana (Gemini Image) image editing node
# ABOUTME: Edits images using text instructions with Google's Nano Banana Pro or Nano Banana models

import io

import numpy as np
import torch
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, Part
from PIL import Image


class SFVertexAINanaBananaProEdit:
    """
    Edits images using text instructions with Google Vertex AI Nano Banana models.

    - Nano Banana Pro (gemini-3-pro-image-preview): Up to 4096px (4K), best quality, preview
    - Nano Banana (gemini-2.5-flash-image): Up to 1024px (1K), faster, stable

    Supports up to 14 input images for editing, style transfer, and combining elements.
    Use the image_count dropdown and "Update inputs" button to add more image slots.
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
                "edit_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text instruction describing the edit to make (e.g., 'Make it look like a cartoon' or 'Change the background to a beach')",
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
                        "control_after_generate": "fixed",
                        "tooltip": "Random seed for reproducible results (0 for random)",
                    },
                ),
                "image_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 14,
                        "tooltip": "Number of image input slots (click 'Update inputs' button after changing)",
                    },
                ),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Input image 1"}),
                "image_2": ("IMAGE", {"tooltip": "Input image 2"}),
                "image_3": ("IMAGE", {"tooltip": "Input image 3"}),
                "image_4": ("IMAGE", {"tooltip": "Input image 4"}),
                "image_5": ("IMAGE", {"tooltip": "Input image 5"}),
                "image_6": ("IMAGE", {"tooltip": "Input image 6"}),
                "image_7": ("IMAGE", {"tooltip": "Input image 7"}),
                "image_8": ("IMAGE", {"tooltip": "Input image 8"}),
                "image_9": ("IMAGE", {"tooltip": "Input image 9"}),
                "image_10": ("IMAGE", {"tooltip": "Input image 10"}),
                "image_11": ("IMAGE", {"tooltip": "Input image 11"}),
                "image_12": ("IMAGE", {"tooltip": "Input image 12"}),
                "image_13": ("IMAGE", {"tooltip": "Input image 13"}),
                "image_14": ("IMAGE", {"tooltip": "Input image 14"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("edited_image", "text_response", "thoughts")
    FUNCTION = "edit"
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

    def edit(
        self,
        project_id,
        location,
        edit_instruction,
        model,
        aspect_ratio,
        image_size,
        seed,
        image_count,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        image_11=None,
        image_12=None,
        image_13=None,
        image_14=None,
    ):
        if not project_id:
            raise ValueError(
                "project_id is required. Provide your Google Cloud project ID."
            )

        if not edit_instruction.strip():
            raise ValueError("edit_instruction cannot be empty")

        # Collect all image inputs
        all_images = [
            image_1,
            image_2,
            image_3,
            image_4,
            image_5,
            image_6,
            image_7,
            image_8,
            image_9,
            image_10,
            image_11,
            image_12,
            image_13,
            image_14,
        ]

        # Filter to only include provided images
        images = [img for img in all_images if img is not None]

        if not images:
            raise ValueError("At least one image input is required for editing")

        # Validate image_size for Nano Banana (only supports 1K)
        if model == "gemini-2.5-flash-image" and image_size != "1K":
            print(
                f"[SF VertexAI Nano Banana Pro Edit] Warning: Nano Banana (gemini-2.5-flash-image) only supports 1K. Falling back to 1K."
            )
            image_size = "1K"

        # Initialize the client
        client = genai.Client(vertexai=True, project=project_id, location=location)

        # Build contents - images first, then edit instruction
        contents = []

        # Add all provided images (up to 14)
        for img_tensor in images[:14]:
            pil_img = self._tensor_to_pil(img_tensor)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()
            contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # Add the edit instruction with aspect ratio and size hints
        # Note: aspect_ratio and image_size are passed via prompt instructions
        # as the SDK doesn't support image_config parameter directly
        size_instruction = f"Generate the image with aspect ratio {aspect_ratio} and resolution {image_size}."
        contents.append(f"{size_instruction} {edit_instruction}")

        # Build generation configuration
        config = GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
            seed=seed if seed > 0 else None,
        )

        # Call the Gemini API
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Nano Banana Edit API call failed ({model}): {str(e)}")

        # Check finish reason for errors
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason != types.FinishReason.STOP:
                raise ValueError(
                    f"Generation stopped due to: {finish_reason}. Try rephrasing your edit instruction."
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
                            f"[SF VertexAI Nano Banana Pro Edit] Skipping image that could not be decoded: {e}"
                        )
                        continue

        # Combine text responses
        text_response = "\n".join(text_parts) if text_parts else ""
        thoughts = "\n".join(thought_parts) if thought_parts else ""

        if not image_tensors:
            raise ValueError(
                "No valid images were returned by the API. Your request was likely blocked by safety filters "
                "or the model chose to respond with text only. Try rephrasing your edit instruction."
            )

        # Stack all images into a batch tensor
        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor, text_response, thoughts)


# Node registration
NODE_CLASS_MAPPINGS = {"SFVertexAINanaBananaProEdit": SFVertexAINanaBananaProEdit}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SFVertexAINanaBananaProEdit": "SF VertexAI Nano Banana Pro Edit"
}
