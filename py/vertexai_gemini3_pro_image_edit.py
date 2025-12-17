# ABOUTME: Vertex AI Nano Banana (Gemini Image) image editing node
# ABOUTME: Edits images using text instructions with Google's Nano Banana Pro or Nano Banana models

import asyncio
import io
import random

import torch
from google import genai
from google.genai import types
from google.genai.types import Part
from PIL import Image

from .vertexai_utils import base64_to_tensor, pil_to_base64, tensor_to_pil


class SFVertexAINanaBananaProEdit:
    """
    Edits images using text instructions with Google Vertex AI Nano Banana models.

    - Nano Banana Pro (gemini-3-pro-image-preview): Up to 4096px, best quality, preview
    - Nano Banana (gemini-2.5-flash-image): Up to 1024px, faster, stable

    Supports multi-turn conversational editing.
    """

    # Available models
    MODELS = [
        "gemini-2.5-flash-image",  # Nano Banana - stable, 1024px
        "gemini-3-pro-image-preview",  # Nano Banana Pro - preview, 4096px
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

    def __init__(self):
        self.client = None

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
                "image": (
                    "IMAGE",
                    {"tooltip": "Input image to edit"},
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
                        "default": "gemini-2.5-flash-image",
                        "tooltip": "Nano Banana (2.5 flash, stable, 1024px) or Nano Banana Pro (3 pro, preview, 4096px)",
                    },
                ),
                "aspect_ratio": (
                    cls.ASPECT_RATIOS,
                    {
                        "default": "1:1",
                        "tooltip": "Output image aspect ratio",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": random.randint(0, 2147483647),
                        "min": 0,
                        "max": 2147483647,
                        "control_after_generate": True,
                        "tooltip": "Random seed for reproducible results",
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional second reference image for style transfer or combining elements",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "text_response")
    FUNCTION = "edit"
    CATEGORY = "Stillfront/VertexAI"

    async def edit(
        self,
        project_id,
        location,
        image,
        edit_instruction,
        model,
        aspect_ratio,
        seed,
        reference_image=None,
    ):
        if not project_id:
            raise ValueError(
                "project_id is required. Provide your Google Cloud project ID."
            )

        if not edit_instruction.strip():
            raise ValueError("edit_instruction cannot be empty")

        # Initialize the client if needed
        if self.client is None:
            self.client = genai.Client(
                vertexai=True, project=project_id, location=location
            )

        # Build contents - images first, then edit instruction
        contents = []

        # Add the main image to edit
        pil_img = tensor_to_pil(image)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # Add optional reference image
        if reference_image is not None:
            ref_pil_img = tensor_to_pil(reference_image)
            ref_byte_arr = io.BytesIO()
            ref_pil_img.save(ref_byte_arr, format="PNG")
            ref_bytes = ref_byte_arr.getvalue()
            contents.append(Part.from_bytes(data=ref_bytes, mime_type="image/png"))

        # Add the edit instruction
        contents.append(edit_instruction)

        # Build configuration
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
            seed=seed,
        )

        # Call the Gemini API
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Nano Banana Edit API call failed: {str(e)}")

        # Process response - extract both images and text
        image_tensors = []
        text_parts = []

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.inline_data:
                    try:
                        pil_image = Image.open(
                            io.BytesIO(part.inline_data.data)
                        ).convert("RGBA")
                        image_tensors.append(base64_to_tensor(pil_to_base64(pil_image)))
                    except (ValueError, AttributeError) as e:
                        print(
                            f"[SF VertexAI Nano Banana Pro Edit] Skipping image that could not be decoded: {e}"
                        )
                        continue

        # Combine text responses
        text_response = "\n".join(text_parts) if text_parts else ""

        if not image_tensors:
            raise ValueError(
                "No valid images were returned by the API. Your request was likely blocked by the safety filters or the model chose to respond with text only. Try rephrasing your edit instruction."
            )

        # Stack all images into a batch tensor
        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor, text_response)


# Node registration
NODE_CLASS_MAPPINGS = {"SFVertexAINanaBananaProEdit": SFVertexAINanaBananaProEdit}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SFVertexAINanaBananaProEdit": "SF VertexAI Nano Banana Pro Edit"
}
