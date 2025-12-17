# ABOUTME: Vertex AI Nano Banana Pro (Gemini 3 Pro Image) text-to-image generation node
# ABOUTME: Generates images from text prompts using Google's Nano Banana Pro model

import asyncio
import io
import random

import torch
from google import genai
from google.genai import types
from google.genai.types import Part
from PIL import Image

from .vertexai_utils import base64_to_tensor, pil_to_base64, tensor_to_pil


class SFVertexAINanaBananaPro:
    """
    Generates images from text prompts using Google Vertex AI Nano Banana Pro.

    Also known as Gemini 3 Pro Image, this model combines state-of-the-art
    reasoning capabilities with image generation. Supports up to 14 input
    images for editing/reference and can generate images up to 4096px.
    """

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
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text description of the image to generate. Be specific and detailed for best results.",
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

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text_response")
    FUNCTION = "generate"
    CATEGORY = "Stillfront/VertexAI"

    async def generate(
        self,
        project_id,
        location,
        prompt,
        aspect_ratio,
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

        # Initialize the client if needed
        if self.client is None:
            self.client = genai.Client(
                vertexai=True, project=project_id, location=location
            )

        # Build contents - images first, then prompt
        contents = []
        images = [image1, image2, image3, image4]

        for img_tensor in images:
            if img_tensor is not None:
                pil_img = tensor_to_pil(img_tensor)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format="PNG")
                img_bytes = img_byte_arr.getvalue()
                contents.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # Add the text prompt
        contents.append(prompt)

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
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini 3 Pro Image API call failed: {str(e)}")

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
                            f"[SF VertexAI Nano Banana Pro] Skipping image that could not be decoded: {e}"
                        )
                        continue

        # Combine text responses
        text_response = "\n".join(text_parts) if text_parts else ""

        if not image_tensors:
            raise ValueError(
                "No valid images were returned by the API. Your request was likely blocked by the safety filters or the model chose to respond with text only. Try being more explicit about wanting an image."
            )

        # Stack all images into a batch tensor
        batch_tensor = torch.cat(image_tensors, 0)
        return (batch_tensor, text_response)


# Node registration
NODE_CLASS_MAPPINGS = {"SFVertexAINanaBananaPro": SFVertexAINanaBananaPro}

NODE_DISPLAY_NAME_MAPPINGS = {"SFVertexAINanaBananaPro": "SF VertexAI Nano Banana Pro"}
