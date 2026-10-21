#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local OpenCLIP scorer for Yupoo product images.

- Runs locally.
- No OpenAI API.
- Uses CPU when CUDA is unavailable.
- Loads the model only when first needed.
- Returns None when AI is unavailable, allowing the old selector to continue.
"""

import io
import hashlib
import threading
from functools import lru_cache
from typing import Optional, Dict, Any

import torch
import open_clip
from PIL import Image, UnidentifiedImageError


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "ViT-B-32"
PRETRAINED_NAME = "laion2b_s34b_b79k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# AI confidence required before applying a strong bonus.
AI_GOOD_THRESHOLD = 0.58

# AI confidence below this value receives a penalty.
AI_BAD_THRESHOLD = 0.38


# =========================================================
# PROMPTS
# =========================================================

# Positive prompts describe images suitable for supplier/product checking.
POSITIVE_PROMPTS = [
    "a clear ecommerce product photo showing the complete product",
    "a full product visible in the image",
    "a clean supplier product image",
    "a centered fashion product photo",
    "a complete sneaker product photo",
    "a complete clothing product photo",
    "a full view of a shoe",
    "a full view of a garment",
    "a person clearly wearing the complete fashion product",
]

# Negative prompts describe unsuitable hero images.
NEGATIVE_PROMPTS = [
    "a close up detail of fabric",
    "a close up detail of a logo",
    "a close up detail of stitching",
    "a close up detail of a zipper",
    "a product tag or clothing label",
    "a size chart",
    "a measurement chart",
    "product packaging only",
    "a cropped partial product",
    "an empty background",
    "a website logo",
    "a blurry low quality image",
]


# =========================================================
# LAZY MODEL STATE
# =========================================================

_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_POSITIVE_FEATURES = None
_NEGATIVE_FEATURES = None

_MODEL_LOCK = threading.Lock()
_MODEL_FAILED = False


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize CLIP feature vectors."""
    return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _load_model() -> bool:
    """
    Load OpenCLIP only when the first image needs scoring.

    Returns:
        True when the model is ready.
        False when loading failed.
    """
    global _MODEL
    global _PREPROCESS
    global _TOKENIZER
    global _POSITIVE_FEATURES
    global _NEGATIVE_FEATURES
    global _MODEL_FAILED

    if _MODEL is not None:
        return True

    if _MODEL_FAILED:
        return False

    with _MODEL_LOCK:
        if _MODEL is not None:
            return True

        if _MODEL_FAILED:
            return False

        try:
            print(
                f"[AI SELECTOR] Loading OpenCLIP "
                f"{MODEL_NAME}/{PRETRAINED_NAME} on {DEVICE}"
            )

            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME,
                pretrained=PRETRAINED_NAME,
            )

            tokenizer = open_clip.get_tokenizer(MODEL_NAME)

            model = model.to(DEVICE)
            model.eval()

            positive_tokens = tokenizer(POSITIVE_PROMPTS).to(DEVICE)
            negative_tokens = tokenizer(NEGATIVE_PROMPTS).to(DEVICE)

            with torch.inference_mode():
                positive_features = model.encode_text(positive_tokens)
                negative_features = model.encode_text(negative_tokens)

                positive_features = _normalize_features(positive_features)
                negative_features = _normalize_features(negative_features)

            _MODEL = model
            _PREPROCESS = preprocess
            _TOKENIZER = tokenizer
            _POSITIVE_FEATURES = positive_features
            _NEGATIVE_FEATURES = negative_features

            print("[AI SELECTOR] OpenCLIP ready")
            return True

        except Exception as e:
            _MODEL_FAILED = True
            print(f"[AI SELECTOR DISABLED] Could not load OpenCLIP: {e}")
            return False


def _prepare_image(content: bytes) -> Image.Image:
    """Open image bytes safely and return an RGB image."""
    image = Image.open(io.BytesIO(content))
    image.load()
    return image.convert("RGB")


def _content_hash(content: bytes) -> str:
    """Create a stable cache key for image bytes."""
    return hashlib.sha1(content).hexdigest()


@lru_cache(maxsize=512)
def _score_cached(image_hash: str, content: bytes) -> Optional[Dict[str, Any]]:
    """
    Internal cached OpenCLIP scoring.

    image_hash is included so equivalent images can be identified clearly.
    """
    del image_hash

    if not _load_model():
        return None

    try:
        image = _prepare_image(content)
        image_tensor = _PREPROCESS(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            image_features = _MODEL.encode_image(image_tensor)
            image_features = _normalize_features(image_features)

            positive_similarities = (
                image_features @ _POSITIVE_FEATURES.T
            ).squeeze(0)

            negative_similarities = (
                image_features @ _NEGATIVE_FEATURES.T
            ).squeeze(0)

            best_positive_value, best_positive_index = (
                positive_similarities.max(dim=0)
            )
            best_negative_value, best_negative_index = (
                negative_similarities.max(dim=0)
            )

            positive_mean = positive_similarities.mean()
            negative_mean = negative_similarities.mean()

            # Compare positive and negative meaning through a two-class softmax.
            comparison_logits = torch.stack([
                positive_mean,
                negative_mean,
            ]) * 100.0

            confidence = torch.softmax(
                comparison_logits,
                dim=0,
            )[0].item()

        best_positive_prompt = POSITIVE_PROMPTS[
            int(best_positive_index.item())
        ]
        best_negative_prompt = NEGATIVE_PROMPTS[
            int(best_negative_index.item())
        ]

        return {
            "confidence": float(confidence),
            "positive_similarity": float(positive_mean.item()),
            "negative_similarity": float(negative_mean.item()),
            "best_positive_similarity": float(
                best_positive_value.item()
            ),
            "best_negative_similarity": float(
                best_negative_value.item()
            ),
            "best_positive_prompt": best_positive_prompt,
            "best_negative_prompt": best_negative_prompt,
            "device": DEVICE,
        }

    except (UnidentifiedImageError, OSError) as e:
        print(f"[AI SELECTOR IMAGE ERROR] {e}")
        return None

    except Exception as e:
        print(f"[AI SELECTOR SCORE ERROR] {e}")
        return None


def score_product_image_bytes(
    content: bytes,
) -> Optional[Dict[str, Any]]:
    """
    Score one product image.

    Returns a dictionary such as:

        {
            "confidence": 0.72,
            "positive_similarity": 0.25,
            "negative_similarity": 0.20,
            ...
        }

    Returns None if AI is unavailable or the image cannot be processed.
    """
    if not content:
        return None

    try:
        image_hash = _content_hash(content)
        return _score_cached(image_hash, content)
    except Exception as e:
        print(f"[AI SELECTOR ERROR] {e}")
        return None


def calculate_ai_score_adjustment(
    confidence: Optional[float],
) -> float:
    """
    Convert AI confidence into a bonus or penalty for the old selector.

    The old rule-based score remains active.

    Returns:
        Positive number: AI believes this is a good complete product image.
        Negative number: AI believes this is probably a detail/bad image.
        0: AI unavailable or uncertain.
    """
    if confidence is None:
        return 0.0

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return 0.0

    if confidence >= 0.82:
        return 420.0

    if confidence >= 0.72:
        return 300.0

    if confidence >= AI_GOOD_THRESHOLD:
        return 170.0

    if confidence <= 0.20:
        return -520.0

    if confidence <= 0.30:
        return -340.0

    if confidence <= AI_BAD_THRESHOLD:
        return -180.0

    return 0.0