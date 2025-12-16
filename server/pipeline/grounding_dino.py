from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import os
import numpy as np
from PIL import Image

import torch

# GroundingDINO utility functions (installed from /opt/GroundingDINO)
from groundingdino.util.inference import load_model, predict
from groundingdino.datasets.transforms import Compose
import groundingdino.datasets.transforms as T


@dataclass
class Detection:
    phrase: str
    score: float
    bbox_xyxy: Tuple[float, float, float, float]  # in original image pixel coords


class GroundingDINOEngine:
    """
    Minimal engine for text-conditioned detection.
    - Loads model once at startup.
    - Runs predict(image, caption) and returns bbox in pixel coords.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        self.device = device
        self.model = load_model(config_path, checkpoint_path, device=device)
        self.model.eval()

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        # GroundingDINO expects RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image_tensor, _ = transform(image, None)
        image_tensor = image_tensor.to(self.device)

        # predict() returns boxes in normalized cxcywh format (0~1), and phrases
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        dets: List[Detection] = []
        if boxes is None or len(boxes) == 0:
            return dets

        # boxes: (N, 4) normalized cxcywh
        # Convert to pixel xyxy
        boxes = boxes.cpu().numpy()
        logits = logits.cpu().numpy()

        for i in range(boxes.shape[0]):
            cx, cy, bw, bh = boxes[i]
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h

            # clamp
            x1 = float(max(0.0, min(w - 1.0, x1)))
            y1 = float(max(0.0, min(h - 1.0, y1)))
            x2 = float(max(0.0, min(w - 1.0, x2)))
            y2 = float(max(0.0, min(h - 1.0, y2)))

            phrase = str(phrases[i]) if i < len(phrases) else "object"
            score = float(logits[i])

            dets.append(Detection(phrase=phrase, score=score, bbox_xyxy=(x1, y1, x2, y2)))

        # sort by score desc
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets
