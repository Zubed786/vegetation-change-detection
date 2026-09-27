"""
CDVQA (Change Detection-based Visual Question Answering) Dataset Adapter.
Indexes and interfaces with the supplied CDVQA dataset:
- Val / Test / Train question-answer annotations
- Resolution metadata (res_x, res_y)
- Question categories (decrease_or_not, increase_or_not, change_ratio, change_or_not)
- Sample pair catalog for quick loading and interactive demonstration.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CDVQA_ROOT = PROJECT_ROOT / "cdvqa_repo"


@dataclass
class CDVQASample:
    image_id: int
    file_name: str
    res_x: Optional[str]
    res_y: Optional[str]
    sample_questions: List[Dict[str, Any]]


class CDVQAAdapter:
    _instance: Optional["CDVQAAdapter"] = None

    def __init__(self, root_dir: Optional[Path] = None):
        self.root = root_dir or CDVQA_ROOT
        self.val_images_file = self.root / "Val_images.json"
        self.val_questions_file = self.root / "Val_questions.json"
        self.val_answers_file = self.root / "Val_answers.json"

        self.images: List[Dict[str, Any]] = []
        self.questions_by_img: Dict[int, List[Dict[str, Any]]] = {}
        self.is_loaded = False
        self._load()

    def _load(self):
        if not self.val_images_file.exists():
            return
        try:
            with open(self.val_images_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.images = data.get("images", [])

            if self.val_questions_file.exists():
                with open(self.val_questions_file, "r", encoding="utf-8") as f:
                    q_data = json.load(f)
                    all_questions = q_data.get("questions", [])
                    for q in all_questions:
                        img_id = q.get("img_id")
                        if img_id not in self.questions_by_img:
                            self.questions_by_img[img_id] = []
                        self.questions_by_img[img_id].append(q)

            self.is_loaded = True
        except Exception as e:
            print(f"Warning: Could not fully load CDVQA dataset: {e}")

    @classmethod
    def get_instance(cls) -> "CDVQAAdapter":
        if cls._instance is None:
            cls._instance = CDVQAAdapter()
        return cls._instance

    def get_sample_catalogue(self, limit: int = 15) -> List[CDVQASample]:
        """
        Returns a curated list of representative CDVQA samples across different question types.
        """
        samples: List[CDVQASample] = []
        for img in self.images[:limit]:
            img_id = img["id"]
            fn = img.get("file_name", f"{img_id:05d}.png")
            rx = img.get("res_x")
            ry = img.get("res_y")
            qs = self.questions_by_img.get(img_id, [])

            # Filter or pick diverse questions
            picked_qs = []
            seen_types = set()
            for q in qs:
                qtype = q.get("type")
                if qtype not in seen_types:
                    picked_qs.append({
                        "id": q["id"],
                        "type": qtype,
                        "question": q["question"],
                    })
                    seen_types.add(qtype)
                if len(picked_qs) >= 4:
                    break

            samples.append(
                CDVQASample(
                    image_id=img_id,
                    file_name=fn,
                    res_x=rx,
                    res_y=ry,
                    sample_questions=picked_qs,
                )
            )
        return samples

    def find_image_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Finds resolution and metadata by filename in the dataset."""
        for img in self.images:
            if img.get("file_name") == filename:
                return img
        return None
