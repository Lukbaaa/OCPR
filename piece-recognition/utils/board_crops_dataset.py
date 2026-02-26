"""
Dataset: Aus Brett-Bildern (pieces_on_board) werden die Bounding-Boxen gecroppt.
Jedes Sample = ein Crop (Figur) + Klassen-ID. Label-Format: Klasse X Y Breite Höhe (normalisiert).
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image


CLASS_NAMES = [
    "black-bishop", "black-rook", "black-horse", "black-king", "black-pawn", "black-queen",
    "white-bishop", "white-rook", "white-horse", "white-king", "white-pawn", "white-queen",
]


def _parse_label_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def _norm_to_pixels(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2))


class BoardCropsDataset(Dataset):
    """
    Lädt pieces_on_board (Bilder + Label-Dateien), croppt jede Box → ein Sample = (Crop-Tensor, class_id).
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        image_paths = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.jpeg")) + list(self.img_dir.glob("*.png"))
        self.samples = []
        for img_path in sorted(image_paths):
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            with open(label_path) as f:
                for line in f:
                    parsed = _parse_label_line(line)
                    if parsed is not None:
                        cid, x, y, w, h = parsed
                        self.samples.append((str(img_path), cid, x, y, w, h))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cid, x_center, y_center, w, h = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        x1, y1, x2, y2 = _norm_to_pixels(x_center, y_center, w, h, iw, ih)
        if x2 - x1 < 4 or y2 - y1 < 4:
            x1, y1, x2, y2 = max(0, x1 - 2), max(0, y1 - 2), min(iw, x2 + 2), min(ih, y2 + 2)
        crop = img.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)
        return crop, cid
