"""
Dataset für Schachbretter mit Bounding-Box-Labels (Bilder laden, nicht Crops).
Label-Format pro Zeile: Klasse X Y Breite Höhe (normalisiert 0–1, Mittelpunkt X,Y).
Nutzt Ordner pieces_on_board/ (train/, valid/, test/).
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

CLASS_NAMES = [
    'black-bishop', 'black-rook', 'black-horse', 'black-king', 'black-pawn', 'black-queen',
    'white-bishop', 'white-rook', 'white-horse', 'white-king', 'white-pawn', 'white-queen',
]


def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    return (
        int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
    )


def load_labels(path: str):
    out = []
    with open(path) as f:
        for line in f:
            parsed = parse_label_line(line)
            if parsed is not None:
                out.append(parsed)
    return out


class PiecesOnBoardDataset(Dataset):
    """Bilder von Schachbrettern + Labels (Klasse, x_center, y_center, width, height, 0–1)."""
    def __init__(self, root_dir, split="train", transform=None, img_size=None):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.img_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        self.image_paths = (
            sorted(self.img_dir.glob("*.jpg"))
            + sorted(self.img_dir.glob("*.jpeg"))
            + sorted(self.img_dir.glob("*.png"))
        )
        self.image_paths = [p for p in self.image_paths if (self.label_dir / (p.stem + ".txt")).exists()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")
        image = Image.open(img_path).convert("RGB")
        if self.img_size:
            image = image.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)
        labels = load_labels(str(label_path))
        targets = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))
        if self.transform:
            image = self.transform(image)
        return image, targets


def collate_pieces_on_board(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
