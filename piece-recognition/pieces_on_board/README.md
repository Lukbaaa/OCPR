# Schachbretter-Dataset

Nur Daten: Bilder + Bounding-Box-Labels für Objekterkennung / Crop-Klassifikation.

## Inhalt

- **train/images/** und **train/labels/**
- **valid/images/** und **valid/labels/**
- **test/images/** und **test/labels/**
- **data.yaml** – Klassenanzahl und -namen (Reihenfolge = Klassen-ID in den Labels)

## Label-Format (pro Zeile)

```
Klasse  X  Y  Breite  Höhe
```

Alle Werte normalisiert 0–1. X, Y = Mittelpunkt der Box.  
Zuordnung Klasse → Name siehe **data.yaml** (`names`).

## Code

Laden und Croppen passiert in **piece-recognition/**:
- `board_crops_dataset.py` – Crops pro Box für Classifier-Training
- `pieces_on_board_dataset.py` – volle Brett-Bilder + Labels (z. B. für Visualisierung)
