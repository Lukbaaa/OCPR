"""Hilfsfunktionen: normalisierte Box (x,y,w,h) in Pixel umrechnen und zeichnen."""
import matplotlib.patches as patches


def norm_box_to_pixels(box, img_w, img_h):
    """box: (class, x_center, y_center, width, height) normalisiert 0-1 → (x1, y1, x2, y2) Pixeln."""
    xc, yc, w, h = box[1], box[2], box[3], box[4]
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def crop_box(img, box, img_w, img_h):
    """Schneidet eine Box aus dem Bild (PIL oder Tensor). box: (class, x, y, w, h) normalisiert."""
    x1, y1, x2, y2 = norm_box_to_pixels(box, img_w, img_h)
    x1, x2 = max(0, int(x1)), min(img_w, int(x2))
    y1, y2 = max(0, int(y1)), min(img_h, int(y2))
    if hasattr(img, 'numpy'):
        img = img.permute(1, 2, 0).numpy()
        return img[y1:y2, x1:x2]
    return img.crop((x1, y1, x2, y2))


def draw_boxes(ax, img_tensor_or_array, targets, class_names, normalized=True, img_wh=None):
    if hasattr(img_tensor_or_array, 'numpy'):
        img = img_tensor_or_array.permute(1, 2, 0).numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype('uint8')
        h, w = img.shape[:2]
    else:
        img = img_tensor_or_array
        if img.max() <= 1.0:
            img = (img * 255).astype('uint8')
        h, w = img.shape[:2]
    if img_wh is not None:
        w, h = img_wh
    ax.imshow(img)
    for i in range(targets.shape[0]):
        row = targets[i]
        cid = int(row[0])
        if normalized:
            x1, y1, x2, y2 = norm_box_to_pixels(row, w, h)
        else:
            x1, y1, x2, y2 = row[1], row[2], row[3], row[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        name = class_names[cid] if cid < len(class_names) else str(cid)
        ax.text(x1, y1 - 4, name, color='lime', fontsize=8, weight='bold')
    ax.axis('off')
