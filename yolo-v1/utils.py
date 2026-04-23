import torch
import json
import os
import math
import config
import torchvision.transforms as T
from PIL import ImageDraw


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]       # (batch, S, S, B, B)

    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = config.EPSILON
    intersection[zero_unions] = 0.0

    return intersection / union


def get_ellipse_iou(p, a, samples=None):
    """
    Approximates IOU between rotated ellipses using grid sampling over each pair's joint AABB.
    Returns tensor with shape (batch, S, S, B, B).
    """
    if samples is None:
        samples = config.ELLIPSE_IOU_SAMPLES

    eps = config.EPSILON

    p_x = bbox_attr(p, 0).unsqueeze(4)
    p_y = bbox_attr(p, 1).unsqueeze(4)
    p_rx = torch.clamp(torch.abs(bbox_attr(p, 2)).unsqueeze(4) / 2.0, min=eps)
    p_ry = torch.clamp(torch.abs(bbox_attr(p, 3)).unsqueeze(4) / 2.0, min=eps)
    p_theta = bbox_attr(p, 4).unsqueeze(4)

    a_x = bbox_attr(a, 0).unsqueeze(3)
    a_y = bbox_attr(a, 1).unsqueeze(3)
    a_rx = torch.clamp(torch.abs(bbox_attr(a, 2)).unsqueeze(3) / 2.0, min=eps)
    a_ry = torch.clamp(torch.abs(bbox_attr(a, 3)).unsqueeze(3) / 2.0, min=eps)
    a_theta = bbox_attr(a, 4).unsqueeze(3)

    x_min = torch.min(p_x - p_rx, a_x - a_rx)
    x_max = torch.max(p_x + p_rx, a_x + a_rx)
    y_min = torch.min(p_y - p_ry, a_y - a_ry)
    y_max = torch.max(p_y + p_ry, a_y + a_ry)

    dtype = p.dtype
    device = p.device
    s = torch.arange(samples, dtype=dtype, device=device)
    s = (s + 0.5) / samples
    gy, gx = torch.meshgrid(s, s, indexing='ij')
    gx = gx.reshape(*([1] * x_min.ndim), -1)
    gy = gy.reshape(*([1] * y_min.ndim), -1)

    dx = (x_max - x_min).unsqueeze(-1)
    dy = (y_max - y_min).unsqueeze(-1)
    sample_x = x_min.unsqueeze(-1) + gx * dx
    sample_y = y_min.unsqueeze(-1) + gy * dy

    def inside_ellipse(cx, cy, rx, ry, theta):
        px = sample_x - cx.unsqueeze(-1)
        py = sample_y - cy.unsqueeze(-1)
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        # Inverse of CCW rotation in image coordinates (x right, y down).
        u = ct * px - st * py
        v = st * px + ct * py
        return ((u / rx.unsqueeze(-1)) ** 2 + (v / ry.unsqueeze(-1)) ** 2) <= 1.0

    p_inside = inside_ellipse(p_x, p_y, p_rx, p_ry, p_theta)
    a_inside = inside_ellipse(a_x, a_y, a_rx, a_ry, a_theta)

    intersection_ratio = (p_inside & a_inside).to(dtype).mean(dim=-1)
    pair_bbox_area = torch.clamp((x_max - x_min) * (y_max - y_min), min=eps)
    intersection = intersection_ratio * pair_bbox_area

    p_area = torch.pi * p_rx * p_ry
    a_area = torch.pi * a_rx * a_ry
    union = p_area + a_area - intersection

    zero_unions = (union <= 0.0)
    union = torch.where(zero_unions, torch.full_like(union, eps), union)
    intersection = torch.where(zero_unions, torch.zeros_like(intersection), intersection)

    return intersection / union


def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def scheduler_lambda(epoch):
    if epoch < config.WARMUP_EPOCHS + 75:
        return 1
    elif epoch < config.WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj):
    folder = os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


def get_dimensions(label):
    size = label['annotation']['size']
    return int(size['width']), int(size['height'])


def get_bounding_boxes(label):
    width, height = get_dimensions(label)
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label['annotation']['object']
    for obj in objects:
        box = obj['bndbox']
        coords = (
            int(int(box['xmin']) * x_scale),
            int(int(box['xmax']) * x_scale),
            int(int(box['ymin']) * y_scale),
            int(int(box['ymax']) * y_scale)
        )
        name = obj['name']
        boxes.append((name, coords))
    return boxes


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::config.BBOX_ATTRS]


def scale_bbox_coord(coord, center, scale):
    return ((coord - center) * scale) + center


def get_overlap(a, b):
    """Returns ellipse overlap ratio max(intersection/areaA, intersection/areaB).

    Input format for each box:
    (cx, cy, width, height, theta, confidence, class_index)
    """

    a_cx, a_cy, a_w, a_h, a_theta, _, _ = a
    b_cx, b_cy, b_w, b_h, b_theta, _, _ = b

    a_rx = max(abs(a_w) / 2.0, config.EPSILON)
    a_ry = max(abs(a_h) / 2.0, config.EPSILON)
    b_rx = max(abs(b_w) / 2.0, config.EPSILON)
    b_ry = max(abs(b_h) / 2.0, config.EPSILON)

    # Tight axis-aligned bounds for rotated ellipses.
    a_ex = math.sqrt((a_rx * math.cos(a_theta)) ** 2 + (a_ry * math.sin(a_theta)) ** 2)
    a_ey = math.sqrt((a_rx * math.sin(a_theta)) ** 2 + (a_ry * math.cos(a_theta)) ** 2)
    b_ex = math.sqrt((b_rx * math.cos(b_theta)) ** 2 + (b_ry * math.sin(b_theta)) ** 2)
    b_ey = math.sqrt((b_rx * math.sin(b_theta)) ** 2 + (b_ry * math.cos(b_theta)) ** 2)

    x_min = min(a_cx - a_ex, b_cx - b_ex)
    x_max = max(a_cx + a_ex, b_cx + b_ex)
    y_min = min(a_cy - a_ey, b_cy - b_ey)
    y_max = max(a_cy + a_ey, b_cy + b_ey)
    pair_area = max((x_max - x_min) * (y_max - y_min), config.EPSILON)

    s = max(int(config.ELLIPSE_IOU_SAMPLES), 3)
    xs = torch.linspace(x_min, x_max, steps=s)
    ys = torch.linspace(y_min, y_max, steps=s)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    def inside(cx, cy, rx, ry, theta):
        px = gx - cx
        py = gy - cy
        ct = math.cos(theta)
        st = math.sin(theta)
        u = ct * px - st * py
        v = st * px + ct * py
        return ((u / rx) ** 2 + (v / ry) ** 2) <= 1.0

    a_inside = inside(a_cx, a_cy, a_rx, a_ry, a_theta)
    b_inside = inside(b_cx, b_cy, b_rx, b_ry, b_theta)
    intersection = (a_inside & b_inside).float().mean().item() * pair_area

    a_area = math.pi * a_rx * a_ry
    b_area = math.pi * b_rx * b_ry
    return max(
        intersection / max(a_area, config.EPSILON),
        intersection / max(b_area, config.EPSILON)
    )


def plot_boxes(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None):
    """Plots bounding boxes on the given image."""

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S
    m = labels.size(dim=0)
    n = labels.size(dim=1)

    bboxes = []
    for i in range(m):
        for j in range(n):
            for k in range((labels.size(dim=2) - config.C) // config.BBOX_ATTRS):
                bbox_start = config.BBOX_ATTRS * k + config.C
                bbox_end = config.BBOX_ATTRS * (k + 1) + config.C
                bbox = labels[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(labels[i, j, :config.C]).item()
                confidence = labels[i, j, class_index].item() * bbox[config.BBOX_ATTRS - 1].item()          # pr(c) * IOU
                if confidence > min_confidence:
                    width = float((bbox[2] * config.IMAGE_SIZE[0]).item())
                    height = float((bbox[3] * config.IMAGE_SIZE[1]).item())
                    cx = float((bbox[0] * config.IMAGE_SIZE[0] + j * grid_size_x).item())
                    cy = float((bbox[1] * config.IMAGE_SIZE[1] + i * grid_size_y).item())
                    theta = float(bbox[4].item())
                    bboxes.append([cx, cy, width, height, theta, confidence, class_index])

    # Sort by highest to lowest confidence
    bboxes = sorted(bboxes, key=lambda x: x[5], reverse=True)

    # Calculate IOUs between each pair of boxes
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    # Non-maximum suppression and render image
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            cx, cy, width, height, _, confidence, class_index = bboxes[i]
            tl = (cx - width / 2.0, cy - height / 2.0)

            # Decrease confidence of other conflicting bboxes
            for j in range(num_boxes):
                other_class = bboxes[j][6]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Annotate image
            draw.rectangle((tl, (tl[0] + width, tl[1] + height)), outline='orange')
            text_pos = (max(0, tl[0]), max(0, tl[1] - 11))
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            text_bbox = draw.textbbox(text_pos, text)
            draw.rectangle(text_bbox, fill='orange')
            draw.text(text_pos, text)
    if file is None:
        image.show()
    else:
        output_dir = os.path.dirname(file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not file.endswith('.png'):
            file += '.png'
        image.save(file)
