import torch
import os
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class YoloPascalVocDataset(Dataset):
    def __init__(self, set_type, normalize=False, augment=False):
        assert set_type in {'train', 'val', 'test'}
        self.root = os.path.join(config.DATA_PATH, config.ELLIPSE_DATASET)
        split_name = self.resolve_split_name(set_type)
        split_path = os.path.join(self.root, 'splits', f'{split_name}.txt')
        with open(split_path, 'r') as file:
            self.ids = [line.strip() for line in file if line.strip()]

        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'labels')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(config.IMAGE_SIZE)
        ])
        self.normalize = normalize
        self.augment = augment
        self.classes = self.load_class_dict_from_ellipse()
        utils.save_class_dict(self.classes)

    def __getitem__(self, i):
        image_id = self.ids[i]
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        label_path = os.path.join(self.label_dir, f'{image_id}.txt')

        image = Image.open(image_path).convert('RGB')
        data = self.transform(image)
        original_data = data.clone()
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        grid_size_x = data.size(dim=2) / config.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / config.S

        # Process labels into the SxSx(B*6+C) ground truth tensor.
        boxes = {}
        class_names = {}
        depth = config.BBOX_ATTRS * config.B + config.C
        ground_truth = torch.zeros((config.S, config.S, depth))
        for class_index, cx, cy, width, height, theta in self.load_ellipse_labels(label_path):
            assert 0 <= class_index < config.C, f'Invalid class index {class_index}'
            mid_x = cx * config.IMAGE_SIZE[0]
            mid_y = cy * config.IMAGE_SIZE[1]
            box_w = width * config.IMAGE_SIZE[0]
            box_h = height * config.IMAGE_SIZE[1]

            # Augment labels
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                mid_x = utils.scale_bbox_coord(mid_x, half_width, scale) + x_shift
                mid_y = utils.scale_bbox_coord(mid_y, half_height, scale) + y_shift
                box_w *= scale
                box_h *= scale

            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or class_index == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :config.C] = one_hot
                    class_names[cell] = class_index

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],     # X coord relative to grid square
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],     # Y coord relative to grid square
                            box_w / config.IMAGE_SIZE[0],                            # Width
                            box_h / config.IMAGE_SIZE[1],                            # Height
                            theta,                                                   # Ellipse orientation
                            1.0                                                     # Confidence
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = config.BBOX_ATTRS * bbox_index + config.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        return len(self.ids)

    def resolve_split_name(self, set_type):
        split_map = {
            'train': config.TRAIN_SPLIT,
            'val': config.VAL_SPLIT,
            'test': config.TEST_SPLIT,
        }
        preferred = split_map[set_type]
        candidates = [preferred]
        if set_type == 'val':
            candidates.extend(['val', 'test', 'trainval'])
        if set_type == 'test':
            candidates.extend(['test', 'val', 'trainval'])
        for split in candidates:
            split_path = os.path.join(self.root, 'splits', f'{split}.txt')
            if os.path.exists(split_path):
                return split
        raise FileNotFoundError(
            f'No valid split file found for set_type={set_type} in {os.path.join(self.root, "splits")}. '
            f'Tried {candidates}.'
        )

    def load_class_dict_from_ellipse(self):
        class_path = os.path.join(self.root, 'classes.txt')
        with open(class_path, 'r') as file:
            class_names = [line.strip() for line in file if line.strip()]
        class_dict = {name: idx for idx, name in enumerate(class_names)}
        assert len(class_dict) == config.C, f'Expected {config.C} classes, got {len(class_dict)}'
        return class_dict

    @staticmethod
    def load_ellipse_labels(label_path):
        labels = []
        with open(label_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                class_id, cx, cy, width, height, theta = line.split()
                labels.append(
                    (
                        int(class_id),
                        float(cx),
                        float(cy),
                        float(width),
                        float(height),
                        float(theta)
                    )
                )
        return labels


if __name__ == '__main__':
    # Display data
    obj_classes = utils.load_class_array()
    train_set = YoloPascalVocDataset('train', normalize=True, augment=True)

    negative_labels = 0
    smallest = 0
    largest = 0
    for data, label, _ in train_set:
        negative_labels += torch.sum(label < 0).item()
        smallest = min(smallest, torch.min(data).item())
        largest = max(largest, torch.max(data).item())
        utils.plot_boxes(data, label, obj_classes, max_overlap=float('inf'))
    # print('num_negatives', negative_labels)
    # print('dist', smallest, largest)
