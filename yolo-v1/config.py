import os
import torchvision.transforms as T

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(PROJECT_DIR, '..', 'data'))
CLASSES_PATH = os.path.join(DATA_PATH, 'classes.json')
ELLIPSE_DATASET = 'VOC2012_ellipse'
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'
USE_PRETRAINED_BACKBONE = True
ELLIPSE_IOU_SAMPLES = 15
WANDB_ENABLED = True
WANDB_MODE = "offline"  # "offline" on restricted clusters, "online" when internet is available
WANDB_PROJECT = "yolo-v1-ellipse"
WANDB_ENTITY = None
WANDB_RUN_NAME = None

BATCH_SIZE = 128
EPOCHS = 100
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-4

EPSILON = 1E-6
IMAGE_SIZE = (448, 448)

S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 20      # Number of classes in the dataset
BBOX_ATTRS = 6   # [x, y, w, h, theta, confidence]
