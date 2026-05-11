#configurations for the code
#make changes for the parameters and arguments here!
import torch

BATCH_SIZE: int = 8
NUM_WORKERS: int = None
KAGGLE_DATA_PATH: str = "kailaspsudheer/tiny-object-detection"
DESTINATION_DIR = "data_1"
TRAIN_RATIO: float = 0.8
DIRECTORY: str = "C:\\Users\\LENOVO\\Desktop\\object-detection-pipeline\\Object_Det-Model\\data\\data_1"
NUM_CLASSES : int = None
INPUT_SIZE : int = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NMS_CONFIDENCE_THRESHOLD :float = None