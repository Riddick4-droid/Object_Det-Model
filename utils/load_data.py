from utils.logger import get_logger
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from configs.config import TRAIN_RATIO, DIRECTORY



logger = get_logger()

#utility function to load data from directory
def load_from_directory(directory:str, train_ratio: float = 0.8):
    """Function to load data from directory and perform train/test split.
    Args:
        directory (str): Path to the directory containing the data.
        train_ratio (float): Ratio of data to be used for training (e.g., 0.8 for 80% train, 20% test).
    Returns:
        tuple: A tuple containing (train_dataset, test_dataset)
    """
    global logger
    try:
        logger.info(f"Loading data from {directory}")
        full_dataset = ImageFolder(directory)
        logger.info(f"Full dataset loaded from {directory} with {len(full_dataset)} images.")

        total_size = len(full_dataset)
        if total_size < 2:
            logger.warning(f"Dataset size ({total_size}) is too small for a train/test split. "
                           f"Returning full dataset as training (if not empty), test_dataset as None.")
            return (full_dataset, None) if total_size > 0 else (None, None)

        train_size = int(train_ratio * total_size)
        # Ensure both train and test sets have at least one sample
        if train_size == 0:
            train_size = 1
        test_size = total_size - train_size
        if test_size == 0:
            train_size = total_size - 1
            test_size = 1

        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        logger.info(f"Dataset split into training ({len(train_dataset)} images) "
                    f"and testing ({len(test_dataset)} images).")
        return train_dataset, test_dataset
    except Exception as e:
        logger.error(f"Error while loading data from directory: {e}")
        return None, None
    
if  __name__ == "__main__":
    logger.info('====Loading data from directory====')
    train_dataset, test_dataset = load_from_directory(directory=DIRECTORY, train_ratio=TRAIN_RATIO)

    if train_dataset:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Training dataset classes: {train_dataset.dataset.classes}")
    if test_dataset:
        logger.info(f"Test dataset size: {len(test_dataset)}")
        logger.info(f"Test dataset classes: {test_dataset.dataset.classes}")
    else:
        logger.info("No test dataset was created (possibly due to small total size).")