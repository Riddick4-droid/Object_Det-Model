import kagglehub
import os
import shutil
from utils.logger import get_logger
from configs.config import KAGGLE_DATA_PATH, DESTINATION_DIR

logger = get_logger()

def download_data(kaggle_path:str,destination_dir:str):
    """Function to download dataset from kagglehub
    Args:
        kaggle_path (str): Path to the dataset on kaggle
        destination_dir (str): Where we want to store the downloaded data
    """

    global logger
    try:
        logger.info(f"Downloading data from {kaggle_path}")
        # kagglehub.dataset_download downloads and extracts the dataset,
        # returning the path to the root directory of the extracted files.
        downloaded_path = kagglehub.dataset_download(kaggle_path)
        logger.info(f"Dataset downloaded and extracted to {downloaded_path}")

        # Set the full destination directory path
        full_destination_dir = f".\\data\\{destination_dir}"
        os.makedirs(full_destination_dir, exist_ok=True)
        for item in os.listdir(downloaded_path):
            s = os.path.join(downloaded_path, item)
            d = os.path.join(full_destination_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d) # copy2 preserves metadata
        logger.info(f"Contents moved/copied from {downloaded_path} to {full_destination_dir}")

    except Exception as e:
        logger.error(f"Error while downloading data {e}")

if __name__=="__main__":
    logger.info("===Downloading Data===")
    download_data(kaggle_path=KAGGLE_DATA_PATH,destination_dir=DESTINATION_DIR)