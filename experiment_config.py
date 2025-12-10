from pathlib import Path
import sys


class Configuration(object):
    def __init__(self, mode="2D", data_dir=None) -> None:

        # Working directory - use repo-local path
        self.WORKDIR = Path(__file__).resolve().parent
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )
        self.MODEL_3D_RESNET = (
            self.RESOURCES / "resnet_18_23dataset.pth"
        )
        
        # Data parameters - Auto-detect dataset location
        default_block_dir = Path(__file__).resolve().parent.parent / "dataset" / "luna25_nodule_blocks"
        repo_dataset_dir = Path(__file__).resolve().parent.parent / "dataset"

        if default_block_dir.exists():
            self.DATADIR = default_block_dir
        elif (repo_dataset_dir / "image").exists() and (repo_dataset_dir / "metadata").exists():
            # use the repo-local dataset folder which contains image/ and metadata/
            self.DATADIR = repo_dataset_dir
        else:
            # fallback to the original default
            self.DATADIR = default_block_dir

        if (data_dir):
            self.DATADIR = Path.cwd() / data_dir
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path(__file__).resolve().parent.parent / "luna25-baseline-public" / "dataset" / "luna25_csv"
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        self.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
            
        self.EXPERIMENT_NAME = "LUNA25-I3D-DUO"
        self.MODE = mode  # Can be overridden by command line argument (2D or 3D)

        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 4
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 16
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.EPOCHS = 100
        self.PATIENCE = 20
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 5e-4


# Default configuration (can be overridden by command line args in train.py)
config = Configuration(mode="2D")