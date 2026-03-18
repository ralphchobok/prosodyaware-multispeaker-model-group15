import kagglehub
import shutil
from pathlib import Path

target_dir = Path("..") / "datasets"
target_dir.mkdir(parents=True, exist_ok=True)

download_path = kagglehub.dataset_download(
    "nguyenthanhlim/emotional-speech-dataset-esd"
)

shutil.move(download_path, target_dir / Path(download_path).name)

print("Moved dataset to:", target_dir)