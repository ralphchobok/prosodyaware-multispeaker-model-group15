import os
import urllib.request
import tarfile
from tqdm import tqdm

target_dir = os.path.abspath("../datasets")
os.makedirs(target_dir, exist_ok=True)

url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
archive_path = os.path.join(target_dir, "LJSpeech-1.1.tar.bz2")

# ---- Download with progress ----
with tqdm(unit='B', unit_scale=True, desc="Downloading") as pbar:
    def hook(block_num, block_size, total_size):
        pbar.total = total_size
        pbar.update(block_num * block_size - pbar.n)

    urllib.request.urlretrieve(url, archive_path, reporthook=hook)

# ---- Extract with progress ----
with tarfile.open(archive_path, "r:bz2") as tar:
    members = tar.getmembers()
    with tqdm(total=len(members), desc="Extracting") as pbar:
        for member in members:
            tar.extract(member, path=target_dir)
            pbar.update(1)

# ---- Cleanup: delete archive ----
if os.path.exists(archive_path):
    os.remove(archive_path)
    print("Deleted archive:", archive_path)
else:
    print("Archive not found, nothing to delete.")