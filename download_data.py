"""
Facade-Renderer: Dataset Download Script
Downloads the Pix2Pix CMP Facades dataset into D:\\Datasets using kagglehub
as specified in the project requirements.
"""

import os
import sys
import shutil
import tarfile
import argparse
import urllib.request

# Default download cache directory specified by user
DEFAULT_DATA_DIR = r"D:\Datasets"


def download_with_kagglehub(cache_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Downloads the pix2pix-dataset from Kaggle using kagglehub,
    caching and storing it in the specified directory.
    """
    print("=" * 60)
    print("Downloading dataset via kagglehub...")
    print(f"Target directory: {cache_dir}")
    print("=" * 60)

    # Set kagglehub cache directory to D:\Datasets
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = cache_dir

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Please install it using: pip install kagglehub")
        sys.exit(1)

    # Download latest version of the dataset
    path = kagglehub.dataset_download("vikramtiwari/pix2pix-dataset")
    print("Path to dataset files:", path)

    # Check for facades subfolder within downloaded dataset
    facades_path = os.path.join(path, "facades")
    if os.path.exists(facades_path):
        print(f"Located CMP Facades dataset at: {facades_path}")
        return facades_path
    
    return path


def download_facades_direct(target_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Fast direct download fallback for CMP Facades (~30MB) from the official
    UC Berkeley Pix2Pix repository directly into target_dir/facades.
    Useful when bandwidth or time is constrained.
    """
    facades_dir = os.path.join(target_dir, "facades")
    if os.path.exists(os.path.join(facades_dir, "train")) and os.path.exists(os.path.join(facades_dir, "val")):
        print(f"Facades dataset already exists at: {facades_dir}")
        return facades_dir

    os.makedirs(target_dir, exist_ok=True)
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
    tar_path = os.path.join(target_dir, "facades.tar.gz")

    print("=" * 60)
    print(f"Downloading CMP Facades dataset (~30 MB) from {url}...")
    print(f"Saving to {tar_path}...")
    print("=" * 60)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            print(f"\rDownloading: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="")
        else:
            print(f"\rDownloading: {downloaded / (1024*1024):.1f} MB", end="")

    urllib.request.urlretrieve(url, tar_path, reporthook=_progress)
    print("\nDownload complete. Extracting archive...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    if os.path.exists(tar_path):
        os.remove(tar_path)

    print(f"Extracted CMP Facades dataset to: {facades_dir}")
    return facades_dir


def main():
    parser = argparse.ArgumentParser(description="Download CMP Facades dataset for Facade-Renderer")
    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Target dataset directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Directly download CMP Facades archive (~30 MB) instead of entire 2.4 GB multi-dataset Kaggle archive."
    )

    args = parser.parse_args()

    if args.direct:
        facades_path = download_facades_direct(args.dest)
    else:
        facades_path = download_with_kagglehub(args.dest)

    print("\nDataset ready!")
    print(f"CMP Facades path: {facades_path}")


if __name__ == "__main__":
    main()
