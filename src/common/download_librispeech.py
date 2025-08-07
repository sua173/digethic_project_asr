#!/usr/bin/env python3
"""
Download and extract LibriSpeech datasets (dev-clean, test-clean, train-clean-100).

This script downloads the LibriSpeech datasets from OpenSLR and extracts them
to the data/ directory in the project root.

Usage:
    python src/common/download_librispeech.py           # Download all datasets
    python src/common/download_librispeech.py --dev     # Download only dev-clean
    python src/common/download_librispeech.py --test    # Download only test-clean
    python src/common/download_librispeech.py --train   # Download only train-clean-100
"""

import os
import sys
import tarfile
import urllib.request
import argparse
from pathlib import Path
from typing import Optional, List
import hashlib


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def download_file(url: str, dest_path: Path, expected_md5: Optional[str] = None) -> bool:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        expected_md5: Expected MD5 hash for verification
    
    Returns:
        True if download successful, False otherwise
    """
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        if expected_md5:
            print("Verifying MD5 hash...")
            if verify_md5(dest_path, expected_md5):
                print("MD5 verification passed")
                return True
            else:
                print("MD5 verification failed, re-downloading...")
                dest_path.unlink()
        else:
            return True
    
    print(f"Downloading: {url}")
    print(f"Destination: {dest_path}")
    
    try:
        # Create parent directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f'\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
        print()  # New line after progress bar
        
        # Verify MD5 if provided
        if expected_md5:
            print("Verifying MD5 hash...")
            if not verify_md5(dest_path, expected_md5):
                print(f"MD5 verification failed for {dest_path}")
                dest_path.unlink()
                return False
            print("MD5 verification passed")
        
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """
    Verify MD5 hash of a file.
    
    Args:
        file_path: Path to file
        expected_md5: Expected MD5 hash
    
    Returns:
        True if MD5 matches, False otherwise
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    return actual_md5 == expected_md5


def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """
    Extract tar.gz file to destination.
    
    Args:
        tar_path: Path to tar.gz file
        extract_to: Directory to extract to
    
    Returns:
        True if extraction successful, False otherwise
    """
    print(f"Extracting: {tar_path}")
    print(f"Destination: {extract_to}")
    
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get total members for progress
            members = tar.getmembers()
            total = len(members)
            
            # Extract with progress
            for i, member in enumerate(members):
                tar.extract(member, extract_to)
                if i % 100 == 0:  # Update progress every 100 files
                    percent = (i / total) * 100
                    sys.stdout.write(f'\rExtracting: {percent:.1f}% ({i}/{total} files)')
                    sys.stdout.flush()
            
            sys.stdout.write(f'\rExtracting: 100.0% ({total}/{total} files)\n')
            sys.stdout.flush()
        
        return True
        
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False


def download_librispeech_dataset(dataset_name: str, data_dir: Path, keep_archive: bool = False) -> bool:
    """
    Download and extract a LibriSpeech dataset.
    
    Args:
        dataset_name: Name of dataset (dev-clean, test-clean, train-clean-100)
        data_dir: Directory to save data
        keep_archive: Whether to keep the tar.gz file after extraction
    
    Returns:
        True if successful, False otherwise
    """
    # LibriSpeech dataset information
    datasets = {
        'dev-clean': {
            'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
            'md5': '42e2234ba48799c1f50f24a7926300a1',
            'size_mb': 337  # Approximate size in MB
        },
        'test-clean': {
            'url': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
            'md5': '32fa31d27d2e1cad72775fee3f4849a9',
            'size_mb': 346
        },
        'train-clean-100': {
            'url': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
            'md5': '2a93770f6d5c6c964bc36631d331a522',
            'size_mb': 6387
        }
    }
    
    if dataset_name not in datasets:
        print(f"Unknown dataset: {dataset_name}")
        return False
    
    dataset_info = datasets[dataset_name]
    
    # Check if already extracted
    extracted_dir = data_dir / "LibriSpeech" / dataset_name
    if extracted_dir.exists():
        print(f"Dataset already exists: {extracted_dir}")
        return True
    
    # Download archive
    archive_name = f"{dataset_name}.tar.gz"
    archive_path = data_dir / archive_name
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Size: ~{dataset_info['size_mb']} MB")
    print(f"{'='*60}")
    
    if not download_file(dataset_info['url'], archive_path, dataset_info['md5']):
        return False
    
    # Extract archive
    if not extract_tar(archive_path, data_dir):
        return False
    
    # Move to correct location (remove LibriSpeech intermediate directory)
    librispeech_dir = data_dir / "LibriSpeech" / dataset_name
    target_dir = data_dir / dataset_name
    
    if librispeech_dir.exists() and not target_dir.exists():
        print(f"Moving {librispeech_dir} to {target_dir}")
        librispeech_dir.rename(target_dir)
        
        # Remove empty LibriSpeech directory if it exists
        librispeech_parent = data_dir / "LibriSpeech"
        if librispeech_parent.exists() and not any(librispeech_parent.iterdir()):
            librispeech_parent.rmdir()
    
    # Remove archive if not keeping
    if not keep_archive and archive_path.exists():
        print(f"Removing archive: {archive_path}")
        archive_path.unlink()
    
    print(f"Successfully downloaded and extracted {dataset_name}")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and extract LibriSpeech datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--dev', action='store_true',
                       help='Download dev-clean dataset')
    parser.add_argument('--test', action='store_true',
                       help='Download test-clean dataset')
    parser.add_argument('--train', action='store_true',
                       help='Download train-clean-100 dataset')
    parser.add_argument('--keep-archive', action='store_true',
                       help='Keep tar.gz files after extraction')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory to save data (default: project_root/data)')
    
    args = parser.parse_args()
    
    # Determine which datasets to download
    datasets_to_download = []
    if args.dev:
        datasets_to_download.append('dev-clean')
    if args.test:
        datasets_to_download.append('test-clean')
    if args.train:
        datasets_to_download.append('train-clean-100')
    
    # If no specific dataset specified, download all
    if not datasets_to_download:
        datasets_to_download = ['dev-clean', 'test-clean', 'train-clean-100']
    
    # Set data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = get_project_root() / "data"
    
    print(f"Data directory: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    print(f"\nDownloading {len(datasets_to_download)} dataset(s): {', '.join(datasets_to_download)}")
    print(f"Total estimated size: ~{sum([{'dev-clean': 337, 'test-clean': 346, 'train-clean-100': 6387}[d] for d in datasets_to_download])} MB")
    
    success_count = 0
    failed_datasets = []
    
    for dataset in datasets_to_download:
        if download_librispeech_dataset(dataset, data_dir, args.keep_archive):
            success_count += 1
        else:
            failed_datasets.append(dataset)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{len(datasets_to_download)}")
    
    if failed_datasets:
        print(f"Failed: {', '.join(failed_datasets)}")
        print("\nTo retry failed downloads, run:")
        for dataset in failed_datasets:
            flag = dataset.replace('-clean', '').replace('train-clean-100', 'train')
            print(f"  python src/common/download_librispeech.py --{flag}")
        return 1
    else:
        print("\nAll datasets downloaded successfully!")
        print(f"Data location: {data_dir}")
        print("\nYou can now proceed with dataset creation:")
        print("  python src/ctc/tools/create_dataset.py --mode all")
        return 0


if __name__ == "__main__":
    sys.exit(main())