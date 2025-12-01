#!/usr/bin/env python3
"""Download MNIST dataset using a reliable method"""

import urllib.request
import gzip
import shutil
import os

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def decompress_gz(gz_file, out_file):
    """Decompress a .gz file"""
    print(f"Decompressing {gz_file}...")
    try:
        with gzip.open(gz_file, 'rb') as f_in:
            with open(out_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Decompressed to {out_file}")
        os.remove(gz_file)  # Remove .gz file after decompression
        return True
    except Exception as e:
        print(f"Error decompressing {gz_file}: {e}")
        return False

def main():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    files = [
        ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte")
    ]
    
    for gz_name, out_name in files:
        # Skip if already existswha
        if os.path.exists(out_name):
            print(f"{out_name} already exists, skipping...")
            continue
        
        # Download
        if not download_file(base_url + gz_name, gz_name):
            print(f"Failed to download {gz_name}")
            continue
        
        # Decompress
        if not decompress_gz(gz_name, out_name):
            print(f"Failed to decompress {gz_name}")
            continue
    
    print("\nMNIST dataset download complete!")

if __name__ == "__main__":
    main()

