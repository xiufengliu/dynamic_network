"""
Download the datasets used in the experiments.

This script downloads the datasets used in the experiments from their original sources.
"""

import os
import sys
import logging
import urllib.request
import shutil
import tarfile
import zipfile
import gzip

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_datasets.log')
    ]
)
logger = logging.getLogger(__name__)

# Dataset URLs
ROADNET_URL = "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
WIKI_TALK_URL = "https://snap.stanford.edu/data/wiki-Talk.txt.gz"
REDDIT_URL = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv.gz"

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

def download_file(url, path):
    """Download a file from a URL to a local path."""
    logger.info(f"Downloading {url} to {path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Download the file
    try:
        # Download to a temporary file
        temp_path = path + '.tmp'
        urllib.request.urlretrieve(url, temp_path)
        
        # If the file is gzipped, extract it
        if url.endswith('.gz'):
            logger.info(f"Extracting {temp_path}...")
            with gzip.open(temp_path, 'rb') as f_in:
                with open(path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(temp_path)
        else:
            # Otherwise, just rename the file
            os.rename(temp_path, path)
        
        logger.info(f"Downloaded {url} to {path} successfully.")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return False

def download_roadnet():
    """Download the roadNet-CA dataset."""
    logger.info("Downloading roadNet-CA dataset...")
    return download_file(ROADNET_URL, ROADNET_PATH)

def download_wiki_talk():
    """Download the wiki-Talk dataset."""
    logger.info("Downloading wiki-Talk dataset...")
    return download_file(WIKI_TALK_URL, WIKI_TALK_PATH)

def download_reddit():
    """Download the soc-redditHyperlinks-body dataset."""
    logger.info("Downloading soc-redditHyperlinks-body dataset...")
    return download_file(REDDIT_URL, REDDIT_PATH)

def main():
    """Main function to download datasets."""
    logger.info("Downloading datasets...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/real_world', exist_ok=True)
    
    # Download datasets
    roadnet_ok = download_roadnet()
    wiki_talk_ok = download_wiki_talk()
    reddit_ok = download_reddit()
    
    # Print summary
    logger.info("\n=== Download Summary ===")
    logger.info(f"roadNet-CA: {'OK' if roadnet_ok else 'FAILED'}")
    logger.info(f"wiki-Talk: {'OK' if wiki_talk_ok else 'FAILED'}")
    logger.info(f"soc-redditHyperlinks-body: {'OK' if reddit_ok else 'FAILED'}")
    
    # Return success if all downloads are OK
    return roadnet_ok and wiki_talk_ok and reddit_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
