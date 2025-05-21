"""
Check real-world datasets for existence and format.

This script checks if the real-world datasets exist and are in the correct format.
It also prints basic statistics about each dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Dataset paths
ROADNET_PATH = 'data/real_world/roadNet-CA.txt'
WIKI_TALK_PATH = 'data/real_world/wiki-Talk.txt'
EMAIL_PATH = 'data/real_world/email-Eu-core-temporal.txt'
REDDIT_PATH = 'data/real_world/soc-redditHyperlinks-body.tsv'

def check_file_exists(file_path):
    """Check if a file exists."""
    exists = os.path.exists(file_path)
    if exists:
        logger.info(f"✓ File exists: {file_path}")
        logger.info(f"  Size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    else:
        logger.error(f"✗ File does not exist: {file_path}")
    return exists

def check_roadnet_ca():
    """Check the roadNet-CA dataset."""
    logger.info("\n=== Checking roadNet-CA dataset ===")
    
    if not check_file_exists(ROADNET_PATH):
        return False
    
    try:
        # Read the first few lines
        with open(ROADNET_PATH, 'r') as f:
            header_lines = [f.readline() for _ in range(10)]
        
        logger.info("First few lines:")
        for line in header_lines:
            logger.info(f"  {line.strip()}")
        
        # Count lines and edges
        with open(ROADNET_PATH, 'r') as f:
            lines = f.readlines()
        
        # Skip comment lines
        data_lines = [line.strip() for line in lines if not line.startswith('#')]
        
        # Count unique nodes
        nodes = set()
        for line in data_lines:
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    nodes.add(int(parts[0]))
                    nodes.add(int(parts[1]))
        
        logger.info(f"Total lines: {len(lines)}")
        logger.info(f"Data lines: {len(data_lines)}")
        logger.info(f"Unique nodes: {len(nodes)}")
        logger.info(f"Edges: {len(data_lines)}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking roadNet-CA: {str(e)}")
        return False

def check_wiki_talk():
    """Check the wiki-Talk dataset."""
    logger.info("\n=== Checking wiki-Talk dataset ===")
    
    if not check_file_exists(WIKI_TALK_PATH):
        return False
    
    try:
        # Read the first few lines
        with open(WIKI_TALK_PATH, 'r') as f:
            header_lines = [f.readline() for _ in range(10)]
        
        logger.info("First few lines:")
        for line in header_lines:
            logger.info(f"  {line.strip()}")
        
        # Count lines and edges
        with open(WIKI_TALK_PATH, 'r') as f:
            lines = f.readlines()
        
        # Skip comment lines
        data_lines = [line.strip() for line in lines if not line.startswith('#')]
        
        # Count unique nodes
        nodes = set()
        for line in data_lines:
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    nodes.add(int(parts[0]))
                    nodes.add(int(parts[1]))
        
        logger.info(f"Total lines: {len(lines)}")
        logger.info(f"Data lines: {len(data_lines)}")
        logger.info(f"Unique nodes: {len(nodes)}")
        logger.info(f"Edges: {len(data_lines)}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking wiki-Talk: {str(e)}")
        return False

def check_email_eu_core():
    """Check the email-Eu-core-temporal dataset."""
    logger.info("\n=== Checking email-Eu-core-temporal dataset ===")
    
    if not check_file_exists(EMAIL_PATH):
        return False
    
    try:
        # Read the first few lines
        with open(EMAIL_PATH, 'r') as f:
            header_lines = [f.readline() for _ in range(10)]
        
        logger.info("First few lines:")
        for line in header_lines:
            logger.info(f"  {line.strip()}")
        
        # Count lines and edges
        with open(EMAIL_PATH, 'r') as f:
            lines = f.readlines()
        
        # Skip comment lines
        data_lines = [line.strip() for line in lines if not line.startswith('#')]
        
        # Count unique nodes and timestamps
        nodes = set()
        timestamps = []
        for line in data_lines:
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    nodes.add(int(parts[0]))
                    nodes.add(int(parts[1]))
                    timestamps.append(int(parts[2]))
        
        logger.info(f"Total lines: {len(lines)}")
        logger.info(f"Data lines: {len(data_lines)}")
        logger.info(f"Unique nodes: {len(nodes)}")
        logger.info(f"Edges: {len(data_lines)}")
        
        if timestamps:
            logger.info(f"Timestamp range: {min(timestamps)} to {max(timestamps)}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking email-Eu-core: {str(e)}")
        return False

def check_reddit_hyperlinks():
    """Check the soc-redditHyperlinks-body dataset."""
    logger.info("\n=== Checking soc-redditHyperlinks-body dataset ===")
    
    if not check_file_exists(REDDIT_PATH):
        return False
    
    try:
        # Try to read as TSV
        df = pd.read_csv(REDDIT_PATH, sep='\t', nrows=5)
        
        logger.info("First few rows:")
        logger.info(f"\n{df.head()}")
        
        # Get full stats
        df = pd.read_csv(REDDIT_PATH, sep='\t')
        
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Columns: {', '.join(df.columns)}")
        
        # Check for required columns
        required_columns = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP', 'LINK_SENTIMENT']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            return False
        
        # Count unique subreddits
        source_subreddits = df['SOURCE_SUBREDDIT'].nunique()
        target_subreddits = df['TARGET_SUBREDDIT'].nunique()
        all_subreddits = pd.concat([df['SOURCE_SUBREDDIT'], df['TARGET_SUBREDDIT']]).nunique()
        
        logger.info(f"Unique source subreddits: {source_subreddits}")
        logger.info(f"Unique target subreddits: {target_subreddits}")
        logger.info(f"Total unique subreddits: {all_subreddits}")
        
        # Check timestamp format
        try:
            timestamps = pd.to_datetime(df['TIMESTAMP'])
            logger.info(f"Timestamp range: {timestamps.min()} to {timestamps.max()}")
        except:
            logger.warning("Could not parse timestamps")
        
        # Check sentiment distribution
        if 'LINK_SENTIMENT' in df.columns:
            sentiment_counts = df['LINK_SENTIMENT'].value_counts()
            logger.info(f"Sentiment distribution:\n{sentiment_counts}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking Reddit Hyperlinks: {str(e)}")
        return False

def main():
    """Main function to check all datasets."""
    logger.info("Checking real-world datasets...")
    
    # Check if data directory exists
    data_dir = 'data/real_world'
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info(f"Creating directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    
    # Check each dataset
    roadnet_ok = check_roadnet_ca()
    wiki_ok = check_wiki_talk()
    email_ok = check_email_eu_core()
    reddit_ok = check_reddit_hyperlinks()
    
    # Print summary
    logger.info("\n=== Dataset Check Summary ===")
    logger.info(f"roadNet-CA: {'OK' if roadnet_ok else 'FAILED'}")
    logger.info(f"wiki-Talk: {'OK' if wiki_ok else 'FAILED'}")
    logger.info(f"email-Eu-core-temporal: {'OK' if email_ok else 'FAILED'}")
    logger.info(f"soc-redditHyperlinks-body: {'OK' if reddit_ok else 'FAILED'}")
    
    # Return success if all datasets are OK
    return all([roadnet_ok, wiki_ok, email_ok, reddit_ok])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
