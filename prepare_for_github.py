"""
Prepare the project for GitHub.

This script prepares the project for GitHub by:
1. Creating necessary directories
2. Checking if all required files exist
3. Cleaning up unnecessary files
"""

import os
import sys
import shutil
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

def create_directories():
    """Create necessary directories."""
    logger.info("Creating necessary directories...")
    
    # Create data directories
    os.makedirs('data/real_world', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)
    
    # Create results directories
    os.makedirs('results/minimal_test', exist_ok=True)
    os.makedirs('results/single_dataset/data', exist_ok=True)
    os.makedirs('results/single_dataset/figures', exist_ok=True)
    
    logger.info("Directories created successfully.")

def check_required_files():
    """Check if all required files exist."""
    logger.info("Checking required files...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        '.gitignore',
        'run_minimal_test.py',
        'run_single_dataset.py',
        'run_all_experiments.py',
        'run_real_world_experiments.py',
        'README_DEBUGGING.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("All required files exist.")
    return True

def clean_up():
    """Clean up unnecessary files."""
    logger.info("Cleaning up unnecessary files...")
    
    # Files to remove
    files_to_remove = [
        '*.log',
        '*.pyc',
        '__pycache__',
        '.DS_Store',
        '.vscode',
        '.idea'
    ]
    
    # Remove log files
    for file in os.listdir('.'):
        if file.endswith('.log'):
            try:
                os.remove(file)
                logger.info(f"Removed {file}")
            except Exception as e:
                logger.error(f"Error removing {file}: {str(e)}")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir in dirs:
            if dir == '__pycache__':
                try:
                    shutil.rmtree(os.path.join(root, dir))
                    logger.info(f"Removed {os.path.join(root, dir)}")
                except Exception as e:
                    logger.error(f"Error removing {os.path.join(root, dir)}: {str(e)}")
    
    logger.info("Cleanup completed successfully.")

def create_gitkeep_files():
    """Create .gitkeep files in empty directories."""
    logger.info("Creating .gitkeep files in empty directories...")
    
    # Directories that should have .gitkeep files
    directories = [
        'data/real_world',
        'data/synthetic',
        'results/minimal_test',
        'results/single_dataset/data',
        'results/single_dataset/figures'
    ]
    
    for directory in directories:
        if os.path.exists(directory) and not os.listdir(directory):
            with open(os.path.join(directory, '.gitkeep'), 'w') as f:
                pass
            logger.info(f"Created .gitkeep in {directory}")
    
    logger.info("Created .gitkeep files successfully.")

def main():
    """Main function to prepare the project for GitHub."""
    logger.info("Preparing project for GitHub...")
    
    # Create necessary directories
    create_directories()
    
    # Check required files
    if not check_required_files():
        logger.error("Some required files are missing. Please create them before pushing to GitHub.")
        return False
    
    # Clean up unnecessary files
    clean_up()
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    logger.info("Project prepared for GitHub successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
