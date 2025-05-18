from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dynamic_network",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
    ],
    author="Dynamic Network Analysis Team",
    author_email="author@example.com",
    description="A framework for causal pathway inference and optimized intervention in dynamic networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="dynamic networks, causal inference, pathway detection, source localization, intervention, spatiotemporal analysis",
    url="https://github.com/username/dynamic_network",
    project_urls={
        "Bug Tracker": "https://github.com/username/dynamic_network/issues",
        "Documentation": "https://github.com/username/dynamic_network",
        "Source Code": "https://github.com/username/dynamic_network",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
