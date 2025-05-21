# Real-World Network Datasets

This directory contains real-world network datasets used in the experiments.

## Datasets

1. **email-Eu-core-temporal** (5.3MB)
   - Email communication network from a European research institution
   - Included directly in the repository
   - Format: Source node, target node, timestamp
   - Source: [SNAP](https://snap.stanford.edu/data/email-Eu-core-temporal.html)

2. **roadNet-CA** (84MB)
   - Road network of California
   - Downloaded by the `download_datasets.py` script
   - Format: Source node, target node
   - Source: [SNAP](https://snap.stanford.edu/data/roadNet-CA.html)

3. **wiki-Talk** (64MB)
   - Wikipedia talk (communication) network
   - Downloaded by the `download_datasets.py` script
   - Format: Source node, target node, timestamp
   - Source: [SNAP](https://snap.stanford.edu/data/wiki-Talk.html)

4. **soc-redditHyperlinks-body** (305MB)
   - Reddit hyperlinks network
   - Downloaded by the `download_datasets.py` script
   - Format: Source subreddit, target subreddit, timestamp, properties
   - Source: [SNAP](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

## Downloading the Datasets

The smaller dataset (email-Eu-core-temporal) is included directly in the repository, while the larger datasets need to be downloaded using the `download_datasets.py` script:

```bash
python download_datasets.py
```

This script will download the datasets from their original sources and extract them to this directory.

## Dataset Statistics

| Dataset | Nodes | Edges | Size |
|---------|-------|-------|------|
| email-Eu-core-temporal | 986 | 332,334 | 5.3MB |
| roadNet-CA | 1,965,206 | 5,533,214 | 84MB |
| wiki-Talk | 2,394,385 | 5,021,410 | 64MB |
| soc-redditHyperlinks-body | 35,776 | 286,561 | 305MB |

## Citations

If you use these datasets in your research, please cite the original sources:

```
@inproceedings{snapnets,
  author       = {Jure Leskovec and Andrej Krevl},
  title        = {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
  howpublished = {\url{http://snap.stanford.edu/data}},
  month        = jun,
  year         = 2014
}
```
