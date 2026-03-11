# Flickr30k Dataset Setup

This project trains on [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/), a dataset of ~31,000 images with 5 human-written captions each (~155,000 image-caption pairs total).

**Citation:**
> Young et al. (2014). "From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions."
> *Transactions of the Association for Computational Linguistics*, 2, 67–78.

---

## Download Instructions

Flickr30k requires a simple academic license request. It's free for research use.

### Step 1 — Request Access

Go to: https://shannon.cs.illinois.edu/DenotationGraph/

Fill out the short request form. You'll receive a download link by email (usually within a few hours).

### Step 2 — Download

Once you have the link, download:
- `flickr30k_images.tar.gz` — the image archive (~4.4GB)
- `results.csv` — the captions file

### Step 3 — Extract and Place

```bash
# From the root of this repo:
mkdir -p data/flickr30k_images

# Extract images
tar -xzf flickr30k_images.tar.gz -C data/flickr30k_images/

# Move captions file
cp results.csv data/results.csv
```

### Step 4 — Verify

Your `data/` directory should look like this:

```
data/
├── flickr30k_images/
│   ├── 1000092795.jpg
│   ├── 10002456.jpg
│   ├── ...  (~31,000 .jpg files)
└── results.csv
```

You can verify the CSV format with:

```bash
head -3 data/results.csv
```

Expected output:
```
image_name | comment_number | comment
1000092795.jpg| 0| Two young guys with shaggy hair look at their hands...
1000092795.jpg| 1| Two young white males are outside near many bushes...
```

---

## Dataset Stats

| Stat | Value |
|---|---|
| Images | ~31,000 |
| Captions per image | 5 |
| Total pairs | ~155,000 |
| Avg caption length | ~12 words |
| Train split (this project) | 85% (~26,350 images) |
| Val split | 7.5% (~2,325 images) |
| Test split | 7.5% (~2,325 images) |

---

## Alternative: Flickr8k

If Flickr30k access is delayed, you can prototype on **Flickr8k** (~8,000 images), which is available directly on Kaggle:

```
https://www.kaggle.com/datasets/adityajn105/flickr8k
```

Update the `--data_dir` and adjust the CSV column names in `src/dataset.py` accordingly (Flickr8k uses slightly different formatting).

---

## License

Flickr30k images are sourced from Flickr and are licensed under Creative Commons. The dataset annotations are released for non-commercial research use. See the dataset homepage for full terms.
