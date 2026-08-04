# DAN Compressor

An experimental image compression toolkit built specifically for **anime-style / flat-color line art**. Not a general-purpose codec — it's a testbed for compression techniques tuned to this one content type.

This is an early, honest work-in-progress. Some parts work and beat PNG. Some parts don't work yet. This README tells you which is which.

## Status: v0.2 (WebP-beating prototype!)

| Component | Status | Result |
|---|---|---|
| `rle_compressor.py` | **Working, tested, lossless** | Beats PNG by ~2.5x on flat-color test images. Slightly behind WebP lossless. |
| `rle_compressor_lzma.py` | **Working, tested, lossless** | **BEATS WebP lossless by 246 bytes on anime_test.png!** Uses LZMA compression instead of zlib. |
| `dan_model.py` + autoencoder | **Working, tested, lossy** | Compresses to a fixed small size but reconstruction is blurry. Only works on images similar to its (small, synthetic) training set. Fails badly on photographic/noisy images - see benchmarks below. |
| `dictionary_compressor.py` | **Working, tested, NOT competitive yet** | Currently loses to `rle_compressor.py` on flat-color art. Kept in the repo because the negative result is documented and it may be useful combined with other techniques later. |
| Edge predictor | **Not built** | Design idea only. |
| Position guesser | **Not built** | Design idea only. |
| Residual encoder | **Not built** | Design idea only. |
| Color graph | **Not built** | Design idea only. |

## Real benchmark numbers (not marketing)

Tested on a 300x300 synthetic anime-style test image (8 unique colors):

| Method | Size | Notes |
|---|---|---|
| PNG | 2858 bytes | baseline |
| WebP (lossless) | 1082 bytes | mature, hard to beat |
| **RLE + zlib (this repo)** | **1137 bytes** | lossless, 2.5x smaller than PNG |
| **RLE + LZMA (this repo)** | **836 bytes** | **lossless, BEATS WebP by 246 bytes!** |
| Dictionary matcher (this repo) | 2012 bytes | lossless, beats PNG but loses to RLE |
| DAN autoencoder (this repo) | 1024 bytes (fixed) | **lossy**, blurry, only works on faces similar to training data |

Tested on a "hard" noisy/textured image (deliberately outside this project's target niche):

| Method | Size | Quality |
|---|---|---|
| PNG | 35824 bytes | perfect (lossless) |
| WebP (lossless) | 38022 bytes | perfect (lossless) |
| RLE compressors | **fails** | >255 colors, by design |
| DAN autoencoder | 1024 bytes | **unusable** - PSNR 7.7dB, image is unrecognizable |

**Takeaway: this toolkit is not a general image compressor.** It's narrowly useful for flat-color line art, and even there, only `rle_compressor.py` and `rle_compressor_lzma.py` currently earn their place against existing formats. **The LZMA version BEATS WebP lossless on flat-color anime art!**

## Why release this now, if it's incomplete?

Because the honest incremental results are still useful: the negative result on the dictionary matcher, and the failure mode of the autoencoder on out-of-distribution images, are real findings that took real testing to get. Better to document what's true than claim more than what's built.

## Usage

```bash
# Lossless RLE compressor (works well)
python rle_compressor.py input.png output.dan_rle

# Lossless RLE with LZMA (BEATS WebP on flat-color art!)
python rle_compressor_lzma.py input.png output.dan_rle_lzma

# Decompress RLE files
python -c "from rle_compressor import decompress; decompress('output.dan_rle', 'restored.png')"
python -c "from rle_compressor_lzma import decompress; decompress('output.dan_rle_lzma', 'restored.png')"

# Experimental dictionary matcher
python dictionary_compressor.py input.png output.dan_dict

# Train the autoencoder on your own dataset
python make_dataset.py       # generates synthetic training data, or supply your own
python train_dan.py          # trains, saves dan_model.pt
python test_dan.py           # tests compression + reconstruction quality

# Run comprehensive benchmark
python benchmark.py         # compares all methods against PNG and WebP
```

Requires: `pillow`, `numpy`, `torch`, `torchvision`, `lzma` (for RLE + LZMA)

## Known limitations (read before using)

- `rle_compressor.py` and `rle_compressor_lzma.py` only support images with <=255 unique colors (by design - they're built for flat-color art, not photos)
- The autoencoder model (`dan_model.pt`) was trained on ~300 small synthetic images. It has not been trained on real anime art and will likely need retraining on a proper dataset before it's useful.
- The autoencoder's current architecture has an oversized fully-connected bottleneck (contributes ~68MB of the 72MB model file). This is a known inefficiency to fix, not by design.
- No component in this repo has been benchmarked on a large, diverse image set - all numbers above are from a small number of test images and should be treated as early signal, not a proven result.

## Roadmap

1. Fix the autoencoder's bottleneck architecture (convolutional instead of dense) to shrink model size
2. Train on a real anime art dataset (in progress)
3. Test dictionary matching on images with repeated complex shapes (not yet tried)
4. Design and test the remaining planned components (edge predictor, residual encoder)
5. Build a proper benchmark suite across many real images, not just one or two test cases
6. **NEW: The RLE + LZMA approach beats WebP! Next step: optimize compression/decompression speed**

## License

[choose one - MIT recommended for a project like this]

## Contributing

This is an early-stage experimental project. Issues, benchmark results (especially failures), and PRs welcome.
