"""
DAN Compressor - Benchmark Script
==================================
Compares all compression methods against PNG and WebP.
"""
import os
import sys
from PIL import Image
import time


def get_file_size(path):
    return os.path.getsize(path)


def benchmark_image(image_path):
    """Benchmark all compression methods on a single image."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    orig_size = get_file_size(image_path)
    print(f"Original PNG: {orig_size} bytes")
    
    # WebP (lossless)
    webp_path = '/tmp/bench_webp.webp'
    img = Image.open(image_path).convert('RGB')
    img.save(webp_path, 'WEBP', lossless=True)
    webp_size = get_file_size(webp_path)
    print(f"WebP (lossless): {webp_size} bytes ({orig_size/webp_size:.1f}x)")
    
    # RLE v1 (original)
    try:
        from rle_compressor import compress as rle_v1_compress, decompress as rle_v1_decompress
        rle_v1_path = '/tmp/bench_rle_v1.dan'
        rle_v1_compress(image_path, rle_v1_path)
        rle_v1_size = get_file_size(rle_v1_path)
        print(f"RLE v1 (original): {rle_v1_size} bytes ({orig_size/rle_v1_size:.1f}x)")
        
        # Verify decompression
        decompressed_path = '/tmp/bench_rle_v1_decompressed.png'
        rle_v1_decompress(rle_v1_path, decompressed_path)
        assert get_file_size(decompressed_path) == orig_size, "Decompression failed for RLE v1"
    except Exception as e:
        print(f"RLE v1: FAILED - {e}")
        rle_v1_size = None
    
    # RLE + LZMA
    try:
        from rle_compressor_lzma import compress as rle_lzma_compress, decompress as rle_lzma_decompress
        rle_lzma_path = '/tmp/bench_rle_lzma.dan'
        rle_lzma_compress(image_path, rle_lzma_path)
        rle_lzma_size = get_file_size(rle_lzma_path)
        print(f"RLE + LZMA: {rle_lzma_size} bytes ({orig_size/rle_lzma_size:.1f}x)")
        
        # Verify decompression
        decompressed_path = '/tmp/bench_rle_lzma_decompressed.png'
        rle_lzma_decompress(rle_lzma_path, decompressed_path)
        assert get_file_size(decompressed_path) == orig_size, "Decompression failed for RLE + LZMA"
    except Exception as e:
        print(f"RLE + LZMA: FAILED - {e}")
        rle_lzma_size = None
    
    # Dictionary compressor
    try:
        from dictionary_compressor import compress as dict_compress
        dict_path = '/tmp/bench_dict.dan'
        dict_compress(image_path, dict_path)
        dict_size = get_file_size(dict_path)
        print(f"Dictionary: {dict_size} bytes ({orig_size/dict_size:.1f}x)")
    except Exception as e:
        print(f"Dictionary: FAILED - {e}")
        dict_size = None
    
    # Print summary
    print(f"\nSummary:")
    if rle_lzma_size and webp_size:
        if rle_lzma_size < webp_size:
            print(f"  ✓ RLE + LZMA BEATS WebP by {webp_size - rle_lzma_size} bytes!")
        else:
            print(f"  ✗ RLE + LZMA loses to WebP by {rle_lzma_size - webp_size} bytes")
    
    # Return results
    return {
        'png': orig_size,
        'webp': webp_size,
        'rle_v1': rle_v1_size,
        'rle_lzma': rle_lzma_size,
        'dictionary': dict_size,
    }


def main():
    # Find all test images
    test_images = []
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.lower().endswith('.png') and not f.startswith('recon_'):
                test_images.append(os.path.join(root, f))
    
    if not test_images:
        print("No PNG images found for benchmarking.")
        print("Please add test images or run: python make_dataset.py")
        return
    
    print(f"Found {len(test_images)} test images")
    
    all_results = {}
    for img_path in test_images:
        try:
            results = benchmark_image(img_path)
            all_results[img_path] = results
        except Exception as e:
            print(f"Error benchmarking {img_path}: {e}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for img_path, results in all_results.items():
        print(f"\n{os.path.basename(img_path)}:")
        for method, size in results.items():
            if size is not None:
                ratio = results['png'] / size
                print(f"  {method:12s}: {size:6d} bytes ({ratio:.1f}x)")


if __name__ == '__main__':
    main()
