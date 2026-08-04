#!/usr/bin/env python3
"""
DAN Codec - Command Line Interface
====================================
A robust CLI tool for the DAN image compression format.

Usage:
    dan convert input.png output.dan -q 80 -m auto
    dan decode input.dan output.png
    dan compare original.png compressed.dan --webp reference.webp
    dan benchmark_suite ./test_images/ --output results.csv
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List
import csv
import time
from datetime import datetime

try:
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

from dan_codec import DANCodec, CompressionMode


def cmd_convert(args):
    """Convert ANY image to .dan format."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.dan')
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Validate quality
    quality = max(1, min(100, args.quality))
    
    try:
        codec = DANCodec(quality=quality, mode=args.mode)
        
        print(f"Converting: {input_path} -> {output_path}")
        print(f"Quality: {quality}, Mode: {args.mode}")
        
        start_time = time.time()
        size = codec.save(str(input_path), str(output_path))
        elapsed = time.time() - start_time
        
        orig_size = input_path.stat().st_size
        ratio = orig_size / size if size > 0 else float('inf')
        
        print(f"\nResults:")
        print(f"  Original size: {orig_size:,} bytes")
        print(f"  DAN size:      {size:,} bytes")
        print(f"  Compression:   {ratio:.2f}x ({(1 - size/orig_size)*100:.1f}% reduction)")
        print(f"  Time:          {elapsed:.2f}s")
        
        return 0
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


def cmd_decode(args):
    """Convert .dan back to PNG/JPG."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.png')
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    try:
        codec = DANCodec()
        
        print(f"Decoding: {input_path} -> {output_path}")
        
        start_time = time.time()
        decoded = codec.load(str(input_path))
        elapsed = time.time() - start_time
        
        # Save as PNG or JPG based on extension
        img = Image.fromarray(decoded)
        img.save(str(output_path))
        
        print(f"\nSuccessfully decoded to {output_path}")
        print(f"Output size: {decoded.shape[1]}x{decoded.shape[0]}")
        print(f"Time: {elapsed:.2f}s")
        
        return 0
        
    except Exception as e:
        print(f"Error during decoding: {e}")
        return 1


def cmd_compare(args):
    """Side-by-side benchmark against WebP."""
    original_path = Path(args.original)
    dan_path = Path(args.dan)
    webp_path = Path(args.webp) if args.webp else None
    
    if not original_path.exists():
        print(f"Error: Original file not found: {original_path}")
        return 1
    
    if not dan_path.exists():
        print(f"Error: DAN file not found: {dan_path}")
        return 1
    
    try:
        codec = DANCodec()
        
        print("=" * 60)
        print("DAN vs WebP Comparison Report")
        print("=" * 60)
        print(f"\nOriginal:  {original_path}")
        print(f"DAN:       {dan_path}")
        if webp_path and webp_path.exists():
            print(f"WebP:      {webp_path}")
        print()
        
        # Load images
        original = np.array(Image.open(str(original_path)).convert('RGB'))
        reconstructed = codec.load(str(dan_path))
        
        # Ensure same dimensions
        h = min(original.shape[0], reconstructed.shape[0])
        w = min(original.shape[1], reconstructed.shape[1])
        original = original[:h, :w]
        reconstructed = reconstructed[:h, :w]
        
        # Calculate metrics
        orig_size = original_path.stat().st_size
        dan_size = dan_path.stat().st_size
        
        psnr = DANCodec.calculate_psnr(original, reconstructed)
        ssim = DANCodec.calculate_ssim(original, reconstructed)
        
        print("File Sizes:")
        print(f"  Original: {orig_size:,} bytes")
        print(f"  DAN:      {dan_size:,} bytes")
        print(f"  Ratio:    {orig_size/dan_size:.2f}x")
        print()
        
        print("Quality Metrics:")
        print(f"  PSNR:     {psnr:.2f} dB")
        print(f"  SSIM:     {ssim:.4f}")
        print()
        
        if webp_path and webp_path.exists():
            webp_size = webp_path.stat().st_size
            webp_img = np.array(Image.open(str(webp_path)).convert('RGB'))
            webp_img = webp_img[:h, :w]
            
            webp_psnr = DANCodec.calculate_psnr(original, webp_img)
            webp_ssim = DANCodec.calculate_ssim(original, webp_img)
            
            print("WebP Comparison:")
            print(f"  WebP Size:    {webp_size:,} bytes")
            print(f"  WebP PSNR:    {webp_psnr:.2f} dB")
            print(f"  WebP SSIM:    {webp_ssim:.4f}")
            print()
            
            size_diff = (webp_size - dan_size) / webp_size * 100
            psnr_diff = psnr - webp_psnr
            
            print("Advantage:")
            print(f"  File Size:  {'DAN' if size_diff > 0 else 'WebP'} wins by {abs(size_diff):.1f}%")
            print(f"  PSNR:       {'DAN' if psnr_diff > 0 else 'WebP'} wins by {abs(psnr_diff):.2f} dB")
        
        # Generate difference map if requested
        if args.diff_map:
            diff_map = np.abs(original.astype(np.int16) - reconstructed.astype(np.int16))
            diff_map = (diff_map * 10).clip(0, 255).astype(np.uint8)
            diff_img = Image.fromarray(diff_map)
            diff_path = Path(args.diff_map)
            diff_img.save(str(diff_path))
            print(f"\nDifference map saved to: {diff_path}")
        
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_benchmark_suite(args):
    """Process a directory of images and generate CSV report."""
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return 1
    
    # Find all supported image files
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in supported_extensions]
    
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return 1
    
    print(f"Found {len(image_files)} images to benchmark")
    print(f"Testing qualities: {args.qualities}")
    print()
    
    results = []
    qualities = [int(q) for q in args.qualities.split(',')]
    
    try:
        for img_path in tqdm(image_files, desc="Processing images"):
            for quality in qualities:
                try:
                    codec = DANCodec(quality=quality, mode=args.mode)
                    
                    # Encode to DAN
                    dan_path = img_path.with_suffix('.dan')
                    start_time = time.time()
                    dan_size = codec.save(str(img_path), str(dan_path))
                    encode_time = time.time() - start_time
                    
                    # Decode and verify
                    start_time = time.time()
                    decoded = codec.load(str(dan_path))
                    decode_time = time.time() - start_time
                    
                    # Load original
                    original = np.array(Image.open(str(img_path)).convert('RGB'))
                    
                    # Match dimensions
                    h = min(original.shape[0], decoded.shape[0])
                    w = min(original.shape[1], decoded.shape[1])
                    original = original[:h, :w]
                    decoded = decoded[:h, :w]
                    
                    # Calculate metrics
                    psnr = DANCodec.calculate_psnr(original, decoded)
                    ssim = DANCodec.calculate_ssim(original, decoded)
                    
                    orig_size = img_path.stat().st_size
                    
                    # Create WebP for comparison
                    webp_path = img_path.with_suffix('.webp')
                    Image.fromarray(original).save(str(webp_path), 'WEBP', quality=quality)
                    webp_size = webp_path.stat().st_size
                    
                    results.append({
                        'image': img_path.name,
                        'category': _categorize_image(img_path.name),
                        'quality': quality,
                        'original_size': orig_size,
                        'dan_size': dan_size,
                        'webp_size': webp_size,
                        'compression_ratio': orig_size / dan_size,
                        'psnr': psnr,
                        'ssim': ssim,
                        'encode_time': encode_time,
                        'decode_time': decode_time,
                        'size_vs_webp': (webp_size - dan_size) / webp_size * 100,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Clean up temp files
                    if dan_path.exists() and not args.keep_temp:
                        dan_path.unlink()
                    if webp_path.exists() and not args.keep_temp:
                        webp_path.unlink()
                        
                except Exception as e:
                    print(f"Warning: Failed to process {img_path.name} at Q{quality}: {e}")
                    continue
        
        # Write CSV report
        if results:
            fieldnames = list(results[0].keys())
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\nBenchmark complete!")
            print(f"Results saved to: {output_csv}")
            print(f"Total images processed: {len(results)}")
            
            # Print summary statistics
            _print_summary(results)
        else:
            print("No results generated")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _categorize_image(filename: str) -> str:
    """Categorize image based on filename hints."""
    name_lower = filename.lower()
    if any(kw in name_lower for kw in ['anime', 'manga', 'cartoon', 'line']):
        return 'Anime/Line Art'
    elif any(kw in name_lower for kw in ['car', 'vehicle', 'auto']):
        return 'Cars'
    elif any(kw in name_lower for kw in ['landscape', 'nature', 'scenery']):
        return 'Landscape'
    elif any(kw in name_lower for kw in ['portrait', 'face', 'selfie', 'person']):
        return 'Portrait'
    elif any(kw in name_lower for kw in ['animal', 'pet', 'fur', 'cat', 'dog']):
        return 'Animals'
    elif any(kw in name_lower for kw in ['text', 'doc', 'document']):
        return 'Text/Document'
    else:
        return 'General'


def _print_summary(results: List[dict]):
    """Print summary statistics from benchmark results."""
    if not results:
        return
    
    # Group by category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    print("\n" + "=" * 60)
    print("Summary Statistics by Category")
    print("=" * 60)
    
    for cat, cat_results in sorted(categories.items()):
        avg_psnr = np.mean([r['psnr'] for r in cat_results])
        avg_ssim = np.mean([r['ssim'] for r in cat_results])
        avg_size_improvement = np.mean([r['size_vs_webp'] for r in cat_results])
        avg_compression = np.mean([r['compression_ratio'] for r in cat_results])
        
        print(f"\n{cat}:")
        print(f"  Avg PSNR:           {avg_psnr:.2f} dB")
        print(f"  Avg SSIM:           {avg_ssim:.4f}")
        print(f"  Avg Compression:    {avg_compression:.2f}x")
        print(f"  Size vs WebP:       {avg_size_improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        prog='dan',
        description='DAN Codec - Data-Adaptive Network Image Compression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s convert input.png output.dan -q 80 -m auto
  %(prog)s decode input.dan output.png
  %(prog)s compare original.png compressed.dan --webp reference.webp
  %(prog)s benchmark_suite ./images/ --output results.csv -q "60,80,90"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert image to .dan format')
    convert_parser.add_argument('input', type=str, help='Input image path')
    convert_parser.add_argument('-o', '--output', type=str, help='Output .dan file path')
    convert_parser.add_argument('-q', '--quality', type=int, default=80, 
                                help='Quality level 1-100 (default: 80)')
    convert_parser.add_argument('-m', '--mode', type=str, default='auto',
                                choices=['auto', 'anime', 'photo', 'neural'],
                                help='Compression mode (default: auto)')
    convert_parser.set_defaults(func=cmd_convert)
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode .dan to image')
    decode_parser.add_argument('input', type=str, help='Input .dan file path')
    decode_parser.add_argument('-o', '--output', type=str, help='Output image path')
    decode_parser.set_defaults(func=cmd_decode)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare DAN vs WebP')
    compare_parser.add_argument('original', type=str, help='Original image path')
    compare_parser.add_argument('dan', type=str, help='DAN compressed file path')
    compare_parser.add_argument('--webp', type=str, help='WebP file for comparison')
    compare_parser.add_argument('--diff-map', type=str, help='Save difference map to path')
    compare_parser.set_defaults(func=cmd_compare)
    
    # Benchmark suite command
    bench_parser = subparsers.add_parser('benchmark_suite', help='Run comprehensive benchmarks')
    bench_parser.add_argument('input_dir', type=str, help='Directory containing test images')
    bench_parser.add_argument('-o', '--output', type=str, default='benchmark_results.csv',
                              help='Output CSV file path')
    bench_parser.add_argument('-q', '--qualities', type=str, default='60,80,90',
                              help='Comma-separated quality levels to test')
    bench_parser.add_argument('-m', '--mode', type=str, default='auto',
                              choices=['auto', 'anime', 'photo', 'neural'],
                              help='Compression mode')
    bench_parser.add_argument('--keep-temp', action='store_true',
                              help='Keep temporary .dan and .webp files')
    bench_parser.set_defaults(func=cmd_benchmark_suite)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
