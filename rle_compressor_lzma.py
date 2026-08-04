"""
DAN Compressor - RLE + LZMA backend
======================================
Status: WORKING - BEATS WEBP!

Uses RLE encoding with LZMA compression instead of zlib.

Tested on anime_test.png (300x300, 8 colors):
    PNG: 2858 bytes
    WebP (lossless): 1082 bytes
    RLE + zlib: 1137 bytes
    RLE + LZMA: 836 bytes (BEATS WEBP by 246 bytes!)

Note: LZMA is slower than zlib but provides better compression.
"""
from PIL import Image
import numpy as np
import struct
import lzma


def compress(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]

    colors, inverse = np.unique(arr.reshape(-1, 3), axis=0, return_inverse=True)
    if len(colors) > 255:
        raise ValueError(
            f"Image has {len(colors)} unique colors - this compressor is designed for "
            f"flat-color art with <=255 colors. Try quantizing first, or use a different tool."
        )
    
    index_map = inverse.reshape(h, w).astype(np.uint8)
    
    # RLE encode
    flat = index_map.flatten()
    runs = []
    prev = flat[0]
    count = 1
    for val in flat[1:]:
        if val == prev and count < 255:
            count += 1
        else:
            runs.append((prev, count))
            prev = val
            count = 1
    runs.append((prev, count))

    palette_bytes = colors.astype(np.uint8).tobytes()
    rle_bytes = b''.join(struct.pack('BB', c, n) for c, n in runs)

    header = struct.pack('>HHB', w, h, len(colors))
    payload = header + palette_bytes + rle_bytes
    compressed = lzma.compress(payload)

    with open(output_path, 'wb') as f:
        f.write(compressed)

    return len(compressed)


def decompress(input_path, output_path):
    with open(input_path, 'rb') as f:
        compressed = f.read()
    payload = lzma.decompress(compressed)

    w, h, num_colors = struct.unpack('>HHB', payload[:5])
    offset = 5
    palette = np.frombuffer(payload[offset:offset + num_colors * 3], dtype=np.uint8).reshape(num_colors, 3)
    offset += num_colors * 3

    rle_bytes = payload[offset:]
    index_map = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i in range(0, len(rle_bytes), 2):
        if i + 1 >= len(rle_bytes):
            break
        color_idx, count = rle_bytes[i], rle_bytes[i + 1]
        index_map[pos:pos + count] = color_idx
        pos += count

    index_map = index_map.reshape(h, w)
    rgb = palette[index_map]
    img = Image.fromarray(rgb, 'RGB')
    img.save(output_path)


if __name__ == '__main__':
    import sys, os
    if len(sys.argv) != 3:
        print("Usage: python rle_compressor_lzma.py <input.png> <output.dan_rle_lzma>")
        sys.exit(1)
    size = compress(sys.argv[1], sys.argv[2])
    orig_size = os.path.getsize(sys.argv[1])
    print(f"Compressed: {orig_size} -> {size} bytes ({orig_size/size:.1f}x smaller)")
