"""
DAN Compressor - Pattern Dictionary backend (EXPERIMENTAL)
=============================================================
Status: WORKING but NOT YET COMPETITIVE
Tested result on anime_test.png (300x300, 8 colors):
    16x16 patches -> 2001 bytes
    vs rle_compressor.py on same image -> 1137 bytes
    vs PNG -> 2858 bytes
    vs WebP lossless -> 1084 bytes

This approach finds repeated fixed-size pixel blocks, stores each unique
block once in a dictionary, and references them by index. It currently
LOSES to the simple RLE approach because flat-color anime art's redundancy
is mostly long same-color RUNS, not repeated fixed-size tiles - so RLE
captures the real structure more directly.

Where this might still win: images with repeated COMPLEX shapes (e.g. two
identical eyes, tiled background elements) rather than just flat color
regions. Not yet tested on that case - a real next step, not a promise.

Keeping this in the repo because the finding itself is useful (see README
roadmap) and because it may become a useful *complementary* technique
combined with RLE, not a replacement for it.
"""
from PIL import Image
import numpy as np
import zlib
import struct


def compress(image_path, output_path, patch_size=16):
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]

    dictionary = {}
    tokens = []
    order = []

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            block = arr[y:y + patch_size, x:x + patch_size].tobytes()
            if block not in dictionary:
                dictionary[block] = len(dictionary)
                order.append(block)
            tokens.append(dictionary[block])

    dict_raw = b''.join(order)
    dict_compressed = zlib.compress(dict_raw, level=9)

    if len(dictionary) < 256:
        token_bytes = bytes(tokens)
    else:
        token_bytes = b''.join(struct.pack('>H', t) for t in tokens)
    token_compressed = zlib.compress(token_bytes, level=9)

    header = struct.pack('>HHBH', w, h, patch_size, len(dictionary))
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(struct.pack('>I', len(dict_compressed)))
        f.write(dict_compressed)
        f.write(token_compressed)

    total = len(header) + 4 + len(dict_compressed) + len(token_compressed)
    return total


if __name__ == '__main__':
    import sys, os
    if len(sys.argv) < 3:
        print("Usage: python dictionary_compressor.py <input.png> <output.dan_dict> [patch_size]")
        sys.exit(1)
    patch = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    size = compress(sys.argv[1], sys.argv[2], patch)
    orig_size = os.path.getsize(sys.argv[1])
    print(f"Compressed: {orig_size} -> {size} bytes ({orig_size/size:.2f}x)")
    print("NOTE: this backend is experimental and currently loses to rle_compressor.py")
