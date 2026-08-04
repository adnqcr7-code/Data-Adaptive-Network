import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os, struct
from dan_model import DAN

device = 'cpu'
model = DAN().to(device)
model.load_state_dict(torch.load('dan_model.pt', map_location=device))
model.eval()

tf = transforms.ToTensor()

def compress(path):
    img = Image.open(path).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        code = model.encoder(x)
    code_np = code.squeeze(0).numpy().astype(np.float16)  # half precision to save space
    return code_np, img

def decompress(code_np):
    code = torch.from_numpy(code_np.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model.decoder(code)
    arr = (recon.squeeze(0).permute(1,2,0).numpy() * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr)

# Test on images NOT overfit -- use a couple from the dataset for now (same distribution)
test_files = ['dataset/face_0005.png', 'dataset/face_0150.png', 'dataset/face_0299.png']

for f in test_files:
    orig_png_size = os.path.getsize(f)

    code, orig_img = compress(f)
    code_bytes = code.tobytes()  # this IS the compressed file content
    code_size = len(code_bytes)

    recon_img = decompress(code)
    out_name = f.replace('dataset/', 'recon_').replace('.png', '_recon.png')
    recon_img.save(out_name)

    # pixel error
    orig_arr = np.array(orig_img).astype(np.float32)
    recon_arr = np.array(recon_img).astype(np.float32)
    mse = np.mean((orig_arr - recon_arr) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')

    print(f"{f}")
    print(f"  Original PNG:  {orig_png_size} bytes")
    print(f"  DAN code:      {code_size} bytes  ({orig_png_size/code_size:.1f}x smaller)")
    print(f"  PSNR (quality, higher=better, >30 is decent): {psnr:.1f} dB")
    print(f"  Reconstructed saved to: {out_name}")
    print()
