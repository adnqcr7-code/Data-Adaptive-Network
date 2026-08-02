from PIL import Image, ImageDraw
import numpy as np
import random
import os

os.makedirs('dataset', exist_ok=True)

SIZE = 128  # keep small for fast CPU testing

def rand_color(base, spread=40):
    return tuple(int(max(0, min(255, c + random.randint(-spread, spread)))) for c in base)

def make_face(seed):
    random.seed(seed)
    skin = rand_color((255, 224, 189), 20)
    hair = random.choice([(80,40,30), (30,30,30), (200,180,60), (150,80,150), (60,60,180)])
    hair = rand_color(hair, 15)
    eye = random.choice([(60,120,200), (80,160,80), (150,80,60), (100,60,150)])
    eye = rand_color(eye, 15)

    img = Image.new('RGB', (SIZE, SIZE), rand_color((255,240,230), 10))
    draw = ImageDraw.Draw(img)

    cx, cy = SIZE//2, SIZE//2 + random.randint(-5,5)
    face_w = random.randint(50, 70)
    face_h = random.randint(60, 85)
    draw.ellipse([cx-face_w//2, cy-face_h//2, cx+face_w//2, cy+face_h//2],
                 fill=skin, outline=(20,20,20), width=3)

    # hair - vary style
    style = random.choice(['round', 'side', 'long'])
    if style == 'round':
        draw.pieslice([cx-face_w//2-8, cy-face_h//2-25, cx+face_w//2+8, cy+10], 180, 360,
                      fill=hair, outline=(20,20,20), width=3)
    elif style == 'side':
        draw.pieslice([cx-face_w//2-15, cy-face_h//2-20, cx+face_w//2+5, cy+15], 170, 350,
                      fill=hair, outline=(20,20,20), width=3)
    else:
        draw.pieslice([cx-face_w//2-8, cy-face_h//2-25, cx+face_w//2+8, cy+30], 180, 360,
                      fill=hair, outline=(20,20,20), width=3)

    # eyes
    ew, eh = random.randint(12,18), random.randint(15,22)
    ex_off = random.randint(12,18)
    ey = cy - random.randint(0,8)
    draw.ellipse([cx-ex_off-ew//2, ey-eh//2, cx-ex_off+ew//2, ey+eh//2], fill=eye, outline=(10,10,10), width=2)
    draw.ellipse([cx+ex_off-ew//2, ey-eh//2, cx+ex_off+ew//2, ey+eh//2], fill=eye, outline=(10,10,10), width=2)
    # pupils
    pr = 4
    draw.ellipse([cx-ex_off-pr, ey-pr, cx-ex_off+pr, ey+pr], fill=(10,10,10))
    draw.ellipse([cx+ex_off-pr, ey-pr, cx+ex_off+pr, ey+pr], fill=(10,10,10))

    # mouth
    my = cy + random.randint(15,25)
    draw.arc([cx-15, my-8, cx+15, my+8], 0, 180, fill=(180,60,60), width=3)

    # blush (sometimes)
    if random.random() > 0.4:
        draw.ellipse([cx-ex_off-25, my-15, cx-ex_off-5, my], fill=(255,180,190))
        draw.ellipse([cx+ex_off+5, my-15, cx+ex_off+25, my], fill=(255,180,190))

    return img

N = 300
for i in range(N):
    img = make_face(i)
    img.save(f'dataset/face_{i:04d}.png')

print(f"Generated {N} synthetic anime-face images in dataset/")
