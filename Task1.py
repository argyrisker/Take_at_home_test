from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def read_resize_normalize_image(image_path):
    #read the image
    img = Image.open(image_path).convert("RGB")

    #resize the image to 64x64
    img = img.resize((64, 64))

    #normalize to [0, 1]
    img_arr = np.asarray(img, dtype=np.float32) / 255.0 #(H,W,C)
    img_arr = (img_arr - 0.5) / 0.25
    img_arr = np.transpose(img_arr, (2, 0, 1))  #(C, H, W)

    return img_arr
