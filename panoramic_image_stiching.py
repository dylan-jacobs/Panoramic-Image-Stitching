import cv2
import numpy as np

img_paths = ["./Image-Data/harbor_1.jpg", "./Image-Data/harbor_2.jpg"]


imgs = []
features = []
descriptors = []
sift = cv2.SIFT_create()

for i in range(0, len(img_paths)):
    img = cv2.imread(img_paths[i])
    if img is None:
        print('Failed to load image')
        quit()
    imgs.append(img)

    ### Step 1: Feature Detection using SIFT ###
    features[i], descriptors[i] = sift.detectAndCompute(img)
    




