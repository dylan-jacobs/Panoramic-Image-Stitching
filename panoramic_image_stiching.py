import cv2

img_paths = ["./Image-Data/harbor_1.jpg", "./Image-Data/harbor_2.jpg"]


imgs = []
for i in range(0, len(img_paths)):
    img = cv2.imread(img_paths[i])
    if img is None:
        print('Failed to load image')
        quit()
    imgs.append(img)

### Step 1: Feature Detection using SIFT ###




