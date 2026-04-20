import cv2
import numpy as np

img_paths = ["./Image-Data/harbor_1.jpg", "./Image-Data/harbor_2.jpg"]


imgs = []
features = []
descriptors = []
master_img = cv2.imread(img_paths[0])
sift = cv2.SIFT_create()



### Step 1: Feature Detection using SIFT ###
def match_images(img1, img2, plot=False):
    features1, descriptors1 = sift.detectAndCompute(img1, None)
    features2, descriptors2 = sift.detectAndCompute(img2, None)

    bfMatcher = cv2.BFMatcher(normType=cv2.NORM_L2)

    # Lowe's ratio test:
    # First, we match each feature in img1 with 2 (k=2) features in img2
    # Then, we only use the feature if there is one match that is significantly better than the other--presumeably erroneous--matching
    matches = bfMatcher.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
    best_matches = [m for m, n in matches if m.distance < 0.3*n.distance]

    # Extract the points in each image from the best matches 
    points1 = [features1[m.queryIdx].pt for m in best_matches]
    points2 = [features2[m.queryIdx].pt for m in best_matches]

    if plot:
        matches = cv2.drawMatches(img1, features1, img2, features2, best_matches, None)
        cv2.imshow("Feature matches", matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points1, points2


for i in range(1, len(img_paths)):
    img = cv2.imread(img_paths[i])
    if img is None:
        print('Failed to load image')
        quit()
    imgs.append(img)

    match_images(master_img, img)



