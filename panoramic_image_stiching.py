import cv2
import numpy as np

img_paths = ["../Image-Data/harbor_1.jpg", "../Image-Data/harbor_2.jpg"]
#img_paths = ["../Image-Data/room1.jpg", "../Image-Data/room2.jpg"]

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
    points1 = np.float32([features1[m.queryIdx].pt for m in best_matches])
    points2 = np.float32([features2[m.trainIdx].pt for m in best_matches])

    if plot:
        matches = cv2.drawMatches(img1, features1, img2, features2, best_matches, None)
        cv2.imshow("Feature matches", matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points1, points2


def RANSAC(pts1, pts2, img1, img2, plot=False):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Find homography
    M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    # Creat panorama
    panorama = cv2.warpPerspective(img2, M, (width1 + width2, height2))

    # Place first image
    panorama[0:height1, 0:width1] = img1

    # Align second image
    img2_transformed = cv2.warpPerspective(img2, M, (width1, height1))
    img2_gray = cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY, None)
    _, threshold = cv2.threshold(img2_gray, 1, 255, cv2.THRESH_BINARY)
    img2_nonzero = cv2.findNonZero(threshold)
    x, y, w, h = cv2.boundingRect(img2_nonzero)
    panorama_cropped = panorama[y:y+h, x:x+w]

    cv2.imshow("Original 1", img1)
    cv2.imshow("Original 2", img2)
    cv2.imshow("Transformation (uncropped)", panorama)
    cv2.imshow("Transformation (cropped)", panorama_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for i in range(1, len(img_paths)):
    img = cv2.imread(img_paths[i])
    if img is None:
        print('Failed to load image')
        quit()
    #imgs.append(img)

    pts1, pts2 = match_images(master_img, img)
    RANSAC(pts1, pts2, master_img, img)


