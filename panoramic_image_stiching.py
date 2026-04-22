import cv2
import numpy as np

# jonah-paths 
img_paths = [f"../../material/harbor_{i}.jpg" for i in range(1, 7)]
# img_paths = ["../../material/Landscape1.jpg", "../../material/Landscape2.jpg"]
# img_paths = ["../../material/Traffic1.jpg", "../../material/Traffic2.jpg", "../../material/Traffic3.jpg"]

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
    # 0.75 is the threshold for the Lowe's ratio test
    best_matches = [m for m, n in matches if m.distance < 0.75*n.distance]

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

    # 8 DoF so we need at least 4 point pairs for a homography, 10 is stable apparently
    if len(pts1) < 10:
        print(f"Not enough matches to compute homography ({len(pts1)} < 10)")
        return None

    # Find homography
    M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    if M is None:
        print("findHomography failed to find a valid model")
        return None

    # NOTE for Dylan:
    # The below steps I added is to figure out the canvas size dynamically
    # Warp 4 corners of img2 through M to see where it will land in img1's frame
    # Now we can ensure nothing falls off the edges

    # Get the corners of img1 and img2
    corners1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)

    # warp corners of img2 through M
    warped_corners2 = cv2.perspectiveTransform(corners2, M)
    all_corners = np.concatenate([corners1, warped_corners2], axis=0)

    # canvas size + translation offsets
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    tx, ty = -x_min, -y_min                         # translation offsets
    canvas_w, canvas_h = x_max - x_min, y_max - y_min

    # translation matrix so nothing lands at negative coordinates.
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)

    # warp img2 into the canvas (translation + homography in one shot)
    panorama = cv2.warpPerspective(img2, T @ M, (canvas_w, canvas_h))

    # drop img1 into place at the translated location
    panorama[ty:ty + height1, tx:tx + width1] = img1

    img2_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY, None)
    _, threshold = cv2.threshold(img2_gray, 1, 255, cv2.THRESH_BINARY)
    img2_nonzero = cv2.findNonZero(threshold)
    x, y, w, h = cv2.boundingRect(img2_nonzero)
    panorama_cropped = panorama[y:y+h, x:x+w]


    # setting up final display for panorama
    if plot:
        cv2.imshow("Intermediate stitch", panorama_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # return the bbox-cropped panorama
    return panorama_cropped

for i in range(1, len(img_paths)):
    img = cv2.imread(img_paths[i])
    if img is None:
        print(f"Failed to load {img_paths[i]}")
        quit()

    print(f"Stitching image {i + 1}/{len(img_paths)}: {img_paths[i]}")
    pts1, pts2 = match_images(master_img, img)
    new_pano = RANSAC(pts1, pts2, master_img, img)
    if new_pano is None:
        print(f"  -> skipped {img_paths[i]} (could not stitch)")
        continue
    master_img = new_pano   # grow the panorama for the next iteration

cv2.imshow("Panorama", master_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


