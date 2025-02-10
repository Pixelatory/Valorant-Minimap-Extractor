import cv2
import numpy as np

# NOTE: this is currently unused.

# Load images
img_rgb = cv2.imread("screenshot-2.png")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread("ascent.png", cv2.IMREAD_GRAYSCALE)

# Initialize SURF detector (SURF is part of opencv_contrib and needs to be enabled)
surf = cv2.xfeatures2d.SURF_create()

# Find keypoints and descriptors
keypoints_template, descriptors_template = surf.detectAndCompute(template, None)
keypoints_full, descriptors_full = surf.detectAndCompute(img_gray, None)

# Use FLANN matcher (faster than brute force for SURF)
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Find matches
matches = flann.knnMatch(descriptors_template, descriptors_full, k=2)

# Apply Lowe's ratio test (to filter good matches)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
result = cv2.drawMatches(
    template,
    keypoints_template,
    img_gray,
    keypoints_full,
    good_matches[:20],
    None,
    flags=2,
)

cv2.imwrite("res.png", result)

# Get the matching keypoints
pts_template = np.float32(
    [keypoints_template[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)
pts_full = np.float32([keypoints_full[m.trainIdx].pt for m in good_matches]).reshape(
    -1, 1, 2
)

# Estimate the transformation matrix (Homography or Affine)
# For rotation and scaling, an affine transform should work
something, mask = cv2.estimateAffine2D(pts_template, pts_full)

# Apply the transformation to the template image
rows, cols = img_gray.shape
transformed_template = cv2.warpAffine(template, something, (cols, rows))

cv2.imwrite("res_transformed.png", transformed_template)

# Create a mask of the transformed template
# This mask will help us blend the template onto the full image
_, mask = cv2.threshold(transformed_template, 1, 255, cv2.THRESH_BINARY)

# Overlay the transformed template onto the full image
# Mask the template and place it on the full image
template_color = cv2.cvtColor(transformed_template, cv2.COLOR_GRAY2BGR)
template_masked = cv2.bitwise_and(template_color, template_color, mask=mask)

# Now overlay the masked template onto the full image
full_image_with_overlay = cv2.addWeighted(img_rgb, 1, template_masked, 0.5, 0)

cv2.imwrite("res_overlay.png", full_image_with_overlay)
