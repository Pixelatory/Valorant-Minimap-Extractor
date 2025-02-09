import cv2 as cv

img_rgb = cv.imread("screenshot-1.png")
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread("ascent.png", cv.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

orb = cv.ORB_create(nfeatures=1000)

# Find keypoints and descriptors
keypoints_template, descriptors_template = orb.detectAndCompute(template, None)
keypoints_full, descriptors_full = orb.detectAndCompute(img_gray, None)


# FLANN parameters for ORB (LSH algorithm)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=12,  # 12
                    key_size=20,     # 20
                    multi_probe_level=2)  # 2
search_params = dict(checks=1000)  # or pass an empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

# Find matches
matches = flann.knnMatch(descriptors_template, descriptors_full, k=2)

# Apply Lowe's ratio test (to filter good matches)
good_matches = []
for m, n in matches:
    if m.distance < .75 * n.distance:
        good_matches.append(m)

# Draw matches
result = cv.drawMatches(
    template,
    keypoints_template,
    img_gray,
    keypoints_full,
    good_matches[:20],
    None,
    flags=2,
)

cv.imwrite("res.png", result)
