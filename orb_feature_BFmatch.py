import cv2 as cv

img_rgb = cv.imread("screenshot-1.png")
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread("ascent.png", cv.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

orb = cv.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(template, None)
kp2, des2 = orb.detectAndCompute(img_gray, None)


# Use BFMatcher to find matches
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
result = cv.drawMatches(template, kp1, img_gray, kp2, matches[:15], None, flags=2)

cv.imwrite("res.png", result)
