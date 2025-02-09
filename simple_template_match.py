import cv2 as cv

img_rgb = cv.imread("screenshot-1.png")
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread("ascent.png", cv.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

top_left = max_loc
h, w = template.shape
bottom_right = (top_left[0] + w, top_left[1] + h)

cv.rectangle(img_rgb, top_left, bottom_right, 255, 2)


cv.imwrite("res.png", img_rgb)
