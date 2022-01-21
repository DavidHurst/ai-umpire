import os

import cv2

from skimage.metrics import structural_similarity as compare_ssim
import imutils

frames_dir = os.getcwd() + '/players_sim_blurred'

img1 = cv2.imread(frames_dir + '/blurred0.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(frames_dir + '/blurred1.jpg', cv2.IMREAD_GRAYSCALE)

(score, diff) = compare_ssim(img1, img2, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)

for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Image 1 - With bounding box", img1)
cv2.imshow("Difference", diff)
cv2.imwrite(frames_dir + 'frame_difference.jpg', diff)
cv2.waitKey(0)
