import os
import numpy as np
import cv2

# Calibrate camera using checkerboard method to obtain camera parameters (extrinsic & intrinsic matrices)
checkerboard_images_dir = "/mnt/c/checkerboard images" # get about 10-15 photos with different orientations and angles (if possible)
checkerboard = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

# Loop through checkerboard images for calibration
for file in os.listdir(checkerboard_images_dir):
    img = cv2.imread(os.path.join(checkerboard_images_dir, file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, checkerboard, corners, ret)
        h, w = img.shape[:2]

    # cv2.imshow("img", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# Obtain intrinsic, extrinsic, and distortion coefficients to form the projection matrix
ret, camera_matrix, distCoeffs, rotation_vecs, translation_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Obtain undistorted camera matrix
newcam_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distCoeffs, (w, h), 1, (w, h))

# print(rotation_vecs)

intrinsic_matrix = np.array(newcam_matrix)
rotation_vecs = np.array(rotation_vecs)
translation_vecs = np.array(translation_vecs[0])
print(rotation_vecs.shape)
print(translation_vecs.shape)
# extrinsic_matrix = np.concatenate((rotation_vecs, translation_vecs), axis=0)
# print(extrinsic_matrix.shape)
distCoeffs = np.array(distCoeffs)

'''
np.savetxt("/mnt/c/camera parameters/cam_intrinsic_matrix.txt", intrinsic_matrix, fmt='%d', delimiter=',')
np.savetxt("/mnt/c/camera parameters/cam_extrinsic_matrix.txt", extrinsic_matrix, fmt='%d', delimiter=',')
np.savetxt("/mnt/c/camera parameters/cam_distCoeffs.txt", distCoeffs, fmt='%d', delimiter=',')

'''

