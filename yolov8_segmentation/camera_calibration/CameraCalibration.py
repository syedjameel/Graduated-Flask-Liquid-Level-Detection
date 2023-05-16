import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob


def get_calibration_params(images_path, plot_figs=False):
    # Initializing the variables
    square_size = 16  # in milli-meter
    img_mask = f"{images_path}/photo*.jpg"  # Path to the images
    pattern_size = (9, 6)  # Total Number of squares in rows and columns
    figsize = (20, 80)  # Plot figure size

    img_names = glob(img_mask)
    num_images = len(img_names)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    obj_points = []
    img_points = []
    h, w = cv2.imread(img_names[0]).shape[:2]

    print(f"Image size (Height*Width) = ({h}*{w})")

    if plot_figs:
        plt.figure(figsize=figsize)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i, fn in enumerate(img_names):
        print("processing %s... " % fn)
        imgBGR = cv2.imread(fn)

        if imgBGR is None:
            print("Failed to load", fn)
            continue

        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

        # Finds the positions of internal corners of the chessboard.
        # image	Source chessboard view. It must be an 8-bit grayscale or color image.
        # patternSize	Number of inner corners per a chessboard row and column.
        # corners	Output array of detected corners.

        found, corners = cv2.findChessboardCorners(img, pattern_size)

        if not found:
            print("chessboard not found")
            continue
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria)

        # Draw and display the corners

        img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners2, found)
        if plot_figs:
            plt.plot()
            plt.imshow(img_w_corners)
            plt.show()

        print(f"{fn}... OK")
        img_points.append(corners2)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2 = cv2.projectPoints(obj_points[i], _rvecs[i], _tvecs[i], camera_matrix, dist_coefs)[0]
        error = cv2.norm(img_points[i].reshape(-1, 1, 2), imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error / len(obj_points)))

    # undistort the image with the calibration
    # plt.figure(figsize=figsize)
    if plot_figs:
        for i, fn in enumerate(img_names):

            imgBGR = cv2.imread(fn)
            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

            dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)

            if i < 12:
                plt.subplot(4, 3, i + 1)
                plt.imshow(dst)
        plt.show()
    print("Done")

    return rms, camera_matrix, dist_coefs, _rvecs, _tvecs


if __name__ == "__main__":

    # get the calibration params
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = get_calibration_params(images_path="chess-board-imgs", plot_figs=True)




