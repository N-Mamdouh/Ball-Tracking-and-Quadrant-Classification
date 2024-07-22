# Nahed Mamdouh AbdelRahman #
# 25 - AUG - 2023 #
# MTE 400, Summer Semester 2023 #
# Computer Vision Project #

# import openCV library
import cv2 as cv

# import numpy library
import numpy as np


# define function to determine ball center
def get_contour_center(contour):  # function helps to find the center or area of an object
    m = cv.moments(contour)
    cx = -1
    cy = -1

    if m['m00'] != 0:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

    return cx, cy


# define function to determine which quarter is the ball in
def get_quarter(x, y, image_size):
    if (x >= image_size[1] // 2) and (y <= image_size[0] // 2):
        return 1
    elif (x <= image_size[1] // 2) and (y <= image_size[0] // 2):  # x value is large, y value is small
        return 2
    elif (x <= image_size[1] // 2) and (y >= image_size[0] // 2):
        return 3
    elif (x >= image_size[1] // 2) and (y >= image_size[0] // 2):
        return 4


# import the video
cap = cv.VideoCapture("test.avi")

while cap.isOpened():  # as long as the video is running
    # capture frames from the video
    ret, img = cap.read()

    if ret:  # if ret == True
        # convert original image to HSV image
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # determine lower and upper values for HSV
        # store BGR values in an array, we obtained this values by using online color calculator
        bgr_yellow = np.uint8([[[35, 173, 167]]])
        # convert the given BGR values to HSV
        hsv_yellow = cv.cvtColor(bgr_yellow, cv.COLOR_BGR2HSV)

        # compute lower and upper values
        lower_limit = np.array([[[hsv_yellow[0][0][0] - 10, 80, 80]]])
        upper_limit = np.array([[[hsv_yellow[0][0][0] + 10, 255, 255]]])

        # generate the binary image
        bin_img = cv.inRange(hsv_img, lower_limit, upper_limit)

        # apply the binary image as mask to the original image
        final_img = cv.bitwise_and(img, img, mask=bin_img)

        # get contours from binary image
        contours, hierarchy = cv.findContours(bin_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # filter contour with the area
        filtered_contours = []

        for c in contours:
            # get the contour area to determine where its location
            area = cv.contourArea(c)
            if area > 500:
                filtered_contours.append(c)
                # append() -> inserts a single element into an existing list

            # draw contour on original frame and binary frame
        if len(filtered_contours) == 0:
            print("No ball found!")
        else:
            for c in filtered_contours:
                # draw a closed circle with x, y and radius
                ((x, y), radius) = cv.minEnclosingCircle(c)
                # draw contours for original and final images
                cv.drawContours(img, [c], -1, (255, 0, 0), 1)
                cv.drawContours(final_img, [c], -1, (255, 0, 0), 1)

                # define ball center
                cx, cy = (int(x), int(y))

                # draw a circle about the contour of the ball
                cv.circle(img, (cx, cy), (int)(radius), (0, 0, 255), 1)
                cv.circle(final_img, (cx, cy), (int)(radius), (0, 0, 255), 1)

                # draw a small circle fo the center of the circle/ contour
                cv.circle(img, (cx, cy), 5, (180, 150, 90), -1)
                cv.circle(final_img, (cx, cy), 5, (180, 150, 90), -1)

        quarter = get_quarter(cx, cy, final_img.shape)
        final_img = cv.putText(final_img, str(quarter), (20, 55), cv.FONT_HERSHEY_SIMPLEX, 2.2, (40, 200, 150), 2, cv.LINE_AA)
        # display the frames after detection
        cv.imshow("original", img)
        cv.imshow("binary", bin_img)
        cv.imshow("final", final_img)

        key = cv.waitKey(50)
        if key == 27:
            break
    else:
        break

cap.release()
cv.destroyAllWindows()

