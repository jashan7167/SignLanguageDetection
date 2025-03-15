import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# Initialize video capture (camera)
capture = cv2.VideoCapture(2)

# Initialize hand detectors for detecting hand gestures
hd = HandDetector(maxHands=1)  # First detector for general hand detection
hd2 = HandDetector(maxHands=1)  # Second detector for detailed hand landmarks

# Directory setup for saving images
count = len(os.listdir("/home/jsb/Personal/Major/Sign-Language-To-Text-and-Speech-Conversion/AtoZ_3.0/A/"))
c_dir = 'A'  # Starting with the 'A' directory for capturing images

# Define parameters for image capture
offset = 15  # Padding around the hand for the region of interest (ROI)
step = 1
flag = False  # Flag to control whether gesture capturing is enabled
suv = 0  # Counter for the number of saved images

# Initialize white canvas to draw on (for visualizing hand skeleton)
white = np.ones((400, 400), np.uint8) * 255  # White background image
cv2.imwrite("/home/jsb/Personal/Major/Sign-Language-To-Text-and-Speech-Conversion/white.jpg", white)

# Main loop to process video feed
while True:
    try:
        # Capture frame from webcam
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)  # Flip frame for natural user interaction (mirror image)

        # Detect hands in the current frame
        hands = hd.findHands(frame, draw=False, flipType=True)

        # Load the white canvas for drawing hand landmarks
        white = cv2.imread("/home/jsb/Personal/Major/Sign-Language-To-Text-and-Speech-Conversion/white.jpg")

        if hands:
            hand = hands[0]  # Get the first detected hand
            x, y, w, h = hand['bbox']  # Bounding box coordinates for the detected hand
            # Crop the region of interest (ROI) around the hand
            image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])

            # Detect hand landmarks in the ROI
            handz, imz = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']  # List of landmarks of the hand

                # Calculate the offset to center the hand in the white canvas
                os = ((400 - w) // 2) - 15
                os1 = ((400 - h) // 2) - 15

                # Draw the hand skeleton by connecting landmarks with lines
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)

                # Additional lines to connect certain key points in the hand
                cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)

                # Draw circles at each landmark point to visualize it
                skeleton1 = np.array(white)
                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                # Show the hand skeleton
                cv2.imshow("Hand Skeleton", skeleton1)

        # Display the original frame with directory information
        frame = cv2.putText(frame, f"dir={c_dir}  count={count}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        # Keyboard interactions
        interrupt = cv2.waitKey(1)

        # Exit if the ESC key is pressed
        if interrupt & 0xFF == 27:  # ESC key
            break

        # Change directory (cycle through A-Z) when 'n' is pressed
        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) == ord('Z') + 1:
                c_dir = 'A'
            flag = False
            count = len(os.listdir(f"/home/jsb/Personal/Major/Sign-Language-To-Text-and-Speech-Conversion/AtoZ_3.0/{c_dir}/"))

        # Toggle gesture capture mode when 'a' is pressed
        if interrupt & 0xFF == ord('a'):
            flag = not flag  # Toggle the flag state
            suv = 0 if flag else suv  # Reset counter when flag is turned on

        # Save images when capturing gestures
        if flag:
            if suv == 180:  # Stop after saving 180 images
                flag = False
            if step % 3 == 0:  # Save every 3 frames
                cv2.imwrite(f"/home/jsb/Personal/Major/Sign-Language-To-Text-and-Speech-Conversion/AtoZ_3.0/{c_dir}/{count}.jpg", skeleton1)
                count += 1
                suv += 1
            step += 1

    except Exception:
        # Print any errors encountered
        print("==", traceback.format_exc())

# Release resources and close windows
capture.release()
cv2.destroyAllWindows()
