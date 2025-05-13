# Importing Libraries
import numpy as np
import math
import cv2
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
ddd=enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
import tkinter as tk
from PIL import Image, ImageTk
offset=29

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(2)
        self.current_image = None
        self.model = load_model('Sign-Language-To-Text-and-Speech-Conversion/cnn8grps_rad1_model.h5')

        # Speech Engine - Improved for more natural sound
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)  # Natural speed
        self.speak_engine.setProperty("volume", 1.0)  # Max volume
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)  # Choose a more human-like voice

        self.ct = {'blank': 0}
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10

        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")

        # UI Setup
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1200x700")  # Increased window size
        self.root.configure(bg="#ffffff")  # White background

        # Fonts
        font_title = ("Helvetica", 24, "bold")
        font_labels = ("Helvetica", 16, "bold")
        font_buttons = ("Helvetica", 14)

        # Enlarged Video Panels
        self.panel = tk.Label(self.root, bg="#f0f0f0", relief="solid", bd=2)
        self.panel.place(x=50, y=50, width=500, height=350)

        self.panel2 = tk.Label(self.root, bg="#f0f0f0", relief="solid", bd=2)
        self.panel2.place(x=600, y=50, width=500, height=350)

        # Title
        self.T = tk.Label(self.root, text="Sign Language Detection", font=font_title, fg="#333333", bg="#ffffff")
        self.T.place(x=50, y=10)

        # Enlarged Character Display
        self.panel3 = tk.Label(self.root, bg="#ffffff", relief="solid", bd=2)
        self.panel3.place(x=300, y=450, width=150, height=70)

        self.T1 = tk.Label(self.root, text="Character:", font=font_labels, fg="#333333", bg="#ffffff")
        self.T1.place(x=50, y=460)

        # Enlarged Sentence Display
        self.panel5 = tk.Label(self.root, bg="#ffffff", relief="solid", bd=2)
        self.panel5.place(x=300, y=540, width=500, height=70)

        self.T3 = tk.Label(self.root, text="Sentence:", font=font_labels, fg="#333333", bg="#ffffff")
        self.T3.place(x=50, y=550)

        # Suggestions
        self.T4 = tk.Label(self.root, text="Suggestions:", fg="red", font=font_labels, bg="#ffffff")
        self.T4.place(x=50, y=630)

        # Button Styling
        button_bg = "#4CAF50"
        button_fg = "#ffffff"
        button_width = 120
        button_height = 50

        self.b1 = tk.Button(self.root, bg=button_bg, fg=button_fg, font=font_buttons, relief="solid", bd=2)
        self.b1.place(x=250, y=630, width=button_width, height=button_height)

        self.b2 = tk.Button(self.root, bg=button_bg, fg=button_fg, font=font_buttons, relief="solid", bd=2)
        self.b2.place(x=390, y=630, width=button_width, height=button_height)

        self.b3 = tk.Button(self.root, bg=button_bg, fg=button_fg, font=font_buttons, relief="solid", bd=2)
        self.b3.place(x=530, y=630, width=button_width, height=button_height)

        self.b4 = tk.Button(self.root, bg=button_bg, fg=button_fg, font=font_buttons, relief="solid", bd=2)
        self.b4.place(x=670, y=630, width=button_width, height=button_height)

        # Bigger Speak & Clear Buttons
        self.speak = tk.Button(self.root, text="Speak", font=font_buttons, command=self.speak_fun,
                               bg=button_bg, fg=button_fg, relief="solid", bd=2)
        self.speak.place(x=850, y=540, width=button_width, height=button_height)

        self.clear = tk.Button(self.root, text="Clear", font=font_buttons, command=self.clear_fun,
                               bg=button_bg, fg=button_fg, relief="solid", bd=2)
        self.clear.place(x=1000, y=540, width=button_width, height=button_height)

        # Variables
        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        self.video_loop()

    def destructor(self):
        """
        Clean up resources and close the application.
        """
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def video_loop(self):
        try:
            #video capture and preprocessing
            ok, frame = self.vs.read()
            #captures a frame from the webcam and horizontally flips it for a mirror like effect
            cv2image = cv2.flip(frame, 1)
            #check frame is valid use hand tracking module to detect hands in the frame converts the frame to RGB for displaying in the Tkinter GUI
            if cv2image.any:
                hands = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy=np.array(cv2image)
                #converts the processed frame into a format compatible with tkinter
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                #updates the gui video display with the current frame
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)
                #if hand is detected 
                if hands[0]:
                    hand = hands[0]
                    map = hand[0]
                    #extract bounding box around the hand
                    x, y, w, h=map['bbox']
                    #crops hand region with a margin(offset) important for isolating the hand and improving model perf
                    image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                    #load white background for handlandmarks
                    white = cv2.imread("Sign-Language-To-Text-and-Speech-Conversion/white.jpg")
                    # img_final=img_final1=img_final2=0
                    if image.all:
                        handz = hd2.findHands(image, draw=False, flipType=True)
                        self.ccc += 1
                        if handz[0]:
                            hand = handz[0]
                            handmap=hand[0]
                            #detect landmarks of the cropped hand image
                            self.pts = handmap['lmList']
                            # x1,y1,w1,h1=hand['bbox']
                            
                            #offsets for the proper placement on the white image
                            os = ((400 - w) // 2) - 15
                            os1 = ((400 - h) // 2) - 15
                            #draw lines connecting hand landmarks on the white image
                            for t in range(0, 4, 1):
                                #connect landmarks from the self.pts having landmarks 0 to 4 represents the thumb
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            #index fingers
                            for t in range(5, 8, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            #middle finger
                            for t in range(9, 12, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            #ring finger
                            for t in range(13, 16, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            #pinky finger
                            for t in range(17, 20, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            
                            #connect the palm from base landmarks (0,5,9,13,17)
                            cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                                     (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                                     3)
                            #circles on landmarks
                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)
                            #the processed hand image the green landmarks on the white background is passed onto the method for gesture recognition
                            res=white
                            self.predict(res)


                            #display the processed hand image with landmarks in another panel
                            self.current_image2 = Image.fromarray(res)
                            imgtk = ImageTk.PhotoImage(image=self.current_image2)
                            self.panel2.imgtk = imgtk
                            self.panel2.config(image=imgtk)
                            #self.panel4.config(text=self.word, font=("Courier", 30))


                            #update the gui with the current detected symbol
                            self.panel3.config(text=self.current_symbol, font=("Courier", 30))



                            self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                            self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825,  command=self.action2)
                            self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825,  command=self.action3)
                            self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825,  command=self.action4)

                self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)


#calculates the euclidean distance between two points x and y for analyzing hand geometry
    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


    #modify the senteces displayed on the Gui based on the user interaction using enchant we are detecting suggested words
    def action1(self):
        #finds the last word and replace it with the first suggested word
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()


    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str=self.str[:idx_word]
        self.str=self.str+self.word2.upper()
        #self.str[idx_word:last_idx] = self.word2


    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()



    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()

    #converts the current sentence(self.str) into the speech using the pyttsx3 library
    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    
    def clear_fun(self):
        self.str=" "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "


    def predict(self, test_image):
        # Assign the input image to the 'white' variable
        white=test_image
         # Reshape the image to the required shape for the model (batch size, height, width, channels)
        white = white.reshape(1, 400, 400, 3)
        # Get the predicted probabilities from the model (array of probabilities for each class)
        prob = np.array(self.model.predict(white)[0], dtype='float32')

        # Find the index of the maximum probability (most likely class) and remove it from further consideration
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
         # Find the next most probable class and remove it from further consideration
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0

        # Find the third most probable class
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0
        # List of predefined valid combinations of the top two predicted classes
        pl = [ch1, ch2]

            # Gesture Groups:
            # Group 0: A, E, M, N, S, T
            # Group 1: B, D, F, I, K, R, U, V, W
            # Group 2: C, O
            # Group 3: G, H
            # Group 4: L, X
            # Group 5: P, Q, Z
            # Group 6: X (alternative conditions)
            # Group 7: Y, J

            # Hand Landmarks (self.pts):
            # --------------------------
            # Each index in self.pts corresponds to a specific hand landmark (21 landmarks in total):
            #   0: Wrist
            #   1: Thumb base (CMC joint)
            #   2: Thumb middle joint (MCP joint)
            #   3: Thumb tip (IP joint)
            #   4: Index finger base (MCP joint)
            #   5: Index finger middle joint (PIP joint)
            #   6: Index finger tip (DIP joint)
            #   7: Middle finger base (MCP joint)
            #   8: Middle finger middle joint (PIP joint)
            #   9: Middle finger tip (DIP joint)
            #   10: Ring finger base (MCP joint)
            #   11: Ring finger middle joint (PIP joint)
            #   12: Ring finger tip (DIP joint)
            #   13: Pinky finger base (MCP joint)
            #   14: Pinky finger middle joint (PIP joint)
            #   15: Pinky finger tip (DIP joint)


        #Landmarks used here
        # - Thumb: self.pts[4] (thumb tip),Index 1: Thumb base 
        # - Index Finger: self.pts[8] (index finger tip), self.pts[6] (index finger base)
        # - Middle Finger: self.pts[12] (middle finger tip), self.pts[10] (middle finger base)
        # - Ring Finger: self.pts[16] (ring finger tip), self.pts[14] (ring finger base)
        # - Pinky Finger: self.pts[20] (pinky finger tip), self.pts[18] (pinky finger base)
        # - Wrist: self.pts[0] (wrist)


        # Hierarchy flow of the program: first the groups 0 to 7 are checked, then the specific gestures within each group are checked.
        #predefined list of gesture conditions (pairs of classes) that correspond to specific signs
        # Group 0: A, E, M, N, S, T
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        
          # Check if the combination of ch1 and ch2 exists in the predefined list 'l'
        if pl in l:
                # Condition for Group 0 gestures: Check if all fingers are bend.
            # Fingers are extended if the base joint (e.g., self.pts[6]) is below the tip (e.g., self.pts[8]).
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        # These gestures are characterized by specific thumb and index finger positions.
        l = [[2, 2], [2, 1]]
        if pl in l:
            # Check if the thumb (self.pts[4]) is to the left of the index finger middle joint (self.pts[5]).
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                         

       # Condition for [C, 0][A, E, M, N, S, T] gestures depending on hand geometry
        # These gestures are characterized by the hand's orientation and thumb position.
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the wrist (self.pts[0]) is to the right of all fingertips and if the thumb is to the right of the index finger base.
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2


        # Group-2 : C, O
        # condition for [c,o][aemnst] depending on distance
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            #Check if the distance between the index finger tip (self.pts[8]) and ring finger tip (self.pts[16]) is less than 52.
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        # Group-3 : G,H
        # condition for [gh][bdfikruvw]
        # Check if the index finger is bent (self.pts[6][1] > self.pts[8][1]) and if the wrist is to the left of all fingertips.
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the thumb (self.pts[4]) is to the right of the wrist (self.pts[0]
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the middle finger tip (self.pts[16]) is significantly below the base of the index finger (self.pts[2]).
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3


        # Group 4: L, X
        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the distance between the thumb tip (self.pts[4]) and the middle finger base (self.pts[11]) is greater than 55.
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the distance between the thumb tip (self.pts[4]) and the middle finger base (self.pts[11]) is greater than 50,
            # and if the index finger is bent while other fingers are extended.
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the thumb (self.pts[4]) is to the left of the wrist (self.pts[0])
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the base of the index finger (self.pts[1]) is to the left of the middle finger tip (self.pts[12]).
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4



        # Group 5 : P, Q, Z
        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the index finger is bent while other fingers are extended,
              # and if the thumb (self.pts[4]) is above the middle finger tip (self.pts[10]).
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        
        # Condition for [G, H][P, Q] gestures
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
              # Check if the thumb (self.pts[4]) is significantly above all fingertips
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # Condition for [L][P, Q, Z] gestures
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:

            #check if the thumb (self.pts[4]) is to the right of the wrist (self.pts[0]).
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # Condition for [P, Q, Z][A, E, M, N, S, T] gestures
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the wrist (self.pts[0]) is to the left of all fingertips.
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5


        # Group 7 : P, Q, Z
        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the base of the index finger (self.pts[3]) is to the left of the wrist (self.pts[0]).
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the index finger is extended (self.pts[6][1] < self.pts[8][1]).
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the pinky finger (self.pts[18]) is bent (self.pts[18][1] > self.pts[20][1]).
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7



        # Group 6 : X
        # Condition for [X][A, E, M, N, S, T] gestures
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the base of the index finger (self.pts[5]) is to the right of the ring finger tip (self.pts[16]).
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6


        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
             #Check if the pinky finger is bent and the index finger is extended.
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the distance between the index finger tip (self.pts[8]) and ring finger tip (self.pts[16]) is greater than 50.
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
              # Check if the distance between the thumb tip (self.pts[4]) and the middle finger base (self.pts[11]) is less than 60.
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the base of the index finger (self.pts[5]) is significantly to the right of the thumb tip (self.pts[4]).
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6


        #Group 1 : 
        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if all fingers are bent (self.pts[6][1] > self.pts[8][1], etc.).
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the index finger is extended while other fingers are bent.
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the index finger is bent while other fingers are extended,
            # and if the thumb (self.pts[4]) is above the middle finger tip (self.pts[14]).
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
           
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the index and middle fingers are extended while the pinky finger is bent
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the thumb (self.pts[4]) is close to the index finger base (self.pts[5]),
             # and if the index and middle fingers are extended while the pinky finger is bent.
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
              # Check if the index and middle fingers are bent while the ring and pinky fingers are extended,
    # and if the thumb (self.pts[4]) is above the middle finger tip (self.pts[14]).
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
             # Check if the hand is neither fully to the left nor fully to the right of the fingertips,
    # and if the distance between the thumb tip (self.pts[4]) and the middle finger base (self.pts[11]) is less than 50.
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            # Check if the index, middle, and ring fingers are bent.
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1


        # -------------------------condn for 8 groups  ends


        # condn for subgroups  starts


        # Group 0: A, E, M, N, S, T
        # --------------------------
        if ch1 == 0:
            ch1 = 'S'  # Default gesture for Group 0 is 'S'.

            # Check for gesture 'A':
            # Thumb tip (self.pts[4]) is to the left of all other finger bases.
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'

            # Check for gesture 'T':
            # Thumb tip is between index and middle finger bases, and below ring and pinky finger tips.
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'

            # Check for gesture 'E':
            # Thumb tip is below all other fingertips.
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'

            # Check for gesture 'M':
            # Thumb tip is to the right of index, middle, and ring finger bases, and below pinky finger tip.
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'

            # Check for gesture 'N':
            # Thumb tip is to the right of index and middle finger bases, and below ring and pinky finger tips.
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'


        # Group 2: C, O
        # -------------
        if ch1 == 2:
            # Check for gesture 'C':
            # Distance between middle finger tip (self.pts[12]) and thumb tip (self.pts[4]) is greater than 42.
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'  # Otherwise, gesture is 'O'.


        # Group 3: G, H
        # -------------
        if ch1 == 3:
            # Check for gesture 'G':
            # Distance between index finger tip (self.pts[8]) and middle finger tip (self.pts[12]) is greater than 72.
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'  # Otherwise, gesture is 'H'.


        # Group 7: Y, J
        # -------------
        if ch1 == 7:
            # Check for gesture 'Y':
            # Distance between index finger tip (self.pts[8]) and thumb tip (self.pts[4]) is greater than 42.
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'  # Otherwise, gesture is 'J'.


        # Group 4: L
        # ----------
        if ch1 == 4:
            ch1 = 'L'  # Gesture is always 'L' for Group 4.


        # Group 6: X
        # ----------
        if ch1 == 6:
            ch1 = 'X'  # Gesture is always 'X' for Group 6.


        # Group 5: P, Q, Z
        # ----------------
        if ch1 == 5:
            # Check for gesture 'Z' or 'Q':
            # Thumb tip is to the right of middle, ring, and pinky finger tips.
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                # Check if index finger tip is above its base.
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'  # Otherwise, gesture is 'P'.


        # Group 1: B, D, F, I, K, R, U, V, W
        # -----------------------------------
        if ch1 == 1:
            # Check for gesture 'B':
            # All fingers are bent (tips are below their bases).
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'

            # Check for gesture 'D':
            # Index finger is bent, while other fingers are extended.
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'

            # Check for gesture 'F':
            # Index finger is extended, while other fingers are bent.
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'

            # Check for gesture 'I':
            # Index and middle fingers are extended, while pinky finger is bent.
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'

            # Check for gesture 'W':
            # Index, middle, and ring fingers are bent, while pinky finger is extended.
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'

            # Check for gesture 'K':
            # Index and middle fingers are bent, while ring and pinky fingers are extended, and thumb is below middle finger tip.
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'

            # Check for gesture 'U':
            # Index and middle fingers are close together and bent, while ring and pinky fingers are extended.
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'

            # Check for gesture 'V':
            # Index and middle fingers are spread apart and bent, while ring and pinky fingers are extended, and thumb is above middle finger tip.
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            # Check for gesture 'R':
            # Index finger is to the right of middle finger, and index and middle fingers are bent, while ring and pinky fingers are extended.
            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'



        # Open palm gesture to determine next
        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"


        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'


        if ch1=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.str=self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]


        if ch1=="  " and self.prev_char!="  ":
            self.str = self.str + "  "

        self.prev_char=ch1
        self.current_symbol=ch1
        self.count += 1
        self.ten_prev_char[self.count%10]=ch1


        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            if len(word.strip())!=0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4:
                    self.word4 = ddd.suggest(word)[3]

                if lenn >= 3:
                    self.word3 = ddd.suggest(word)[2]

                if lenn >= 2:
                    self.word2 = ddd.suggest(word)[1]

                if lenn >= 1:
                    self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "


print("Starting Application...")

(Application()).root.mainloop()
