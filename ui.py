import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
import mediapipe as mp
# from mediapipe.python.solutions import selfie_segmentation, drawing_utils, face_mesh, face_detection, hands, pose, holistic
import matplotlib.pyplot as plt
import cv2
import numpy as np

class ImageDisplayApp:
    H = 1000
    W = 1400
    def __init__(self, master):
        H = self.H
        W = self.W
        self.master = master
        self.master.title("Image Display App")
        
        # Create label and text box for file path input
        self.path_label = tk.Label(self.master, text="Enter file path:")
        self.path_label.pack()
        self.path_entry = tk.Entry(self.master)
        self.path_entry.pack()

        # Create button to display original image
        self.original_button = tk.Button(self.master, text="Original Image", command=self.display_original)
        self.original_button.pack()

        # Create button to display augmented image
        self.augmented_button = tk.Button(self.master, text="Augmented Image", command=self.display_augmented)
        self.augmented_button.pack()

        # Create a canvas to contain the image
        self.canvas = tk.Canvas(self.master, width=W, height=H)
        self.canvas.pack()



        self.canvas.config(scrollregion=(0, 0, W, H))

        # Bind mouse wheel events to zoom in and out
        self.canvas.bind_all(("<MouseWheel>", self.zoom))




    def display_original(self):
        H = self.H
        W = self.W
        # Get the file path from the text box
        file_path = self.path_entry.get()

        # Open the image file
        image = Image.open(file_path)

        image = np.asarray(image)

        image = cv2.resize(image, (W, H))

        # Convert the image to a format that tkinter can display
        photo = ImageTk.PhotoImage(image = Image.fromarray(image))

        # Create a label to display the image on the canvas
        self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=photo)

        # Save the original image and photo for zooming
        self.original_image = image
        self.original_photo = photo

    def display_augmented(self):
        H = self.H
        W = self.W
        image = self.original_image
        # Create an enhanced version of the image

        #run facial landmark detection
        with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.6) as face_mesh:
                results = face_mesh.process(image= image)

        face_landmarks = results.multi_face_landmarks[0]
        
        keypoints = np.array([(W*point.x,H*point.y) for point in face_landmarks.landmark[0:468]])#after 468 is iris or something else
        # plt.tight_layout()
        # plt.figure(figsize=(35,35),dpi=250)
        markup = image.copy()
        colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,255,255),(255,255,255)]
        index = 0
        for i in range(len(keypoints)-2):
            #change text size to 4
            # cv2.putText(markup, str(i), (int(keypoints[i,0]), int(keypoints[i,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            #annotate the points using cv2
            cv2.putText(img=markup, text=str(i), org=(int(keypoints[i,0]+5), int(keypoints[i,1]+5)), fontFace=cv2.LINE_AA, fontScale=0.3, color=colors[index], thickness=1)
            #draw points for each landmark
            cv2.circle(img=markup, center=(int(keypoints[i,0]), int(keypoints[i,1])), radius=2, color=colors[index], thickness=2)
            index += 1
            if index >5:
                index = 0
        # Convert the image to a format that tkinter can display
        photo = ImageTk.PhotoImage(image=Image.fromarray(markup))

        # Create a label to display the image on the canvas fill available space
        self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=photo)

        # Save the original image and photo for zooming
        self.original_image = image
        self.original_photo = photo
    def zoom(self, event):
        # Get the current scale of the canvas
        current_scale = self.canvas.scale(tk.ALL, 0, 0)

        # Determine the new scale based on the mouse wheel direction
        if event.delta > 0:
            new_scale = current_scale * 1.1
        else:
            new_scale = current_scale / 1.1

        # Zoom in or out on the canvas
        self.canvas.scale(tk.ALL, 0, 0, new_scale, new_scale)

        # Rescale the image to match the new canvas scale
        scaled_image = self.original_image.resize((int(self.original_image.width * new_scale), int(self.original_image.height * new_scale)))

        # Convert the rescaled image to a format that tkinter can display
        scaled_photo = ImageTk.PhotoImage(Image.fromarray(scaled_image))
    def crop(self, image):
        image = np.asarray(image)
        #get face detection
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(
            min_detection_confidence=0.6) as face_detection:
            results = face_detection.process(image)
            if results.detections:
                for detection in results.detections:
                    #set bounding box size
                    box = detection.location_data.relative_bounding_box
                    center = detection.location_data.relative_keypoints[0]
                    print("center x = {center.x} center y = {center.y}")
                    print("box = {box}")
                    print(f"image height: {image.shape[0]} image width: {image.shape[1]} image shape: {image.shape}")
                    x = box.xmin
                    x = int(x * image.shape[1])
                    y = box.ymin
                    y = int(y * image.shape[0])
                    w = box.width
                    w = int(w * image.shape[1])
                    h = box.height
                    h = int(h * image.shape[0])
                    self.H = h
                    self.W = w
                    print(x,y,w,h)
                    try :
                        face = image[y:y+h, x:x+w]
                        #resize the image to 1000x1000
                        face = cv2.resize(face, (self.H, self.W))
                        
                    except:
                        print("error")
                        face = image

                    return face

        # Update the canvas item with the new image and position
# Create the main window
root = tk.Tk()

# Create the image display app
app = ImageDisplayApp(root)

# Run the main event loop
root.mainloop()
# C:\Users\joeli\OneDrive\Documents\GitHub\EncoderDecoder\Head_02_Albedo.png

