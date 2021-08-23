# Import thư viện
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os



# Định nghĩa màu và font chữ
font_scale = 1.1
thicknessRect = 5
thicknessText = 2
font = cv2.FONT_HERSHEY_SIMPLEX
colorText = (250, 0, 246)

# Load dữ liệu

try:
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eyeDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except Exception:
    st.write("Error loading cascade classifiers!")

# Nhận dạng ảnh
def FAERForImage(image):
    image = np.array(image.convert("RGB"))
    face = 0
    eyeL = 0
    eyeR = 0
    try:
        faces = faceDetect.detectMultiScale(image, 1.3, 5)
        for (x,y,w,h) in faces:
            roi = image[y:y+h, x:x+w]
            face += 1
            try:
                eyes = eyeDetect.detectMultiScale(roi)
                for (xe,ye,we,he) in eyes:
                    if ye < h/3 and xe < w/2:
                        eyeL += 1
                        cv2.rectangle(roi,(xe,ye),(xe+we,ye+he),(255,0,0),2)
                    if ye < h/3 and xe > w/2:
                        eyeR += 1
                        cv2.rectangle(roi,(xe,ye),(xe+we,ye+he),(0,0,255),2)
            except Exception:
                pass
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        return image, face, eyeL, eyeR
    except Exception:
        return "Error loading image!"


# Webcam & Video
def FAERForWebcam():
    face = 0
    eyeL = 0
    eyeR = 0
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened() == None:
        st.write("Error loading webcam!")
        return
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Not connection!")
            break
        frame = cv2.resize(frame, (800,800))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        try:
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                try:
                    eyes = eyeDetect.detectMultiScale(roi)
                    for (xe,ye,we,he) in eyes:
                        if ye < h/3 and xe < w/2:
                            cv2.rectangle(roi,(xe,ye),(xe+we,ye+he),(255,0,0),2)
                            cv2.putText(roi,"eL",(xe+20,ye+20),font,font_scale-0.5,colorText,thicknessText,cv2.LINE_AA)
                        if ye < h/3 and xe > w/2:
                            cv2.rectangle(roi,(xe,ye),(xe+we,ye+he),(0,0,255),2)
                            cv2.putText(roi,"eR",(xe+20,ye+20),font,font_scale-0.5,colorText,thicknessText,cv2.LINE_AA)
                except Exception:
                    pass
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,"face",(x+50,y+50),font,font_scale-0.3,colorText,thicknessText,cv2.LINE_AA)
            # Hiển thị 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.waitKey(10)
            FRAME_WINDOW.image(frame)
        except Exception:
            return "Error loading webcam!"
    else:
        st.write("stopped!")

# Hàm main

def main():
    st.title("Face Detection App :sunglasses: ")
    st.write("**Using the Haar cascade Classifiers**")
    activities = ["Image Detect", "Video Stream" ,"Author"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    # menu
    if choice == "Image Detect":
        image_file = st.file_uploader("Upload File", type=['jpeg', 'png', 'jpg', 'webp'])
        if image_file is not None and st.button("Run"):
            image = Image.open(image_file)
            result = FAERForImage(image)
            if result != "error":
                st.image(result[0], use_column_width = True)
                st.success("Found {} faces\n".format(result[1]))
                st.success("Found {} eyeLeft\n".format(result[2]))
                st.success("Found {} eyeRight\n".format(result[3]))
            else:
                st.write(result)
    elif choice == "Video Stream":
        FAERForWebcam()
    else:
        st.write("**Author: Vũ Ngọc Minh**")
        st.write("Tech: Python, OpenCV, Haar Cascade Classifiers, Streamlit")
        st.image("author.jpg")
        st.write("Github: https://github.com/minhvu0612")
        st.write("Facebook: https://m.facebook.com/nm.vu.3")



main()