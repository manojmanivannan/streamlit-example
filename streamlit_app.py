import threading
from lobe import ImageModel
import cv2
import streamlit as st
from time import sleep

from streamlit_webrtc import webrtc_streamer

lock = threading.Lock()
img_container = {"img": None}
model = ImageModel.load('exported-model/model1/')

def video_frame_callback(frame):
    # img = frame.to_ndarray(format="bgr24")
    img = frame.to_image()
    with lock:
        img_container["img"] = img

    return frame


ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration={ "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

fig_place = st.empty()


while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ax.cla()
    # ax.hist(gray.ravel(), 256, [0, 256])
    # fig_place.pyplot(fig)

    # print(img)

    # OPTION 3: Predict from Pillow image
    result = model.predict(img)

    # Print top prediction
    # fig_place.write(result.prediction)
    

    # Print all classes
    sleep(0.1)
    for label, confidence in result.labels:
        fig_place.write(f'Prediction: {result.prediction}')
        # fig_place.write(f"{label}: {confidence*100}%")

    # Visualize the heatmap of the prediction on the image 
    # this shows where the model was looking to make its prediction.
    # heatmap = model.visualize(img)
    # heatmap.show()