import cv2
import threading
import playsound
import smtplib
import streamlit as st
import numpy as np

# Load the fire detection cascade model
fire_cascade = cv2.CascadeClassifier('fire_detection_cascade_model.xml')

# Set up email parameters from environment variables
import os
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
RECIPIENT_EMAIL = "ayushkallambkar01@gmail.com"

def play_alarm_sound_function():
    """Function to play alarm sound."""
    playsound.playsound('fire_alarm.mp3', True)
    print("Fire alarm end")

def send_mail_function():
    """Function to send email alert."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Sender email or password environment variable not set.")
        return
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        message = "Subject: Fire Alert\n\nWarning: Fire accident detected!"
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message)
        print(f"Alert mail sent successfully to {RECIPIENT_EMAIL}")
        server.close()
    except Exception as e:
        print(f"Error sending email: {e}")

runOnce = False

# Streamlit app
st.title("Fire Detection System")

# Initialize video capture
vid = cv2.VideoCapture(0)

frame_placeholder = st.empty()

while True:
    Alarm_Status = False
    ret, frame = vid.read()
    if not ret:
        st.error("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

    for (x, y, w, h) in fire:
        cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        st.write("Fire alarm initiated")
        threading.Thread(target=play_alarm_sound_function).start()

        if not runOnce:
            st.write("Mail send initiated")
            threading.Thread(target=send_mail_function).start()
            runOnce = True
        else:
            st.write("Mail is already sent once")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
