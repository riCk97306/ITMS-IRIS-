import cv2

# REPLACE THIS with your phone's IP address displayed in the app
# Ensure you keep the 'http://' and add '/video' at the end for IP Webcam app
url = "http://192.168.39.19:8080/video"

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if camera opened successfully
if not cap.isOpened():
    print("Could not open video stream. Check your IP and Wi-Fi connection.")
    exit()

print("Camera opened. Press 'q' to quit.")

while True:
    # Read a frame from the stream
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame (stream end?). Exiting ...")
        break

    # Optional: Resize the frame if it's too big
    frame = cv2.resize(frame, (960, 540))

    # Display the frame
    cv2.imshow('Phone Camera', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()