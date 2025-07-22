import cv2 
import imutils 
import os

# Change to your video directory
os.chdir(r"G:\AI & ML\\Projects\\Object detection")

# Tracker dictionary with scale-adaptive options
TrDict = {
    'csrt': cv2.TrackerCSRT_create, # Best for scale adaptation
    'boosting': cv2.legacy.TrackerBoosting.create,
    'kcf': cv2.TrackerKCF_create,
    'mil': cv2.TrackerMIL_create,   
    'medianflow': cv2.legacy.TrackerMedianFlow.create,
    'tld': cv2.legacy.TrackerTLD.create,
}

Trcker_name = 'csrt'
tracker = TrDict[Trcker_name]()

print("step 1: loading the video to select the object")
# I added this step because camera takes time to launch and adjust it's resolution

# v = cv2.VideoCapture("WIN_20250722_23_10_43_Pro.mp4")
v = cv2.VideoCapture("vid1 (online-video-cutter.com)_2.mp4")

if not v.isOpened():
    print("Error: Cannot open video file!")
    exit()
    
#read the firstt frame
ret,frame = v.read()
if not ret:
    print("Error: Cannot read video frame!")
    v.release()
    exit()
    
frame = imutils.resize(frame, width=600)
cv2.imshow("Select Object from Video", frame)

#select the object from the video to track and store the initial bounding box
bounadary_box = cv2.selectROI("Select Object from Video", frame )
initial_box = bounadary_box 

#initialize tracker with the frame and selected box
tracker.init(frame, bounadary_box)

cv2.waitKey(0)
cv2.destroyAllWindows()
v.release()

if bounadary_box == (0, 0, 0, 0):
    print("No object Exiting")
    exit()
    
print("object selected: ", bounadary_box)
print("Step 2: Starting the tracking process...")

#Start the real-time tracking process
# cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path)
cap = cv2.VideoCapture("vid1 (online-video-cutter.com)_2.mp4")

if not cap.isOpened():
    print("Error: Cannot open video source!")
    exit()
    
print( " real-time tracker started")

while True:
    ret,frame = cap.read()
    
    if not ret:
        print("Error: Cannot read video frame!")
        break
    
    frame = imutils.resize(frame, width=600)
    success, box = tracker.update(frame)
    
    if success:
        (x, y, w, h) = [int(a) for a in box]
        
        # Validate box to prevent crazy results
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        
        # Only draw if box is reasonable size
        if w > 5 and h > 5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
            
            # Show scale change
            if initial_box:
                initial_area = initial_box[2] * initial_box[3]
                current_area = w * h
                scale_percent = (current_area / initial_area) * 100
                cv2.putText(frame, f'Scale: {scale_percent:.1f}%', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 0), 2)
                cv2.putText(frame, f'Size: {w}x{h}', (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 0), 2)
        else:
            cv2.putText(frame, 'Invalid tracking result', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'Tracking failed - Press R to reselect', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Real-time Tracking", frame)
    
    key = cv2.waitKey(30)
    if key == ord('q'):
        
        break
    elif key == ord('r'):
        # Reselect object from current webcam frame
        print("Reselecting object...")
        cv2.destroyWindow("Real-time Tracking")
        cv2.imshow('Reselect Object', frame)
        bounadary_box = cv2.selectROI("Reselect Object", frame)
        cv2.destroyWindow('Reselect Object')
        
        if bounadary_box != (0, 0, 0, 0):
            # Create new tracker and reinitialize
            tracker = TrDict[Trcker_name]()
            initial_box = bounadary_box
            tracker.init(frame, bounadary_box)
            print("Object reselected successfully!")

cv2.destroyAllWindows()
cap.release()
print("Tracking completed!")
