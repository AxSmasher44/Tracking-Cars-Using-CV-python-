import cv2

#our image
img_file = "car_image2.jpg"
video = cv2.VideoCapture("Tesla_dashcam_accident.mp4")

#pre trained car classifier
classifier_file = "car_detector.xml"


#Create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)




#run forever until car stops or something
while True:
    #Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    
    #Drawing rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    #display the images
    cv2.imshow("car detector", frame)

    #don't autoclose, wait until 1ms
    cv2.waitKey(1)




"""#creating opencv image
img = cv2.imread(img_file)

#converting to grayscale(to make algorithm faster)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#Create classifier
car_tracker = cv2.CascadeClassifier(classifier_file)



#detect cars
cars = car_tracker.detectMultiScale(black_n_white)


#Drawing rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


#display the images
cv2.imshow("car detector", img)

#don't autoclose, wait until key-press
cv2.waitKey()
"""

print("Code Completed")