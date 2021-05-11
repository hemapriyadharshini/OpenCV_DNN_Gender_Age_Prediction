import cv2 #OpenCV Module
import math
import argparse #Argument Parser

def DetectFace(net, frame, framework="caffe", conf_threshold=0.8): #Detect Face and make frame using OpenCV
    frameOpencvDnn=frame.copy() #Copy image as frame
    frameHeight=frameOpencvDnn.shape[0] #Measure frame height
    frameWidth=frameOpencvDnn.shape[1] #Measure fram width
    #blob = cv2.dnn.blobFromImage(image, scalefactor, size, mean, swapRB=True, boolcrop=False)
    inputimageblob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (299, 299), [104, 117, 123], True, False) #Input image; Output: image in blob format

    net.setInput(inputimageblob) #Input image in blob
    detectface=net.forward() #Forward/pass blob to compute output of layer to detect face
    Cells=[] #Initiate cpu real time face detector
    for i in range(detectface.shape[2]): #Detect face in the given frame and loop over to draw frames/boxes around the detected face
        confidence=detectface[0,0,i,2] #Extract Prediction confidence  
        if confidence>conf_threshold: #If the extracted prediction confidence is greater than 80% threshold then
            conf = "{:.2f}%".format(confidence * 100) #Compute confidence in %
            x1=int(detectface[0,0,i,3]*frameWidth) # Compute X co-ordinate of the frame/box. Remeber it's a box/frame, so 2x,2y
            y1=int(detectface[0,0,i,4]*frameHeight) #Compute Y co-ordinate of the frame/box
            x2=int(detectface[0,0,i,5]*frameWidth) # Compute X co-ordinate of the frame/box
            y2=int(detectface[0,0,i,6]*frameHeight) # Compute X co-ordinate of the frame/box
            Cells.append([x1,y1,x2,y2]) #Add all 4 sides/co-ordinates of the box
            x = x1 - 10 if x1 - 10 > 10 else x1 + 10 #Adjust Text Display layout
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10 #Adjust Text Display layout
            #cv2.putText(image, text, XY co-ordinates, font, fontScale, RGB color, thickness, lineType)
            cv2.putText(frameOpencvDnn, ' Confidence Score: '+conf, (x,y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2,cv2.LINE_8) #Confidence Score output
            #cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8) #Draw Rectangle line over face 
    return frameOpencvDnn,Cells


parser=argparse.ArgumentParser() #Module to read commandline arguments
parser.add_argument('--image') #Parse image from Command line

args=parser.parse_args()
#Load age and gender Model
faceProto="opencv_face_detector.pbtxt" 
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']


faceNet=cv2.dnn.readNet(faceModel,faceProto) #Load face detection Network to Memory
ageNet=cv2.dnn.readNet(ageModel,ageProto) #Load age detection Network to Memory
genderNet=cv2.dnn.readNet(genderModel,genderProto) #Load gender detection Network to Memory

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read() #Read images/faces in a video
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,faceBoxes=DetectFace(faceNet,frame) #If no face detected, print message
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes: #face detector shape layout
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        #cv2.dnn.blobFromImage(image, scalefactor, size, mean, swapRB)
        imageblob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False) #Parse image as blob
        
        #Detect Gender
        genderNet.setInput(imageblob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
           
        #Detect Age
        ageNet.setInput(imageblob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
        
        #Print Output
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Output", resultImg)
