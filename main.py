from flask import Flask, render_template, Response

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from multiprocessing import Process

app = Flask(__name__)
# variable for imported libraries
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# constant variables
offset = 20
imgSize = 300
counter = 0

#timer for 8 seconds
t_end = time.time() + 8

#aspect ratio
aspctRATIO1 = 0
aspctRATIO2 = 0
aspctRATIO_label = ['W>H', 'H>W']

# hand signal labels
labels = ['Closed Fist', 'Four Finger', 'Open Palm (Point)', 'Open Palm (FHV)','Open Palm (BHV)', 'Point Finger', 'Side Open Palm (H)', 'Thumb Up', 'Two Finger','Open Palm (FHH)', 'Open Palm (BHH)', 'Side Open Palm (V)']
ismoving = ['Moving', 'Static']

lastSignal = ''
finalSignal = ''

# counteron true/false
countingOn = False

#hand movement variables
moving_index1 = 1
moving_index2 = 1

currHP1 = [0, 0]
prevHP1 = [0, 0]

currHP2 = [0, 0]
prevHP2 = [0, 0]

all_signals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#Divide & Conquer Algorithm for array search
def find_max(arr, low, high):
    if low == high:
        return arr[low]

    mid = (low + high) // 2

    left_max = find_max(arr, low, mid)
    right_max = find_max(arr, mid + 1, high)

    return max(left_max, right_max)

def check_signal():
    signal_label = ['Ball In', 'Outside','Service', 'Ball Touch', 'Double Contact', 'Four Hits', 'Crossing', 'Touch Net', 'Beyond The Net', 'Catch', 'Dead Ball', 'Change Court', 'Time Out', 'Substitution', 'End of Set/Match']
    given_signal = all_signals.index(find_max(all_signals, 0, len(all_signals) - 1))
    return signal_label[given_signal]

def clear_signals():
    for i in range(len(all_signals)):
        all_signals[i] = 0

def startcheck():
    global countingOn
    countingOn = True
    clear_signals()
    time.sleep(8)
    countingOn = False
    return check_signal()

def checkerist():
    global finalSignal
    print('Checking Started')
    finalSignal = startcheck()
    clear_signals()
    print('Checking Done')

def generate_frames():
        while True:
        #access globals
            global moving_index1
            global moving_index2
            global currHP1
            global prevHP1
            global currHP2
            global prevHP2
            global countingOn
            global lastSignal
            global finalSignal

            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            #hand gesture code

            if len(hands) == 1:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                middleFinger = (hand['lmList'])[12]
                wrist = (hand['lmList'])[0]
                thumb = (hand['lmList'])[4]

                imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                # checks hand coordinates
                center1 = hand['center']
                currHP1 = center1
                #print(f'{x}, {y}')
                #print(center1)

                # check if left/right
                handType = hand['type']
                #print(f'Hand type: {handType}')

                axisx_hand = 0
                axisy_hand = 0


                # checks distance change of previous hand position to current hand position
                xdistance = (currHP1[0] - prevHP1[0])
                ydistance = (currHP1[1] - prevHP1[1])
                distancechange1 = {xdistance, ydistance}

                # distance change, helps making the change in position more
                dc = 8

                if xdistance > dc or xdistance < -dc or ydistance > dc or ydistance < -dc:
                    #print('Hand Movement: Moving')
                    moving_index1 = 0

                elif xdistance < dc or xdistance > -dc or ydistance < dc or ydistance > -dc:
                    #print('Hand Movement: Static')
                    moving_index1 = 1

                if aspectRatio > 1:
                    try:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize1 = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize1.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite1[:, wGap:wCal + wGap] = imgResize1
                        prediction, index1 = classifier.getPrediction(imgWhite1)
                        score1 = find_max(prediction, 0, len(prediction) - 1)
                        aspctRATIO1 = 1
                    except:
                        print('Hand Exceeding')

                else:
                    try:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize1 = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize1.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite1[hGap:hCal + hGap, :] = imgResize1
                        prediction, index1 = classifier.getPrediction(imgWhite1)
                        score1 = find_max(prediction, 0, len(prediction) - 1)
                        aspctRATIO1 = 0

                    except:
                        print('Hand Exceeding')

                # beyond the net
                try:
                    if index1 == 6 and moving_index1 == 1 and aspctRATIO1 == 0:
                        print('Signal: Beyond The Net')
                        all_signals[8] += 1
                        lastSignal = 'Beyond The Net'

                    # change court
                    if index1 == 0 and moving_index1 == 0 and xdistance > ydistance:
                        print('Signal: Change Court')
                        all_signals[11] += 1
                        lastSignal = 'Change Court'

                    # service (to change: detect yung movement na dapat mostly yung horizontal movement > vertical movement)
                    if index1 == (9 or 10) and moving_index1 == 0 and xdistance > ydistance:
                        print('Signal: Service')
                        all_signals[2] += 1
                        lastSignal = 'Service'

                    # double contact
                    if index1 == 8 and moving_index1 == 1:
                        print('Signal: Double Contact')
                        all_signals[4] += 1
                        lastSignal = 'Double contact'

                    # four hits
                    if index1 == 1 and moving_index1 == 1 and aspctRATIO1 == 1:
                        print('Signal: Four Hits')
                        all_signals[5] += 1
                        lastSignal = 'Four Hits'

                    if index1 == 5 and moving_index1 == 1:
                        print('Signal: Crossing')
                        all_signals[6] += 1
                        lastSignal = 'Crossing'

                    if index1 == 2 and moving_index1 == 1:
                        print('Signal: Touch Net')
                        all_signals[7] += 1
                        lastSignal = 'Touch Net'

                    if index1 == 4 and ydistance > xdistance and moving_index1 == 0:
                        print('Signal: Catch')
                        all_signals[9] += 1
                        lastSignal = 'Catch'

                    if (wrist[1] < middleFinger[1] - 70) and moving_index1 == 1:
                        print('Signal: Ball In')
                        all_signals[0] += 1
                        lastSignal = 'Ball In'

                except:
                    print('Hand Index')

                try:
                    cv2.putText(img, f'Confidence Score Hand 1: {labels[index1]} {round(score1 * 100, 2)}', (100, 430), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 51), 1)
                except:
                    pass

                prevHP1 = currHP1

            elif len(hands)==2:
                hand1 = hands[0]
                hand2 = hands[1]
                x1, y1, w1, h1 = hand1['bbox']
                x2, y2, w2, h2 = hand2['bbox']

                imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8)*255
                imgCrop1 = img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]

                imgWhite2 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop2 = img[y2 - offset:y2 + h2 + offset, x2 - offset:x2 + w2 + offset]

                imgCropShape1 = imgCrop1.shape
                imgCropShape2 = imgCrop2.shape

                aspectRatio1 = h1/w1
                aspectRatio2 = h2/w2

                #checks hand coordinates
                center1 = hand1['center']
                currHP1 = center1
                #print(currHP1)

                center2 = hand2['center']
                currHP2 = center2
                #print(currHP2)

                #check if left/right
                handType1 = hand1['type']
                handType2 = hand2['type']
                #print(f'Hand type 1: {handType1}, Hand type 2: {handType2}')




                # checks distance change of previous hand position to current hand position
                x1distance = (currHP1[0]-prevHP1[0])
                y1distance = (currHP1[1]-prevHP1[1])
                x2distance = (currHP2[0] - prevHP2[0])
                y2distance = (currHP2[1] - prevHP2[1])

                distancechange2 = {x1distance, y1distance, x2distance, y2distance}
                #print(distancechange2)

                # distance change, helps making the change in position more
                dc = 8

                if x1distance > dc or x1distance < -dc or y1distance > dc or y1distance < -dc:
                    moving_index1 = 0

                elif x1distance < dc or x1distance > -dc or y1distance < dc or y1distance > -dc:
                    moving_index1 = 1

                if x2distance > dc or x2distance < -dc or y2distance > dc or y2distance < -dc:
                    moving_index2 = 0

                elif x2distance < dc or x2distance > -dc or y2distance < dc or y2distance > -dc:
                    moving_index2 = 1

                if aspectRatio1 >1:
                    try:
                        k = imgSize/h1
                        w1Cal = math.ceil(k*w1)
                        imgResize = cv2.resize(imgCrop1, (w1Cal, imgSize))
                        imgResizeShape = imgResize.shape
                        w1Gap = math.ceil((imgSize-w1Cal)/2)
                        imgWhite1[:, w1Gap:w1Cal+w1Gap] = imgResize
                        prediction1, index1 = classifier.getPrediction(imgWhite1)
                        score1 = find_max(prediction1, 0, len(prediction1) - 1)
                        aspctRATIO1 = 1
                        #print(f'Hand 1:{labels[index1]}')

                    except:
                        print('Hand Exceeding Border')

                else:
                    try:
                        k = imgSize/w1
                        h1Cal = math.ceil(k*h1)
                        imgResize1 = cv2.resize(imgCrop1, (imgSize, h1Cal))
                        imgResizeShape = imgResize1.shape
                        h1Gap = math.ceil((imgSize-h1Cal)/2)
                        imgWhite1[h1Gap:h1Cal+h1Gap, :] = imgResize1
                        prediction1, index1 = classifier.getPrediction(imgWhite1)
                        aspctRATIO1 = 0
                        score1 = find_max(prediction1, 0, len(prediction1) - 1)
                        #print(f'Hand 1: {labels[index1]}')

                    except:
                        print('Hand Exceeding Border')

                if aspectRatio2 >1:
                    try:
                        k = imgSize/h2
                        w2Cal = math.ceil(k*w2)
                        imgResize2 = cv2.resize(imgCrop2, (w2Cal, imgSize))
                        imgResizeShape = imgResize2.shape
                        w2Gap = math.ceil((imgSize-w2Cal)/2)
                        imgWhite2[:, w2Gap:w2Cal+w2Gap] = imgResize2
                        prediction2, index2 = classifier.getPrediction(cv2.flip(imgWhite2, 0))
                        score2 = find_max(prediction2, 0, len(prediction2) - 1)
                        aspctRATIO2 = 1
                        #print(f'Hand 2: {labels[index2]}')

                    except:
                        print('Hand Exceeding Border')

                else:
                    try:
                        k = imgSize/w2
                        h2Cal = math.ceil(k*h2)
                        imgResize = cv2.resize(imgCrop2, (imgSize, h2Cal))
                        imgResizeShape = imgResize.shape
                        h2Gap = math.ceil((imgSize-h2Cal)/2)
                        imgWhite2[h2Gap:h2Cal+h2Gap, :] = imgResize
                        prediction2, index2 = classifier.getPrediction(cv2.flip(imgWhite2, 0))
                        score2 = find_max(prediction2, 0, len(prediction2) - 1)
                        aspctRATIO2 = 0
                        print(f'Hand 2: {labels[index2]}')

                    except:
                        print('Hand Exceeding border')


                try:
                    # dead ball

                    if index1 == 7 and moving_index1 == 1 and index2 == 7 and moving_index2 == 1:
                        print('Signal: Dead Ball')
                        all_signals[10] += 1
                        lastSignal = 'Dead Ball'

                    #ball touch
                    if index1 == 4 and index2 == 6 or index2 == 4 and index1 == 6:
                        print('Signal: Ball Touch')
                        all_signals[3] += 1
                        lastSignal = 'Ball Touch'

                    #outside
                    if index1 == 4 and index2 == 4 and moving_index1 == 1 and moving_index2 == 1:
                        print('Signal: Outside')
                        all_signals[1] += 1
                        lastSignal = 'Outside'

                    if index1 == 11 and index2 == 6 or index2 == 11 and index1 == 6:
                        print('Signal: Time Out')
                        all_signals[12] += 1
                        lastSignal = 'Time Out'
                    
                    if index1 == 0 and index2 == 0 and moving_index1 == 0 and moving_index2 == 0:
                        print('Signal: Substitution')
                        all_signals[13] += 1
                        lastSignal = 'Substitution'
                        
                    if index1 == (1 or 4) and index2 == (1 or 4) and ((center1[0] - center2[0]) < 180 or ((center1[0] - center2[0])) > -180):
                        print('Signal: End of Set/Match ')
                        all_signals[14] += 1
                        lastSignal = 'End of Set/Match'

                    print('===='*10)

                except:
                    pass

                prevHP1 = currHP1
                prevHP2 = currHP2

                try:
                    cv2.putText(img, f'Confidence Score Hand 1: {labels[index1]} {round(score1 * 100, 2)}', (100, 430),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 51), 1)

                    cv2.putText(img, f'Confidence Score Hand 2: {labels[index2]} {round(score2 * 100, 2)}', (100, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 51), 1)
                except:
                    pass

                if countingOn == True:
                    cv2.putText(img, f'Recording', (450, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


            #final output - gesture detection
            cv2.putText(img, f'Last Signal: {lastSignal}', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (130, 255, 158), 2)
            # cv2.putText(img, f'Final Signal: {finalSignal}', (30, 90), cv2.FONT_HERSHEY_COMPLEX, 1,
            #             (130, 255, 158), 2)
            # cv2.imshow("Image", imgOutput)
            # cv2.imshow('Output', img)
            cv2.waitKey(1)

        # Convert the image to JPEG format
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record', methods=['POST'])
def recordStart():
    if request.method == 'POST':
        print('Start!')
        checkerist()
        
    return 'Recording Start'

if __name__ == "__main__":
    app.run(debug=True)
