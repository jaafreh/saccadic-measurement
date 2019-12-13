"""
This code track the eye movement and display a marker(a white circle in a black screen) to measure the
saccadic parameters gaze velocity, time latency of right eye ,time latency of left eye,
eye angle from center to most left or right location in degree, Amplitude gain
if face the error below make sure that webcam see your pupil ina right manner
NameError: name 'xR' is not defined
[ WARN:0] global D:\Build\OpenCV\opencv-4.1.2\modules\videoio\src\cap_msmf.cpp (674) SourceReaderCB::~SourceReaderCB terminating async callback
"""
import random
import time
from statistics import mean
import tkinter as tk
import cv2
import dlib
import numpy as np

from gaze_tracking import GazeTracking
import pygame
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqs
############################################################################################
Fl=650       # the focal length of the webcam, this value is calculated by focal_length.py            #
eye_dis=600   # distance of eye from the camera in mm                                                 #
timeTest=40   # the delay for the first screen of the web cam (to enable participant to see himself)  #
timeSamp= 25  # shift time of the white circle                                                        #
numSamp=800   # time of the whole test                                                                #
circelSize = 25  # the size of the white circle                                                       #
#############################################################################################
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
pygame.init()

infoObject = pygame.display.Info()
width = infoObject.current_w
height = infoObject.current_h
cv2.namedWindow('DD', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('DD', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cnt = 0
cntOscil = 0
mface =dlib.rectangle()
cntFace = 0
lPosCnt=0
rPosCnt=0

clock = pygame.time.Clock()
timeList = []
xcirList = [] # x cordination of the circle
ycirList = [] # y cordination of the circle
xReyeList = [] # x cordination of the right eye
yReyeList = [] # y cordination of the right eye
xLeyeList = [] # x cordination of the left eye
yLeyeList = [] # y cordination of the left eye
root = tk.Tk()
width_px = root.winfo_screenwidth() # width of the screen in pixel
width_mm = root.winfo_screenmmwidth() # width of the screen in mm
width_dpmm = width_px/width_mm   # density of the screen in pixel/mm
#print("width_px,width_mm,width_dpmm",width_px,width_mm,width_dpmm)

# smoothing of the data_set by moving average filter
def moving_average(data_set, periods):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

# function return the time shift between two sequences using cross correlation
def CalTimeShift(xcirList,xReyeList,fs):
    A = xcirList[100:]
    B = xReyeList[100:]
    A -= np.mean(A) # subtract the mean for the circle coordination
    A /= np.std(A) # devide by the stdDev for the circle coordination
    B -= np.mean(B) # subtract the mean for the right eye coordination
    B /= np.std(B) # devide by the stdDev for the right eye coordination
    B = -B  # flip the eye coordination since it is flipped by the camera
    bFilter= moving_average(np.asarray(B), 10) # smooth the signal
    xReyeListF=moving_average(np.asarray(xReyeList), 10)
   # plt.plot(xReyeListF)
   # plt.plot(xReyeList)
    #plt.show()
    #plt.plot(A)
    BB=np.array(bFilter)
    #plt.plot(BB)
    #plt.show()
    #################### save  xlocation of the circle and pupil to a file after normalization and
    fSignal = open('signal.txt', 'w') # open a file named signal.txt
    for i in range(len(BB)):
        fSignal.write("%f\t%f\n" % (BB[i], A[i]))

    fSignal.close()
    ##############################################################
    # zero_crossings where the gaze change sign
    zero_crossings=np.where(np.diff(np.signbit(BB)))[0]
    #print(" zero_crossings where the gaze change sign", zero_crossings)
    # v is the diffrence of A circle
    v = np.diff(A)
   # plt.plot(v)
#   #plt.plot(bFilter)
    #plt.show()
    mm=np.array(v)
    # find the index of the zero crossings
    zInedx= np.where(mm != 0)
    #print(" zIndex where the circle change location",zInedx)
    tmpList=[]
    # find the slope around the zero crossings values
    for i in range(len(zero_crossings)-1):
        F1,F2=zero_crossings[i],zero_crossings[i+1]
        #print("F1 and F2",F1,F2,zInedx[0])
        iCnt=0
        indx=None

        for j in range(len(zInedx[0])):
            if zInedx[0][j]> F1 and zInedx[0][j]<F2:
                iCnt+=1
                indx=j
            if iCnt>1:
                break
        if iCnt==1:
            tmpList.append(zInedx[0][indx])
    #print("tmplist ", tmpList)
    slope=0
    pulse=[]
    for i in range(len(tmpList)):
        #print("(xReyeListF[tmpList[i]]-xReyeListF[tmpList[i]+1])=",(xReyeListF[tmpList[i]]-xReyeListF[tmpList[i]+1]))
        slope+= abs(xReyeListF[tmpList[i]]-xReyeListF[tmpList[i]+1])
    # calculate the peak velocity
    gase_vel=slope/(len(tmpList))
    print("gase velocity =",gase_vel)
    # pupil movement distance
    pulse_avg_array=[]
    for i in range(len(tmpList)-1):
        pulse=[xReyeListF[tmpList[i]+3:tmpList[i+1]-3]]
        pulse_avg=np.average(pulse)
        pulse_avg_array.append(pulse_avg)
        #print("pulse everage is",pulse_avg)
    # distance of the pupil
    pupil_dist=max(pulse_avg_array)-min(pulse_avg_array)

   # plt.ylabel('Normalized Signal value ')
   # plt.xlabel('Sample Numbers')
   # plt.show()
    # Find cross-correlation between the circel and eye time series
    xcorr = np.correlate(A, B, "full")
    # delta time array to match xcorr
    nsamples = len(A)
    dt = np.arange(1 - nsamples, nsamples)
    time_shift = dt[xcorr.argmax()]
    time_del = abs(time_shift * samptimeavg)
    return time_del,pupil_dist

while True:
    # get a new frame from the webcam
    _, frame = webcam.read()
    # frame2 = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gaze.refresh(frame,cntFace,mface)
    mface=gaze.getFace()
    cntFace+=1

    frame = gaze.annotated_frame()
    text = ""
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    if cnt<timeTest: # cnt represent the delay for the first screen of the web cam
        cv2.imshow('DD', frame)
        cnt += 1
    else: # display the screen for the white circle movement
        done = False
        screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
        #screen = pygame.display.set_mode((width, height))

        cntFace = 0
        while not done and cntOscil < numSamp:

            for event in pygame.event.get():  # User did something
               if event.type == pygame.QUIT:  # If user clicked close
                    done = True  # Flag that we are done so we exit this loop

            # Set the screen background
            screen.fill(BLACK)
            # Draw circle
            xCen = int(width / 2)
            yCen = int(height / 2)
            xShift = int(width / 2 - width / 15)
            #yShift = int(height / 2 - height / 20)
           # xAxis = xCen
            #yAxis = yCen
            rTimePupils = pygame.time.get_ticks()
            #if rTimePupils % 2000<=1000:
            if cntOscil < timeSamp:
                xAxis = xCen
                yAxis = yCen
            #elif rTimePupils % 2000<=1000:
            elif cntOscil % timeSamp == 0:
                randloc = random.randint(1,3) # = random.randint(0,3) to have three different positions

                if randloc==1:
                    if rPosCnt<3:
                        xAxis = xCen + xShift
                        rPosCnt = rPosCnt+1
                        lPosCnt=0
                    else:
                        xAxis = xCen - xShift
                #  yAxis = yCen
                elif randloc==2:
                    if lPosCnt<3:
                        xAxis = xCen - xShift
                        lPosCnt= lPosCnt+1
                        rPosCnt=0
                    else:
                        xAxis = xCen + xShift

            pygame.draw.circle(screen, WHITE, (xAxis, yAxis), circelSize)
            # get a new frame from the webcam
            _, frame = webcam.read()
            gaze.refresh(frame,cntFace,mface)
            mface=gaze.getFace()
            cntFace += 1
            # read the coordinate of the pupils
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            if right_pupil:
                xR,yR = right_pupil
            if left_pupil:
                xL,yL = left_pupil

            rTimeCycle = pygame.time.get_ticks()
            timeList.append(rTimeCycle)
            xcirList.append(xAxis)
            ycirList.append(yAxis)
            xReyeList.append(xR)
            yReyeList.append(yR)
            xLeyeList.append(xL)
            yLeyeList.append(yL)

            cntOscil += 1  # cycle counter
            pygame.display.flip()
            if cv2.waitKey(1) == 27:
                break

        sampleTime=[]
        # write the time and the coordinate of the pupils to file
        fRes = open('Result.txt', 'w')
        for i in range(len(timeList)):
            fRes.write("%d\t%d\t%d\t%d\t%d\t%d\t%d\n" % (timeList[i], xcirList[i], ycirList[i], xReyeList[i], yReyeList[i],xLeyeList[i], yLeyeList[i]))
            if i > 3:
                sampleTime.append(timeList[i]-timeList[i-1])
        fRes.close()
        samptimeavg=sum(sampleTime) / (len(sampleTime) - 3)

        # regularize datasets by subtracting mean and dividing by s.d.
        time_del_R_X,eye_pxl=CalTimeShift(xcirList, xReyeList,int(samptimeavg))
        time_del_L_X,eye_pxl = CalTimeShift(xcirList, xLeyeList,int(samptimeavg))
        #print("Average sampling time ",int(samptimeavg),"\nTime latency Right x axis",int(time_del_R_X))
        print("Time latency of right eye ", int(time_del_R_X))
        print("Time latency of left eye ", int(time_del_L_X))
        #print("circle distance rom center to most left or right location in mm=", (xShift / width_dpmm))
        circle_angle= np.arctan((xShift / width_dpmm) / eye_dis)
        circle_angle_degree= 180 * circle_angle / np.pi
        #print("circle angle from center to most left or right location in degree=",circle_angle_degree )
        #print("distance of eye pupil in pixel", int(eye_pxl))
        eye_dis_mm= (eye_pxl*eye_dis)/Fl
        #print("distance of eye pupil in mm from center to right or left", eye_dis_mm/2)
        eye_angle= np.arctan((eye_dis_mm / 24) )
        eye_angle_degree= 180 * eye_angle / (np.pi)
        print("eye angle from center to most left or right location in degree=",eye_angle_degree )
        print("the radius in pixel of the white circle is =", circelSize)
        print("the radius in mm of the white circle is =",(circelSize/ width_dpmm))
        ############################################################
        Gain=eye_angle_degree/circle_angle_degree
        print("Amplitude gain=",Gain )
        if int(time_del_R_X)>100 and int(time_del_R_X)<350:
            print("latency range is normal")
            if Gain>1.1 and Gain< 1.1:
                print("Cognitive Deficit")
            else:
                print("Gain is normal")
        else:
            print("latency range is abnormal")
            print("Cognitive Deficit")

        pygame.quit()

        break
    if cv2.waitKey(1) == 27:
        break
        cv2.destroyAllWindows()
