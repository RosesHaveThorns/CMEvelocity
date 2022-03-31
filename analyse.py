import glob
import math
from matplotlib.pyplot import contour, draw
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
from datetime import datetime
import matplotlib.dates as mdates

dayfolder = "20170910_3"
imgwidth = 32 # in solar radii
solradii_to_km = 695700 # in km

def getImages(addr, type="png"):
    imgs = []
    names = []

    for file in glob.glob(addr + "/" + dayfolder + "/*." + type):
        names.append(file)
        imgs.append(cv2.imread(file))
        print("Read image: " + file)

    return imgs, names

def toGrayscale(imgs):
    grays = []
    for i in range(len(imgs)):
        print("To Grayscale: image " + str(i))
        grays.append(cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY))

    return grays

def writeToVideo(imgs, times, addr, name="output"):
    
    if len(imgs) < 1:
        print("Cant write to video, no images in array")
        return

    if len(imgs[0].shape) == 3 and imgs[0].shape[2] == 3: # ie rgb or similar
        height, width, layers = imgs[0].shape
    elif len(imgs[0].shape) == 2: # grayscale, so convert to rgb
        rgb = []
        for i in range(len(imgs)):
            rgb.append(cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR))
            
        imgs = rgb
        height, width, layers = imgs[0].shape
    else:
        print("Error reading image size, unexpected size of .shape return value")
        return

    size = (width,height)

    vid = cv2.VideoWriter(addr + "/" + dayfolder + "/output/" + name + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), 5, size)
 
    for i in range(len(imgs)):
        print("Wrote to video: image " + str(i))
        cv2.imwrite(addr + "/" + dayfolder + "/output/" + times[i].strftime("%H%M_%S") + ".png", imgs[i])
        print(addr + "/" + dayfolder + "/output/" + times[i].strftime("%H%M_%S") + ".png")
        vid.write(imgs[i])
        
    vid.release()

    print("Wrote out to video: " + name + ".avi")

def threshold(imgs, min=127, max=255):
    thresh_imgs = []
    for i in range(len(imgs)):
        print("Threshold: image " + str(i))
        ret, thresh = cv2.threshold(imgs[i], min, max, 0)
        thresh_imgs.append(thresh)

    return thresh_imgs

def denoise(imgs, t="median", n=5): 
    result = []
    for i in range(len(imgs)):
        print(t + "denoise: image " + str(i))

        if t == "nlmeans":
            res = cv2.fastNlMeansDenoising(imgs[i], n)
        elif t == "median":
            res = cv2.medianBlur(imgs[i], n)
        elif t == "gaussian":
            res = cv2.GaussianBlur(imgs[i], (n,n), 0)

        result.append(res)

    return result

def invert(imgs):
    result = []
    for i in range(len(imgs)):
        print("Cropped: image " + str(i))
        print("Inverted: image " + str(i))
        res = cv2.bitwise_not(imgs[i])
        result.append(res)

    return result

def blob_detect(imgs):

    result = []
    for i in range(len(imgs)):
        print("Blob detect: image " + str(i))

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create() 

        # Detect blobs
        keypoints = detector.detect(imgs[i])

        # Draw detected blobs as red circles
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        img_with_keypoints = cv2.drawKeypoints(imgs[i], keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        result.append(img_with_keypoints)
    
    return result

def findContours(imgs, drawover):
    contours = []
    for i in range(len(imgs)):
        print("Contour detect: image " + str(i))

        cs,_ = cv2.findContours(imgs[i], mode=cv2.RETR_LIST,
                                method=cv2.CHAIN_APPROX_SIMPLE )

        
        contours.append(cs)

    return contours

def crop(imgs, t=0, b=1, l=0, r=1):
    result = []
    for i in range(len(imgs)):
        print("Crop: image " + str(i))
        result.append(imgs[i][t:-b, l:-r])
    return result

def largestCountour(imgs, cs):
    # find largest contour, this should be cme
    drawn = []
    cmes = []

    for i in range(len(imgs)):
        print("Find Largest Contour: image " + str(i))
        cmes.append(max(cs[i], key = cv2.contourArea))
        drawn.append(cv2.drawContours(imgs[i], [cmes[i]], -1, color=(0,0,255), thickness=-1))

    return drawn, cmes

def findFurthestFromCenter(imgs, cmes, times, centerAdjust=[0, 0]):
    drawnImgs = []
    furthestPoints = []

    if len(imgs[0].shape) == 3:
        h, w, _ = imgs[0].shape
    elif len(imgs[0].shape) == 2:
        h, w = imgs[0].shape
    center = (round(w/2) + centerAdjust[0], round(h/2) + centerAdjust[1])

    for i in range(len(imgs)):
        # draw centre point
        drawn = cv2.line(imgs[i], (center[0]-10, center[1]), (center[0]+10, center[1]), (0, 255, 0), 2)
        drawn = cv2.line(drawn, (center[0], center[1]-10), (center[0], center[1]+10), (0, 255, 0), 2)

        cme = cmes[i]
        furthestPoints.append([0,0,0]) # (x, y, dist)

        # find furthest point from center
        for p in range(len(cme)):
            dist = math.sqrt(math.pow(cme[p][0][0] - center[0], 2) + math.pow(cme[p][0][1] - center[1], 2)) # pythagoras

            if furthestPoints[i][2] < dist: # new furthest point
                furthestPoints[i][2] = dist
                furthestPoints[i][1] = cme[p][0][1]
                furthestPoints[i][0] = cme[p][0][0]
        
        # draw furthest point
        drawn = cv2.line(drawn, (furthestPoints[i][0]-10, furthestPoints[i][1]), (furthestPoints[i][0]+10, furthestPoints[i][1]), (0, 255, 0), 2)
        drawn = cv2.line(drawn, (furthestPoints[i][0], furthestPoints[i][1]-10), (furthestPoints[i][0], furthestPoints[i][1]+10), (0, 255, 0), 2)

        drawn = cv2.putText(drawn, "dist: " + str(round(furthestPoints[i][2])) + "px", (5,30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)
        drawn = cv2.putText(drawn, times[i].strftime("%H:%M %Ss"), (5,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)

        drawnImgs.append(drawn)

        print("Furthest point in image " + str(i) + ": " + str(furthestPoints[i][2]) + "px")

    return drawnImgs, furthestPoints

def getTimes(fnames):
    times = []
    for i in range(len(fnames)):
        filename = fnames[i].split('\\')[-1]
        timearr = filename.split('_')[-3:-1]
        timestr = timearr[0] + ' ' + timearr[1]
        dt = datetime.strptime(timestr, '%Y%m%d %H%M%S')
        times.append(dt)
    return times

if __name__ == "__main__":  
    imgs, fnames = getImages("imgs")
    times = getTimes(fnames)

    imgs = crop(imgs, b=50)
    imgs_gray = toGrayscale(imgs)

    #imgs_inv = invert(imgs)

    denoised_med = denoise(imgs_gray, t="median", n=5)
    denoised_med = denoise(denoised_med, t="median", n=5)
    denoised = denoise(denoised_med, t="gaussian", n=5)
    denoised = denoise(denoised, t="gaussian", n=5)
    denoised = denoise(denoised, t="gaussian", n=5)

    thresholded = threshold(denoised, 200, 255)

    cs = findContours(thresholded, imgs)

    drawn, cmes = largestCountour(imgs, cs)

    final, furthest = findFurthestFromCenter(drawn, cmes, times, centerAdjust=[0, 10])

    #blobs = blob_detect(denoised)

    print("\nDATA:")
    for i in range(len(fnames)):
        print(str(fnames[i]) + ",", end="")
    print("\n")
    for i in range(len(furthest)):
        print(str(furthest[i][2]) + ",", end="")
    print("\n")

    writeToVideo(final, times, "imgs")

    # convert to km
    h, w, _ = imgs[0].shape
    pxwidth = (imgwidth * solradii_to_km) / w
    print(pxwidth)

    #x = np.linspace(0, len(furthest), len(furthest))
    d = []
    for i in range(len(furthest)):
        d.append(furthest[i][2] * pxwidth)

    # calculate velocity
    v = []
    vtimes = []
    for i in range(len(d)):
        i += 1
        if (i >= len(d)):
            break
        vtimes.append(times[i-1] + (times[i]-times[i-1])/2)
        vel = (d[i] - d[i-1]) / (times[i]-times[i-1]).total_seconds()
        v.append(vel)
    avg = np.average(v)
    print("Average Velocity: " + str(avg))

    # export to csv
    with open(dayfolder + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        writer.writerow(times)
        writer.writerow(d)
    
    # show plot
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Analaysis of the Kinematics of a CME occuring on ' + times[0].strftime('%Y/%m/%d'))
    timefmt = mdates.DateFormatter('%H:%M')

    axs[0, 0].plot(times, d)
    axs[0, 0].set_title('Distance of CME Front from Solar Centre')
    axs[0, 0].set_ylabel('Distance [km]')
    axs[0, 0].set_xlabel('Time [HH:MM]')
    axs[0, 0].xaxis.set_major_formatter(timefmt)

    axs[0, 1].plot(vtimes, v)
    axs[0, 1].set_title('CME Front Velocity')
    axs[0, 1].set_ylabel('Velocity [km/s]')
    axs[0, 1].set_xlabel('Time [HH:MM]')
    axs[0, 1].xaxis.set_major_formatter(timefmt)

    plt.show()
    