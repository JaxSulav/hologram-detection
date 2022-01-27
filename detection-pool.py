import cv2
import time
import logging
import argparse
import numpy as np
from multiprocessing import Pool


def write_video(output_video, fourcc, op_fps, op_size, op_array):
    print("Starting to write to video - detected.mp4")
    out = cv2.VideoWriter(output_video, fourcc, op_fps, op_size)
    print("op_array: " + str(op_array))

    for i in range(len(op_array)):
        out.write(op_array[i])
        logger.info("Frame: " + str(i))
    out.release()
    if detected == True:
        print("Hologram detected successfully")
    else:
        logger.warning("Cannot detect Hologram")

    inp_video = cv2.VideoCapture(video)
    op_video = cv2.VideoCapture("detected.mp4")

    while True:
        ret1, frame1 = inp_video.read()
        ret2, frame2 = op_video.read()
        try:
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))

        except:
            inp_video = cv2.VideoCapture(video)
            op_video = cv2.VideoCapture("detected.mp4")
            ret1, frame1 = inp_video.read()
            print(frame1)
            ret2, frame2 = op_video.read()
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))

        both = np.hstack((frame1, frame2))
        both = cv2.resize(both, (640, 480))

        cv2.imshow('output', both)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break


def frame_job(video):
    global resize_width, resize_height, frames_output
    output_video = 'detected.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    op_fps = 45
    op_size = (resize_width, resize_height)
    print("Detecting Hologram.............")
    print("Please wait, it may take some time......")

    cap = cv2.VideoCapture(video)
    currentFrame = 0
    for i in range(50):
    # while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        else:
            pass
        frames.append(frame)

    logger.info("Number of frames captured: " + str(len(frames)))

    bounds_job(frames) # fill corrected array with achieved perspective corrected and cropped images

    logger.info("Number of corrected frames: " + str(len(corrected)))

    del frames[:]
    for each in range(len(corrected)):
        current_stack.append(corrected[each])

        logger.info("length of current_stack: " + str(len(current_stack)))

        if len(current_stack) == 35:

            # print("indexed 7: " + str(current_stack[:7]))
            # print("length of indexed 7: " + str(len(current_stack[:7])))
            start, end = 0, 7
            for i in range(5):
                all_list = [list1, list2, list3, list4, list5]
                print("start: " + str(start))
                print("end: " + str(end))
                # print("akjnsdkjna sdn: " + str(i))
                all_list[i].append(current_stack[start:end])

                start += 7
                end += 7
                # del currentFrame[:7]


            # print("list 1 value: " + str(list1))
            # print("list 2 value: " + str(list2))
            # print("list 3 value: " + str(list3))
            # print("list 4 value: " + str(list4))
            # print("list 5 value: " + str(list5))
            p = Pool(5)
            p.map(holo_overlays, all_list)



            del list1[:]
            del list2[:]
            del list3[:]
            del list4[:]
            del list5[:]

            # p.join()
            # holo_overlays(current_stack)

            logger.info("Number of frames in frames_output: " + str(len(frames_output)))
            del current_stack[:]
            print("current_stack: " + str(current_stack) )

    del corrected[:]

    write_video(output_video, fourcc, op_fps, op_size, frames_output)


def points_order(pts):
    rect = np.zeros((4,  2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def transform(image, pts):
    rect = points_order(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")

    percTransform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, percTransform, (maxWidth, maxHeight))

    return warped


def bounds_job(img_array):
    global resize_width, resize_height
    for image in range(len(img_array)):
        try:
            img = img_array[image]
            img = cv2.resize(img, (resize_width, resize_height))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            kernel = np.ones((3,3), np.uint8)
            erosion = cv2.erode(blur, kernel)
            dilation = cv2.dilate(erosion, kernel)
            edge = cv2.Canny(dilation, 22, 25)

            # cv2.imshow('canny', edge)
            # cv2.waitKey(0)

            cnts, _ = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = max(cnts, key = cv2.contourArea)
            screenCnt = None
            peri = cv2.arcLength(cnts, True)
            approx = cv2.approxPolyDP(cnts, 0.015 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                # cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
                # cv2.imshow('contours', img)
                # cv2.waitKey(0)

                src = np.float32([screenCnt[0][0],screenCnt[1][0],screenCnt[2][0],screenCnt[3][0]])
                warped = transform(img.copy(), src)
                resized = cv2.resize(warped, (resize_width, resize_height))
                corrected.append(resized)
            else:
                logger.info("Skipped: No rectangle detected !!!")

        except:
            logger.warning("NO image !!!!!!")


def holo_overlays(rgb):
    global resize_width, resize_height, detected
    rgb = [image for lst in rgb for image in lst]
    # print("rgb value: " + str(rgb))
    # for lst in rgb:
    #     for image in lst:
    #         newlist.append(image)
    im_hsv = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for image in rgb]
    print("im_hsv :" + str(im_hsv))


    error_list = []

    for i in range(resize_width):
        for j in range(resize_height):
            hue_sum = 0
            for element in im_hsv:
                hue_sum += element[i][j][0]
            hue_mean = hue_sum / len(im_hsv)
            diff = 0
            for element in im_hsv:
                each_hue = element[i][j][0]
                diff += (each_hue - hue_mean) ** 2
            result = diff/(len(im_hsv) - 1)
            error = result ** (0.5)
            error_list.append(error)

    max_error = max(error_list)

    del im_hsv[:]
    for errors in range(len(error_list)):
        current_error = error_list[errors]
        percentage_value = (current_error * 100) / max_error
        substitute_value = (percentage_value / 100) * 255
        error_list[errors] = substitute_value

    for every_error in range(len(error_list)):
        error_list[every_error] = int(round(error_list[every_error]))

    for error in range(len(error_list)):
        if error_list[error] > 180:
            detected = True
            break
        else:
            pass

    reshaper = np.array(error_list, np.uint8)
    del error_list[:]
    error_array = reshaper.reshape(resize_width, resize_height)

    _, threshold = cv2.threshold(error_array, 180, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(error_array,(5,5), 0)
    _, otsu = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshold_array = cv2.adaptiveThreshold(error_array , 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 35)


    error_array = [] # stacking up every errors in error array was causing lag in program
    rgb_thresh = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    rgb_otsu = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

    op_height = rgb_otsu.shape[0]
    op_width = rgb_otsu.shape[1]

    for y in range(0, op_height):
        for x in range(0, op_width):
            if rgb_otsu[y, x, 0] == 255:
                rgb_otsu[y, x, 0] = 0
                rgb_otsu[y, x, 2] = 0

    for i in range(len(rgb)):
        overlayed_img = cv2.addWeighted(rgb_otsu, 1, rgb[i], 1, 0)
        overlayed_img = cv2.resize(overlayed_img, (resize_width, resize_height))
        print("overlayed_img: " + str(overlayed_img))
        frames_output.append(overlayed_img)
        print("frames_output: " + str(frames_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video")
    parser.add_argument("--visible", action="store_true")
    args = parser.parse_args()

    start_ts = time.time()
    logFile = "./hologram " + str(start_ts) + ".log"
    logger = logging.getLogger('hologram')
    hdlr = logging.FileHandler(logFile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    detected = False
    frames = []
    corrected = []
    current_stack = []
    frames_output = []
    list1, list2, list3, list4, list5 = [], [], [], [], []
    all_list = [list1, list2, list3, list4, list5]
    video = str(args.video)
    logger.info("Video name: " + video)
    resize_width = 300
    resize_height = 300

    frame_job(video)

    end_ts = time.time()
    logger.info("Time taken: " + str(end_ts - start_ts) + " seconds")
