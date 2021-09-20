import cv2
import imutils
import easyocr
import numpy as np
import os


def show_img(img):
    '''
    Takes in an image and shows it to the user.
    '''
    return cv2.imshow('Image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def stage_path(input_path: str):
    '''
    Takes a path and checks if it exists. If not, returns true and creates the dir.
    Otherwise, return false and do nothing.
    '''
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        return True
    else:
        return False


def recognise_img_plate(img_path):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(),
                                 cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours,
                      key=cv2.contourArea,
                      reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour,
                                  10,
                                  True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    try:
        new_image = cv2.drawContours(mask,
                                     [location],
                                     0,
                                     255,
                                     -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        return result, cropped_image
    except cv2.error:
        return None, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def vid_to_frames(vid_path: str, destination_path='frames', frame_skip=1000):
    '''
    Takes in the path to a video file, and the name of the folder in the current dir where it the frames should be saved.
    Breaks videos into frames.
    '''
    vid_name = vid_path.split('/')[-1].split('.')[0]  # get the video name
    vid_folder_path = os.path.join(os.getcwd(), destination_path, vid_name)

    stage_path(vid_folder_path)

    vidcap = cv2.VideoCapture(vid_path)
    count = 0
    success, image = vidcap.read()  # read the first video frame

    while success:  # if there are further frames to be read, keep looping
        cv2.imwrite(f'{vid_folder_path}/{vid_name}{count}.png', image)  # save the frame
        success, image = vidcap.read()  # read the next frame & increment the counter
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * frame_skip))  # skip a certain amount of frames
        count += 1

    return vid_folder_path


def recognise_vid_plates(vid_path: str, destination_path='frames', frame_skip=1000):
    frames_path = vid_to_frames(vid_path, destination_path, frame_skip)
    frames = [f for f in os.listdir(frames_path)]
    results = {}
    for frame in frames:
        results[frame] = recognise_img_plate(os.path.join(frames_path, frame))[0]

    return results


if __name__ == '__main__':
    print(recognise_vid_plates("C:/Users/Michal/Desktop/license2.mp4"))
    #show_img(recognise_img_plate('C:/Users/Michal/PycharmProjects/pythonProject1/frames/license/license263.png')[1])
    #cv2.waitKey(0)