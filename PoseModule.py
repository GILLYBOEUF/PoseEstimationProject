import cv2
import mediapipe as mp
import time
import math
import utils
from utils import *
import numpy as np
import json
import os


class poseDetector():

    def __init__(self, holistic = True, mode=False, upBody=False, smooth=True,
                 detectionCon=False, trackCon=0.5, model_complexity=1):
        #set the default parameters to the ones of MadiaPipe pose object
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = model_complexity
        # initialize min angle
        # by default set to 
        self.min_angles = {'left': 180.0, 'right': 180}
        #create a variable to keep track of the frame
        self.count = 0

        self.mpDraw = mp.solutions.drawing_utils #variable to allow us to draw the results on the image
        self.mpDraw_styles = mp.solutions.drawing_styles
        if holistic:
            self.mpPose = mp.solutions.holistic
            self.pose = self.mpPose.Holistic ( smooth_landmarks = self.smooth,
                                                    min_tracking_confidence = self.trackCon,
                                                    model_complexity = self.model_complexity,
                                                    min_detection_confidence=0.5)
        else :
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon, self.model_complexity)
    
        self.pose_connections = self.mpPose.POSE_CONNECTIONS
        self.poselandmarks_list = []
        for idx, elt in enumerate(self.mpPose.PoseLandmark):
            lm_str = repr(elt).split('.')[1].split(':')[0]
            self.poselandmarks_list.append(lm_str)
            
    def findPose(self, img, draw=True):
        #recolor image to RGB
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert image into RGB fromat
        img.flags.writeable = False

        # make detection
        self.results = self.pose.process(imgRGB) #send image to the model for detection
        
        #extract landmarks
        try:
            pose_landmarks = self.results.pose_landmarks.landmark
            #print results of pose landmarks
            #print(self.results.pose_landmarks)
            #draw the landmark if the user want (True by default)
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, #draw the keypoints
                                           self.mpPose.POSE_CONNECTIONS) #fill up the connections
        except:
            #if we make no detection we just pass
            pass

        #return the image in BGR format
        return img

    def myfindPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark): #id is the number of the keypoint, refer to https://mediapipe.dev/images/mobile/pose_tracking_full_body_landmarks.png to see to what it corresponds
                #we only want to collect the hips/ knees/ ankles which correspond to kpts 23 to 28
                if id in range(23, 28+1): 
                    h, w, c = img.shape # getting image dimensions for scaling the x,y coordinates of the keypoints
                    #lm is the ratio of the image, to have the actual pixel value we need to multiply this value by the width w of the image
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * c)
                    self.lmList.append([id, cx, cy, cz, lm.visibility]) #change this so that instead of a list it returns a dict that can later be exported in a JSON file
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) #overlay with the detected point to check we are detecting properly
        return self.lmList
    
    def myfindPosition2(self, img, draw=True):

        #create dict to store the coord of the 6 keypoints we want (hips, knees, ankles)
        self.lmDict = {}
        if self.results.pose_landmarks:
            h, w, c = img.shape 
            # maybe add frame to keep track of the time? 
            # maybe myltiply by h, w, c to get the proper values since the pose_landmarks attribute 
            # of the solution output object provides the landmarks’ normalized x and y coordinates  
            self.lmDict = {'left_hip': { 
                                        'x': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HIP.value].x * w, 
                                        'y': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HIP.value].y * h,
                                        'z': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HIP.value].z * c,
                                        'visibility': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_HIP.value].visibility
                                    },
                         'right_hip': {
                                        'x': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP.value].x * w, 
                                        'y': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP.value].y * h,
                                        'z': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP.value].z * c,
                                        'visibility': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_HIP.value].visibility
                                    },
                         'left_knee': {
                                        'x': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_KNEE.value].x * w, 
                                        'y': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_KNEE.value].y * h,
                                        'z': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_KNEE.value].z * c,
                                        'visibility': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_KNEE.value].visibility
                                    },
                         'right_knee': {
                                        'x': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE.value].x * w, 
                                        'y': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE.value].y * h,
                                        'z': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE.value].z * c,
                                        'visibility': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_KNEE.value].visibility
                                    },
                         'left_ankle': { 
                                        'x': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ANKLE.value].x * w, 
                                        'y': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ANKLE.value].y * h,
                                        'z': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ANKLE.value].z * c,
                                        'visibility': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.LEFT_ANKLE.value].visibility
                                    },
                         'right_ankle': {
                                        'x': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE.value].x * w, 
                                        'y': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE.value].y * h,
                                        'z': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE.value].z * c,
                                        'visibility': self.results.pose_landmarks.landmark[self.mpPose.PoseLandmark.RIGHT_ANKLE.value].visibility
                                    }
                        }

            # if draw:
            #    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) #overlay with the detected point to check we are detecting properly

        return self.lmDict

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        #print(angle)

        # Draw
        if draw:
            if (p1 % 2) == 0: 
                col = (255, 0, 0) #color for the right side
            else:
                col = (0, 0, 255)
            cv2.line(img, (x1, y1), (x2, y2), col, 3)
            cv2.line(img, (x3, y3), (x2, y2), col, 3)
            cv2.circle(img, (x2, y2), 10, col, cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, col, 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, col, 2)
        return angle
    
    def findAngle2(self, img, draw=True):
        self.angles = {}
        if self.lmDict:
            # Get the landmarks
            h, w, c = img.shape
            l1 = [int(self.lmDict['left_hip']['x']), int(self.lmDict['left_hip']['y'])]
            l2 = [int(self.lmDict['left_knee']['x']), int(self.lmDict['left_knee']['y'])]
            l3 = [int(self.lmDict['left_ankle']['x']), int(self.lmDict['left_ankle']['y'])]

            r1 = [int(self.lmDict['right_hip']['x']), int(self.lmDict['right_hip']['y'])]
            r2 = [int(self.lmDict['right_knee']['x']), int(self.lmDict['right_knee']['y'])]
            r3 = [int(self.lmDict['right_ankle']['x']), int(self.lmDict['right_ankle']['y'])]

            # Calculate the Angle
            self.angles['left'] = compute_angle(l1, l2, l3)
            self.angles['right'] = compute_angle(r1, r2, r3)

            if draw: 
                # Draw left
                col_l = (255, 0, 0) 
                cv2.line(img, (l1[0], l1[1]), (l2[0], l2[1]), col_l, 3)
                cv2.line(img, (l3[0], l3[1]), (l2[0], l2[1]), col_l, 3)

                cv2.circle(img, (l1[0], l1[1]), 5, col_l, cv2.FILLED)
                cv2.circle(img, (l2[0], l2[1]), 10, col_l, cv2.FILLED)
                cv2.circle(img, (l2[0], l2[1]), 15, col_l, 2)
                cv2.circle(img, (l3[0], l3[1]), 5, col_l, cv2.FILLED)
                cv2.putText(img, str(int(self.angles['left'])), (l2[0] - 50, l2[1] + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, col_l, 2)
                
                # Draw right
                col_r = (0, 0, 255)
                cv2.line(img, (r1[0], r1[1]), (r2[0], r2[1]), col_r, 3)
                cv2.line(img, (r3[0], r3[1]), (r2[0], r2[1]), col_r, 3)
                cv2.circle(img, (r1[0], r1[1]), 5, col_r, cv2.FILLED)
                cv2.circle(img, (r2[0], r2[1]), 10, col_r, cv2.FILLED)
                cv2.circle(img, (r2[0], r2[1]), 15, col_r, 2)
                cv2.circle(img, (r3[0], r3[1]), 5, col_r, cv2.FILLED)
                cv2.putText(img, str(int(self.angles['right'])), (r2[0] - 50, r2[1] + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, col_r, 2)
                
                cv2.line(img, (l1[0], l1[1]), (r1[0], r1[1]), (255, 255, 255), 3) #draw pelvis line

        return self.angles
    
    def update_min_angle (self):
        self.min_angles = {'left': min(self.min_angles['left'], self.angles['left']),
                           'right': min(self.min_angles['right'], self.angles['right'])}       
    
    def findPosition3D (self, img):
        # Run MediaPipe Pose and plot 3d pose world landmarks.
        with self.mpPose.Pose(
            static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
            for name, image in img.items():
                self.results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print the real-world 3D coordinates of nose in meters with the origin at
                # the center between hips.
                print('Nose world landmark:'),
                print(self.results.pose_world_landmarks.landmark[self.mpPose.PoseLandmark.NOSE])
                
                # Plot pose world landmarks.
                self.mpDraw.plot_landmarks(
                    self.results.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)

    def mediapipe_to_openpose_json(self, img, filename):
        from google.protobuf.json_format import MessageToDict

        # Defining the defauly Openpose JSON structure
        # (template from openpose output)
        json_data = {
            "version": 1.3,
            "people": [
                {
                    "person_id"              : [filename],
                    "pose_keypoints_2d"      : [],
                    "face_keypoints_2d"      : [],
                    "hand_left_keypoints_2d" : [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d"      : [],
                    "face_keypoints_3d"      : [],
                    "hand_left_keypoints_3d" : [],
                    "hand_right_keypoints_3d": []
                }
            ]
        }
        # getting image dimensions for scaling the x,y coordinates of the keypoints
        h, w, c = img.shape
        self.count += 1        
        if self.results.pose_landmarks:
            
            landmarks = self.results.pose_world_landmarks.landmark

            tmp2d =[]
            onlyList2d = []
            list4json2d = []
            tmp3d =[]
            onlyList3d = []
            list4json3d = []

            # converting the landmarks to a list
            for idx, coords in enumerate(landmarks):
                coords_dict = MessageToDict(coords)
                qq2d = (coords_dict['x'], coords_dict['y'], coords_dict['visibility'])
                tmp2d.append(qq2d)
                qq3d = (coords_dict['x'], coords_dict['y'], coords_dict['z'], coords_dict['visibility'])
                tmp3d.append(qq3d)

            # Calculate the two additional joints for openpose and add them
            # NECK KPT
            tmp2d[1] = ( (tmp2d[12][0] - tmp2d[11][0]) / 2 + tmp2d[12][0], \
                (tmp2d[12][1] - tmp2d[11][1]) / 2 + tmp2d[12][1], \
                0.95 )
            # HIP_MID
            tmp2d[8] = ( (tmp2d[24][0] - tmp2d[23][0]) / 2 + tmp2d[24][0], \
                (tmp2d[24][1] - tmp2d[23][1]) / 2 + tmp2d[24][1], \
                    0.95 )
            # NECK KPT
            tmp3d[1] = ( (tmp3d[12][0] - tmp3d[11][0]) / 2 + tmp3d[12][0], \
                (tmp3d[12][1] - tmp3d[11][1]) / 2 + tmp3d[12][1], \
                (tmp3d[12][2] - tmp3d[11][2]) / 2 + tmp3d[12][2], \
                0.95 )
            # HIP_MID
            tmp3d[8] = ( (tmp3d[24][0] - tmp3d[23][0]) / 2 + tmp3d[24][0], \
                (tmp3d[24][1] - tmp3d[23][1]) / 2 + tmp3d[24][1], \
                (tmp3d[24][2] - tmp3d[23][2]) / 2 + tmp3d[24][2], \
                    0.95 )

            #  SSCALING the x, y and z coordinates with the resolution of the image to get px corrdinates
            for i in range(len(tmp2d)):
                tmp2d[i] = ( int(np.multiply(tmp2d[i][0], w)), \
                    int(np.multiply(tmp2d[i][1], h)), \
                    tmp2d[i][2])
                tmp3d[i] = ( int(np.multiply(tmp3d[i][0], w)), \
                    int(np.multiply(tmp3d[i][1], h)), \
                    int(np.multiply(tmp3d[i][2], c)), \
                    tmp3d[i][3])

            # Reordering list to comply to openpose format
            # For the order table,refer to the Notion page
            mp_to_op_reorder = [0, 0, 12, 14, 16, 11, 13, 15, 0, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7, 31, 31, 29, 32, 32, 30, 0, 0, 0, 0, 0, 0, 0, 0]
            onlyList2d = [tmp2d[i] for i in mp_to_op_reorder]
            onlyList3d = [tmp3d[i] for i in mp_to_op_reorder]

            # delete the last 8 elements to conform to OpenPose joint length of 25
            del onlyList2d[-8:]
            del onlyList3d[-8:]

            # OpenPose format requires only a list of all landmarkpoints. So converting to a simple list
            for nums in onlyList2d:
                for val in nums:
                    list4json2d.append(val)
            for nums in onlyList3d:
                for val in nums:
                    list4json3d.append(val)

            # Making the JSON openpose format and adding the data
            json_data = {
                            "version": 1.3,
                            "people": [
                                {
                                    "person_id"              : [filename],
                                    "pose_keypoints_2d"      : list4json2d,
                                    "face_keypoints_2d"      : [],
                                    "hand_left_keypoints_2d" : [],
                                    "hand_right_keypoints_2d": [],
                                    "pose_keypoints_3d"      : list4json3d,
                                    "face_keypoints_3d"      : [],
                                    "hand_left_keypoints_3d" : [],
                                    "hand_right_keypoints_3d": []
                                }
                            ]
                    }
            newpath = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/' + filename
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            os.chdir(newpath)
            json_filename = str(self.count) + filename + ".json"
            json_filename = json_filename.replace(".png","_keypoints")
            with open(json_filename, 'w') as fl:
                fl.write(json.dumps(json_data, indent=2, separators=(',', ': ')))

        else:
            # Making the JSON openpose format and adding the data
            json_data = {
                            "version": 1.3,
                            "people": [
                                {
                                    "person_id"              : [filename],
                                    "pose_keypoints_2d"      : [],
                                    "face_keypoints_2d"      : [],
                                    "hand_left_keypoints_2d" : [],
                                    "hand_right_keypoints_2d": [],
                                    "pose_keypoints_3d"      : [],
                                    "face_keypoints_3d"      : [],
                                    "hand_left_keypoints_3d" : [],
                                    "hand_right_keypoints_3d": []
                                }
                            ]
                    }
            newpath = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/json' + filename
            if not os.path.exists(newpath):
                print('create folder')
                os.makedirs(newpath)
                os.chdir(newpath)
            json_filename = str(self.count) + filename + ".json"
            json_filename = json_filename.replace(".png","_keypoints")
            with open(json_filename, 'w') as fl:
                fl.write(json.dumps(json_data, indent=2, separators=(',', ': ')))


        # REFERENCES:
        # 1. https://github.com/google/mediapipe/issues/1020
        # 2. https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/871
     
    def get_coords_detectron2 (self, coords_path, frame):
        data = np.load(coords_path, allow_pickle=True)
        coords = data['keypoints']
        # keypoints: ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
        # "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", 
        # "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

        self.lmDict = {  'left_hip': { 
                                        'x': coords[frame-1, 11, 0].astype(float), 
                                        'y': coords[frame-1, 11, 1].astype(float),
                                        'visibility': coords[frame-1, 11, 2]
                                    },
                         'right_hip': {
                                        'x': coords[frame-1, 12, 0].astype(float), 
                                        'y': coords[frame-1, 12, 1].astype(float),
                                        'visibility': coords[frame-1, 12, 2]
                                    },
                         'left_knee': {
                                        'x': coords[frame-1, 13, 0].astype(float), 
                                        'y': coords[frame-1, 13, 1].astype(float),
                                        'visibility': coords[frame-1, 13, 2]
                                    },
                         'right_knee': {
                                        'x': coords[frame-1, 14, 0].astype(float), 
                                        'y': coords[frame-1, 14, 1].astype(float),
                                        'visibility': coords[frame-1, 14, 2]
                                    },
                         'left_ankle': { 
                                        'x': coords[frame-1, 15, 0].astype(float), 
                                        'y': coords[frame-1, 15, 1].astype(float),
                                        'visibility': coords[frame-1, 15, 2]
                                    },
                         'right_ankle': {
                                        'x': coords[frame-1, 16, 0].astype(float), 
                                        'y': coords[frame-1, 16, 1].astype(float),
                                        'visibility': coords[frame-1, 16, 2]
                                    }
                        }

        # json_filename = str(frame) + '.json'
        # with open(json_filename, 'w') as fp:
        #     json.dump(self.lmDict, fp)
        return self.lmDict
            
    def keypoints_to_json (self, filename, frame_number):
        kptsDict = {"keypoints" : [
                                    [self.lmDict["left_hip"]["x"], self.lmDict["left_hip"]["y"], self.lmDict["left_hip"]["visibility"]],
                                    [self.lmDict["right_hip"]["x"], self.lmDict["right_hip"]["y"], self.lmDict["right_hip"]["visibility"]],
                                    [self.lmDict["left_knee"]["x"], self.lmDict["left_knee"]["y"], self.lmDict["left_knee"]["visibility"]],
                                    [self.lmDict["right_knee"]["x"], self.lmDict["right_knee"]["y"], self.lmDict["right_knee"]["visibility"]],
                                    [self.lmDict["left_ankle"]["x"], self.lmDict["left_ankle"]["y"], self.lmDict["left_ankle"]["visibility"]],
                                    [self.lmDict["right_ankle"]["x"], self.lmDict["right_ankle"]["y"], self.lmDict["right_ankle"]["visibility"]]
                                    ]
                                }
        json_filename = filename + "_" + str(frame_number) + ".json"
        with open(json_filename, 'w') as fp:
            json.dump(kptsDict, fp)

def main():
    import os
    import matplotlib.pyplot as plt
    from matplotlib import animation

    # ================================== PREPARATION OF THE VIDEO =====================================
    video_name = 'analysisVideo_2QxHrbp5N9Jv8WQpR_D3sgAyNPqcPLXmRzi'
    dir = 'PoseVideos/'
    extension = '.mp4'
    output_path = dir + 'preprocessed/' + 'pre_' + video_name + extension

    # video preprocessing if does not already exist
    if not os.path.exists(output_path):
        preprocess(dir + video_name + extension, output_path)

    # capture the preprocessed video
    cap = cv2.VideoCapture(output_path)

    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError
    
    # We need to set resolutions so, convert them from float to integer.
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Get the number of frames in the video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # detect pose
    detector = poseDetector(trackCon=0.7, model_complexity=2) 

    # ============================= PREPARATION FOR SAVING THE RESULTS =================================
    # result VideoWriter object will create a frame of above defined The output is stored in '*.mp4' file.
    result_path = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/PoseVideos/output_videos/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result = cv2.VideoWriter(result_path + 'output_' + video_name + '.mp4', 
                            cv2.VideoWriter_fourcc(*'MJPG'), #maybe change this 
                            10, (width, height))
    
    # Create a NumPy array to store the pose data to create the animation in 3D
    # The shape is 3x33x144 - 3D XYZ data for 33 landmarks across 144 frames
    # For each image in the video, extract the spatial pose data and save it in the appropriate spot in the `data` array 
    data = np.zeros((3, len(detector.mpPose.PoseLandmark), length))

    # define the saving frame rate to save keypoints coordinates in a JSON file 
    # and also save the frame to later annotate them for bbox
    # here we will save 1 frame per second 
    SAVING_FRAMES_PER_SECOND = 1
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = utils.get_saving_frames_durations(cap, saving_frames_per_second)
    # define a variable to count the frame
    frame_num = 0
    count = 0
    # ======================================= PROCESS EACH FRAME =========================================
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        
        img = detector.findPose(img, draw=False)
        if detector.results.pose_world_landmarks:
            landmarks = detector.results.pose_world_landmarks.landmark
            for i in range(len(detector.mpPose.PoseLandmark)):
                data[:, i, frame_num] = (landmarks[i].x, landmarks[i].y, landmarks[i].z)  # 2D only for now
        else :
            for i in range(17):
                data[:, i, frame_num] = (0.0, 0.0, 0.0)

        # lmDict 
        lmDict = detector.myfindPosition2(img, draw=False)
        # lmDict = detector.get_coords_detectron2('/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/coords.npz', frame_num)
        # lmList = detector.findPosition(img, draw=False)

        # detector.mediapipe_to_openpose_json(img, video_name)

        # get the duration by dividing the frame count by the FPS
        frame_duration = frame_num / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            data_path = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/dataset/' + video_name
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            os.chdir(data_path)
            
            frames_path = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/dataset/' + video_name + '/frames/'
            if not os.path.exists(frames_path):
                os.makedirs(frames_path)
            os.chdir(frames_path)
            count += 1
            cv2.imwrite(video_name + "_" + str(count) + ".jpg", img) 

            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass

            # save lmDict in a JSON file
            keypoints_path = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/dataset/' + video_name + '/keypoints/'
            if not os.path.exists(keypoints_path):
                os.makedirs(keypoints_path)
            os.chdir(keypoints_path)
            detector.keypoints_to_json(video_name, count)

        # increment the frame count
        frame_num += 1

        # calculate the angle of knee flexion
        angles = detector.findAngle2(img, draw=True)
        # update min angles
        detector.update_min_angle()
        
        # save the annotated video
        result.write(img)

        cv2.imshow("Image", img)
        cv2.waitKey(10)
    # print the min angle 
    print(detector.min_angles)

    cap.release()
    result.release()

    # ================================== EXPORT THE DATA IN NUMPY ARCHIVE =========================================
    # save coords in a nparray zip file
    coords_filename = 'coords' + video_name 
    np.savez_compressed('/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/data/' + coords_filename, data)

    # ============================================= PLOT THE DATA =================================================
    fig = plt.figure()
    fig.set_size_inches(5, 5, True)
    ax = fig.add_subplot(projection='3d')
    anim = utils.time_animate(data, fig, ax, detector.pose_connections, rotate_animation=False)

    # Save a rotation animation of the data
    os.chdir('/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/PoseVideos/anim')
    filename = 'pose_rotation' + video_name + '.mp4'
    anim.save(filename, fps=10, extra_args=['-vcodec', 'libx264'], dpi=300)

if __name__ == "__main__":
    main()