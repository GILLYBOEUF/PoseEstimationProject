import json
import os

def merge_bboxes_kpts (file_bboxes, file_kpts, frame_nb, save_path):
    with open(file_bboxes, "r") as fbb :
        bboxes = json.load(fbb)
        data_bboxes = bboxes["annotations"][frame_nb-1]
        data_bboxes = data_bboxes["bbox"]
       
    with open(file_kpts, "r") as fk :
        kpts = json.load(fk)
        data_kpts = kpts["keypoints"]

    data_joined = {"bboxes" :[data_bboxes],
                   "keypoints": [data_kpts]}

    json_filename =  save_path + '_' + str(frame_nb) + ".json"
    with open(json_filename, 'w') as fp:
        json.dump(data_joined, fp)


def main():
    path_to_data = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/PoseEstimationProject/dataset/'
    video_list = os.listdir(path_to_data)

    for video_name in video_list:
        if 'analysisVideo' in video_name:
            frame_list = os.listdir(path_to_data + video_name + '/' + 'keypoints/')
            for i in range (len(frame_list)):
                path_file_bboxes = path_to_data + video_name + '/' + 'bboxes.json'
                path_file_kpts = path_to_data + video_name + '/' + 'keypoints/' + video_name + '_' + str(i+1) + '.json'
                json_path = path_to_data + video_name + '/' + 'json/'
                if not os.path.exists(json_path):
                    os.makedirs(json_path)
                os.chdir(json_path)
                save_path = path_to_data + video_name + '/' + 'json/' + video_name
                merge_bboxes_kpts (path_file_bboxes, path_file_kpts, i+1, save_path)

if __name__ == "__main__":
    main()