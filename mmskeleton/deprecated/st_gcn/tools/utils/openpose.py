from pathlib import Path
import json

def json_pack(snippets_dir, video_name, frame_width, frame_height, label='unknown', label_index=-1):
    sequence_info = []
    p = Path(snippets_dir)
    for path in p.glob(video_name+'*.json'):
        json_path = str(path)
        print(path)
        frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                coordinates += [keypoints[i]/frame_width, keypoints[i + 1]/frame_height]
                score += [keypoints[i + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index

    return video_info