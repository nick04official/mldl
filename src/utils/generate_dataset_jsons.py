import json, os

def get_directories(path):
    return sorted(next(os.walk(path))[1])

def get_files_filtered(path):
    return list(filter(lambda x: '(' not in x, sorted(next(os.walk(path))[2])))

def get_dataset(root_dir, frame_mode='rgb'):

    processed_frames_dir = 'processed_frames2'

    dataset_dict = {}

    for user in get_directories(os.path.join(root_dir, processed_frames_dir)):
        dataset_dict[user] = {}
        dir_user = os.path.join(root_dir, processed_frames_dir, user)
        for label in get_directories(dir_user):
            dataset_dict[user][label] = []
            dir_user_label = os.path.join(root_dir, processed_frames_dir, user, label)
            for scene in get_directories(dir_user_label):
                dir_user_label_scene = os.path.join(root_dir, processed_frames_dir, user, label, scene, frame_mode)
                frames = list(map(lambda x: os.path.join(processed_frames_dir, user, label, scene, frame_mode, x), get_files_filtered(dir_user_label_scene)))
                dataset_dict[user][label].append(frames)

    return dataset_dict

def get_dataset_flow(root_dir):
    processed_frames_dir = 'flow_x_processed'
    processed_frames_dir2 = 'flow_y_processed'

    dataset_dict = {}

    for user in get_directories(os.path.join(root_dir, processed_frames_dir)):
        dataset_dict[user] = {}
        dir_user = os.path.join(root_dir, processed_frames_dir, user)
        for label in get_directories(dir_user):
            dataset_dict[user][label] = []
            dir_user_label = os.path.join(root_dir, processed_frames_dir, user, label)
            for scene in get_directories(dir_user_label):
                frames = {'x': [], 'y': []}
                dir_user_label_scene_x = os.path.join(root_dir, processed_frames_dir, user, label, scene)
                frames['x'] = list(map(lambda x: os.path.join(processed_frames_dir, user, label, scene, x), get_files_filtered(dir_user_label_scene_x)))
                dir_user_label_scene_y = os.path.join(root_dir, processed_frames_dir2, user, label, scene)
                frames['y'] = list(map(lambda x: os.path.join(processed_frames_dir2, user, label, scene, x), get_files_filtered(dir_user_label_scene_y)))
                dataset_dict[user][label].append(frames)

    return dataset_dict

def get_dataset_rgb(root_dir):
    return get_dataset(root_dir)

def get_dataset_mmaps(root_dir):
    return get_dataset(root_dir, 'mmaps')

def generate_dataset_jsons(root_dir, output_dir='.'):
    dataset_rgb = get_dataset_rgb(root_dir)
    dataset_rgb_train = {'S1': dataset_rgb['S1'], 'S3': dataset_rgb['S3'], 'S4': dataset_rgb['S4']}
    dataset_rgb_valid = {'S2': dataset_rgb['S2']}
    
    dataset_mmaps = get_dataset_mmaps(root_dir)
    dataset_mmaps_train = {'S1': dataset_mmaps['S1'], 'S3': dataset_mmaps['S3'], 'S4': dataset_mmaps['S4']}
    dataset_mmaps_valid = {'S2': dataset_mmaps['S2']}
    
    dataset_flow = get_dataset_flow(root_dir)
    dataset_flow_train = {'S1': dataset_flow['S1'], 'S3': dataset_flow['S3'], 'S4': dataset_flow['S4']}
    dataset_flow_valid = {'S2': dataset_flow['S2']}
    
    with open(os.path.join(output_dir, 'dataset_rgb_train.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_rgb_train, f, ensure_ascii=True, indent=4)
    with open(os.path.join(output_dir, 'dataset_rgb_valid.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_rgb_valid, f, ensure_ascii=True, indent=4)
    with open(os.path.join(output_dir, 'dataset_mmaps_train.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_mmaps_train, f, ensure_ascii=True, indent=4)
    with open(os.path.join(output_dir, 'dataset_mmaps_valid.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_mmaps_valid, f, ensure_ascii=True, indent=4)
    with open(os.path.join(output_dir, 'dataset_flow_train.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_flow_train, f, ensure_ascii=True, indent=4)
    with open(os.path.join(output_dir, 'dataset_flow_valid.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_flow_valid, f, ensure_ascii=True, indent=4)