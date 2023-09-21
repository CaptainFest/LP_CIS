import os
import yaml
import argparse

from yolo_tools import convert_dataset_to_yolo_format


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', nargs="*", type=str, default=['train', 'val', 'test'])
    parser.add_argument('--data_folder', type=str,
                        default='/nfs/home/isaitov/NL/data/autoriaNumberplateDataset-2023-03-06')
    args = parser.parse_args()
    return args


def make_yaml(path_to_dataset:str, dataset_splits:dict):
    yaml_path = os.path.join(path_to_dataset, "npdata/numberplate_config.yaml")
    yaml_dataset_config = {
        # Train/val/test sets
        "path": path_to_dataset,
        # Classes
        "nc": 1,  # number of classes
        "names": ["numberplate"]  # class names
    }
    print("[INFO] config:", yaml_dataset_config)
    print("[INFO] config saved into", yaml_path)
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump({**yaml_dataset_config, **dataset_splits},
                  yaml_file,
                  default_flow_style=False)


if __name__ == "__main__":
    args = parse_args()
    PATH_TO_DATASET = args.data_folder
    STATES = args.states
    print(STATES)

    CLASSES = ['numberplate']

    PATH_TO_RES_ANN = os.path.join(PATH_TO_DATASET, "npdata/labels/{}")
    PATH_TO_RES_IMAGES = os.path.join(PATH_TO_DATASET, "npdata/images/{}")
    PATH_TO_JSON = os.path.join(PATH_TO_DATASET, "{}/via_region_data.json")
    PATH_TO_IMAGES = os.path.join(PATH_TO_DATASET, "{}/")
    dataset_splits = {}
    for state in STATES:
        dataset_splits[state] = PATH_TO_RES_IMAGES.format(state)
        path_to_res_ann = PATH_TO_RES_ANN.format(state)
        path_to_res_images = PATH_TO_RES_IMAGES.format(state)

        print("[INFO]", state, "data creating...")
        os.makedirs(path_to_res_ann, exist_ok=True)
        os.makedirs(path_to_res_images, exist_ok=True)

        path_to_json = PATH_TO_JSON.format(state)
        path_to_images = PATH_TO_IMAGES.format(state)

        convert_dataset_to_yolo_format(
            path_to_res_ann,
            path_to_res_images,
            path_to_images,
            path_to_json,
            debug=False
        )

    make_yaml(PATH_TO_DATASET, dataset_splits)
