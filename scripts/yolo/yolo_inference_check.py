import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str,
        default='/nfs/home/isaitov/NL/data/autoriaNumberplateDataset-2023-03-06/numberplate_config.yaml')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='/nfs/home/isaitov/NL/exps/yolo8_b32_')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.model_path)
    model.to(args.device)
    model.predict(data=args.data_folder, batch=args.batch, imgsz=640,
                  name=f'yolov8_pretrained_b{args.batch}',
                  project='../../exps/runs')
