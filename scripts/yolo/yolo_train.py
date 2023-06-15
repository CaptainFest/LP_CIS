import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str,
        default='/nfs/home/isaitov/NL/data/autoriaNumberplateDataset-2023-03-06/npdata/numberplate_config.yaml')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--aug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--model_size', type=str, default='s', choices=['n', 's', 'm'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(f"yolov8{args.model_size}.pt")
    model.to(args.device)
    model.train(data=args.data_folder, batch=args.batch, imgsz=640, epochs=args.epochs,
                augment=args.aug, name=f'yolov8{args.model_size}_b{args.batch}_ep{args.epochs}',
                project='../../exps/runs')
    model.val(name=f'yolov8{args.model_size}_b{args.batch}_ep{args.epochs}_val')
