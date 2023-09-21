# CIS License Plate Detection and Recognition

LP detection and recognition models developed for CIS countries.

# User Guide

License plate detection
```
# prepare data for YOLO model
from scripts/convert_to_yolo import convert_dataset_to_yolo_format
convert_dataset_to_yolo_format(path_to_images, path_to_json)

# train model
python scripts/yolo_train.py --data_folder {path_to_dataset_in_yolo_format} 
```

License plate Optical Character Recognition
```
python scripts/troct_finetune.py --ocr_folder {path_to_OCR_dataset} 
```
