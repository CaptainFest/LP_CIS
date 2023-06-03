import numpy as np
from PIL import Image, ImageFilter, ImageOps

import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from ray.tune.schedulers import PopulationBasedTraining
from torchvision.transforms import RandomChoice, RandomRotation, GaussianBlur, Resize


class Dilation(torch.nn.Module):

    def __init__(self, kernel=3):
        super().__init__()
        self.kernel=kernel

    def forward(self, img):
        return img.filter(ImageFilter.MaxFilter(self.kernel))

class Erosion(torch.nn.Module):

    def __init__(self, kernel=3):
        super().__init__()
        self.kernel=kernel

    def forward(self, img):
        return img.filter(ImageFilter.MinFilter(self.kernel))

class Underline(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        img_np = np.array(img.convert('L'))
        black_pixels = np.where(img_np < 50)
        try:
            y1 = max(black_pixels[0])
            x0 = min(black_pixels[1])
            x1 = max(black_pixels[1])
        except:
            return img
        for x in range(x0, x1):
            for y in range(y1, y1-3, -1):
                try:
                    img.putpixel((x, y), (0, 0, 0))
                except:
                    continue
        return img

class KeepOriginal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img
    
    
def square_padding(img):
    w, h = img.size
    desired_size = max(w, h)
    delta_w = desired_size - w
    delta_h = desired_size - h
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    img = ImageOps.expand(img, padding)
    return img


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128, train=False, aug=False, square=False, size=(224,224)):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = RandomChoice([RandomRotation(degrees=(-10, 10), expand=True, fill=255),
                                       GaussianBlur(3),
                                       Dilation(3),
                                       Erosion(3),
                                       Resize((size[0] // 3, size[1] // 3), interpolation=0),
                                       Underline(),
                                       KeepOriginal()])
        self.square = square
        self.aug = aug
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        
        image = Image.open(self.root_dir + file_name).convert("RGB")
        if self.square:
            image = square_padding(image)
        
        if self.train and self.aug:
            image = self.transform(image)
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


def get_names_and_np(folder:str):
    data = []
    for fn in os.listdir(os.path.join(folder, 'img')):
        with open(os.path.join(folder, 'ann', f"{fn.rsplit('.', 1)[0]}.json"), 'r') as f:
            js_data = json.load(f)
            data.append([fn, js_data['description']])
    return data
    

def get_df_from_folder(train_folder:str, val_folder:str, test_folder:str, columns=['file_name', 'text']):
    train_data, val_data, test_data = get_names_and_np(train_folder), \
                                      get_names_and_np(val_folder),   \
                                      get_names_and_np(test_folder)
    train_df, val_df, test_df = pd.DataFrame(train_data, columns=columns), \
                                pd.DataFrame(val_data, columns=columns),   \
                                pd.DataFrame(test_data, columns=columns)
    return train_df, val_df, test_df