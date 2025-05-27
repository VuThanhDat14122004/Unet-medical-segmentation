import torch
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import Unet
from dataset import Dermatology_data
from loss import loss_combine
from train import trainner

# define folders data dir
folder_img_train_val = "Dataset/Train/Image/"
folder_mask_train_val = "Dataset/Train/Mask/"

# define dataset and dataloader train and val
dataset_train = Dermatology_data(folder_img_train_val, folder_mask_train_val, mode='train')
dataset_val = Dermatology_data(folder_img_train_val, folder_mask_train_val, mode='val')
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)


# define model, optimizer, loss function and hyperparameters
model = Unet()
dataloader_train=dataloader_train
dataloader_val=dataloader_val
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = loss_combine()
criterion_val = loss_combine()
epochs=150
device = "cuda" if torch.cuda.is_available() else "cpu"
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=3,
                                                       verbose=True)

# training
train_1 = trainner(model, dataloader_train, dataloader_val, optimizer, epochs, device, scheduler)
train_1.train()

# test model
model_load = Unet()                              
model_load.load_state_dict(train_1.best_weight)  
model_load = model_load.to(device)             
model_load.eval()

folder_img_test = "Dataset/Test/Image/"
dataset_test = Dermatology_data(folder_img_test, "", mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)

for imgs_raw in dataloader_test:
    imgs_test = imgs_raw/255
    imgs_test = imgs_test.to(torch.float32)
    imgs_test = imgs_test.to(device)
    output = model_load(imgs_test)
    ind=0
    for i in range(1):
        for j in range(8):
            fig, ax = plt.subplots(1, 2)
            img = imgs_raw[ind].cpu().detach().numpy()
            mask = output[ind].cpu().detach().numpy() 
            img = np.transpose(img, (1, 2, 0))
            img = img.astype(np.uint8)
            mask = np.transpose(mask, (1, 2, 0))
            mask [mask >= 0.5] = 1
            mask [mask < 0.5 ] = 0
            ax[0].imshow(img)
            ax[1].imshow(mask)
            ind += 1

# ploting loss
plt.plot(train_1.loss_val_list, label='Validation Loss')
plt.axvline(x=train_1.best_epoch, color='red', linestyle='--', label=f'Best Epoch: {train_1.best_epoch}')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss per Epoch')
plt.show()