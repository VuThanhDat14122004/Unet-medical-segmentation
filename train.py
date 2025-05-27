from tqdm import tqdm
import torch
from loss import loss_combine

class Early_stop:
    def __init__(self, best_loss, current_epoch=0, consider_epochs=10):
        self.stop = False
        self.consider_epochs = consider_epochs
        self.current_epoch = current_epoch
        self.best_loss = best_loss
    def __call__(self, current_loss):
        if current_loss >= self.best_loss:
            self.current_epoch += 1
        else:
            self.current_epoch = 0
            self.best_loss = current_loss
        if self.current_epoch >= self.consider_epochs:
            self.stop = True
            return

class trainner:
    def __init__(self, model, dataloader_train, dataloader_val, optimizer, epochs, device, scheduler):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.criterion = loss_combine()
        self.criterion_val = loss_combine()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.best_loss = float('inf')
        self.best_weight = None
        self.best_epoch = 0
        self.scheduler = scheduler
        self.loss_val_list = []
        self.early_stop = Early_stop(float('inf'))
    def train(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(self.dataloader_train, desc=f"Epoch {epoch+1}/{self.epochs}, learning_rate {self.optimizer.param_groups[0]['lr']}", unit="batch")
            for images, masks in pbar:
                # images: batch_size, 3, 1040, 1040
                # masks: batch_size, 1, 1040, 1040
                images = images/255
                images = images.to(torch.float32)
                masks = masks.to(torch.float32)
                images = images.to(self.device)
                masks = masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=loss.item())
            loss_validation = self.validate()
            self.loss_val_list.append(loss_validation)
            if loss_validation < self.best_loss:
                self.best_loss = loss_validation
                self.best_weight = self.model.state_dict()
                self.best_epoch = epoch
            self.early_stop(loss_validation)
            if self.early_stop.stop == True:
                break
            self.scheduler.step(loss_validation)
            
    def validate(self):
        loss_list = []
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.dataloader_val, desc="Validation", unit="batch")
            for images, masks in pbar:
                images = images/255
                images = images.to(torch.float32)
                masks = masks.to(torch.float32)
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion_val(outputs, masks)
                loss_list.append(loss.item())
                pbar.set_postfix(loss=loss.item())
        print(f"Validation Loss: {sum(loss_list) / len(loss_list)}")
        return sum(loss_list) / len(loss_list)
