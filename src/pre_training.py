import torch
from dotenv import dotenv_values

from functions.mbhm import mbhm_vibration_dataloader as mbhm_loader
from models.FCN import FaultClassificationNetwork as FCN


class HyperParameters:
    def __init__(self):
        self.batch_size = 1024
        self.num_workers = 0
        self.lr = 1e-4
        self.lr_patience = 150
        self.lr_factor = 0.5
        self.epoch_max = 50
        self.device = 'cpu'


class PreTrainner:
    def __init__(self):
        self.hp = HyperParameters()
        self.train_loader = mbhm_loader('train', self.hp.batch_size, num_workers=self.hp.num_workers)
        self.val_loader = mbhm_loader('val', self.hp.batch_size, num_workers=self.hp.num_workers)
        self.test_loader = mbhm_loader('test', self.hp.batch_size, num_workers=self.hp.num_workers)
        self.model = FCN().to(self.hp.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.hp.lr_patience,
                                                                    factor=self.hp.lr_factor)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_val_loss = 1e10
        self.best_val_acc = 0

    def train_epoch(self):
        self.model.train()
        for i, (data, label) in enumerate(self.train_loader):
            data = data.to(self.hp.device)
            label = label.to(self.hp.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

    def eval_epoch(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for i, (data, label) in enumerate(self.val_loader):
                data = data.to(self.hp.device)
                label = label.to(self.hp.device)
                output = self.model(data)
                loss = self.criterion(output, label)
                val_loss += loss.item()
                val_acc += (output.argmax(1) == label).sum().item()
            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader.dataset)
        return val_loss, val_acc

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            test_acc = 0
            for i, (data, label) in enumerate(self.test_loader):
                data = data.to(self.hp.device)
                label = label.to(self.hp.device)
                output = self.model(data)
                test_acc += (output.argmax(1) == label).sum().item()
            test_acc /= len(self.test_loader.dataset)
        return test_acc

    def train(self):
        for epoch in range(self.hp.epoch_max):
            self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_weights(dotenv_values()['FCN_WEIGHTS_DIR'])
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(dotenv_values()['FCN_WEIGHTS_DIR'])
            print(f'epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}')
            if self.scheduler.state_dict()['_last_lr'][0] < 1e-7:
                break
        test_acc = self.test_epoch()
        print(f'test_acc: {test_acc}')


if __name__ == "__main__":
    pre_trainner = PreTrainner()
    pre_trainner.train()