import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
from models.gmm import get_mixture_coef
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import datetime
import sys
from utils.util import coords_render
from PIL import Image

""" device"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#import torch.nn.functional as F

#def one_hot_encoding(gts, label_map):
#    label_indices = [label_map[gt] for gt in gts]
#    label_indices_tensor = torch.tensor(label_indices)
#    one_hot_labels = F.one_hot(label_indices_tensor)
#    return one_hot_labels

class Trainer:
    def __init__(self, model, criterion, optimizer, data_loader, 
                logs, char_dict, valid_data_loader=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.char_dict = char_dict
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        #self.save_sample_dir = logs['sample']
        
    
    def _train_iter(self, data, step):
        self.model.train()
        prev_time = time.time()
        # prepare input
        img_list = data['img_list'].cuda()
        img_label = data['img_label'].long().cuda()
        # forward   
        flat_style_imgs = img_list.view(-1, 1, 64, 64)  # [B*2N, C:1, H, W]
        preds = self.model(flat_style_imgs) # [1920, 6763+1]
        
        # calculate loss
        loss = self.criterion(preds, img_label)
        
        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log file
        loss_dict = {"corss_entropy_loss": loss.item()}
        #self.tb_summary.text("corss_entropy_loss", str(loss.item()), step)
        self.tb_summary.add_scalars("loss", loss_dict, step)
        self.tb_summary.flush()
        iter_left = cfg.SOLVER.MAX_ITER - step
        time_left = datetime.timedelta(
                    seconds=iter_left * (time.time() - prev_time))
        self._progress(step, loss.item(), time_left)

        del data, preds, loss
        torch.cuda.empty_cache()

    def _valid_iter(self, step):
        self.model.eval()
        print('loading test dataset, the number is', len(self.valid_data_loader))
        try:
            test_loader_iter = iter(self.valid_data_loader)
            test_data = next(test_loader_iter)
        except StopIteration:
            test_loader_iter = iter(self.valid_data_loader)
            test_data = next(test_loader_iter)
        # prepare input
        img_list = test_data['img_list'].cuda()
        flat_style_imgs = img_list.view(-1, 1, 64, 64)  # [B*2N, C:1, H, W]
        img_label = test_data['img_label'].long().cuda()
        #img_one_hot_label = one_hot_encoding(img_label, self.label_map)
        #img_one_hot_label = F.one_hot(img_label)
         # forward
        with torch.no_grad():
            preds = self.model(flat_style_imgs)
            predicted_classes = torch.argmax(preds, dim=1).float()
            correct = (predicted_classes == img_label).sum().item()
            
            total = img_label.size(0)
            accuracy = correct / total
            self.tb_summary.add_scalar("accuracy", accuracy * 100, step)
            self.tb_summary.flush()
            print("Inference Accuracy: {:.2f}%".format(accuracy * 100))

    def train(self):
        """start training iterations"""    
        train_loader_iter = iter(self.data_loader)
        for step in range(cfg.SOLVER.MAX_ITER):
            try:
                data = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.data_loader)
                data = next(train_loader_iter)
            self._train_iter(data, step)
            #self._valid_iter(step)

            if (step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
               self._save_checkpoint(step)
            else:
                pass
            if self.valid_data_loader is not None:
                if (step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (step+1) % (cfg.TRAIN.VALIDATE_ITERS/10) == 0:
                    self._valid_iter(step)
            else:
                pass

    def _progress(self, step, loss, time_left):
        terminal_log = 'iter:%d ' % step
        terminal_log += '%s:%.3f ' % ('loss', loss)
        terminal_log += 'ETA:%s\r\n' % str(time_left)
        sys.stdout.write(terminal_log)

    def _save_checkpoint(self, step):
        model_path = '{}/checkpoint-iter{}.pth'.format(self.save_model_dir, step)
        torch.save(self.model.state_dict(), model_path)
        print('save model to {}'.format(model_path))