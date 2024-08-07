import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
from models.gmm import get_mixture_coef
import os
import datetime
import sys
from utils.util import coords_render
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

def get_proper_transform(w):
    if w!=64:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    else:    
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform

class Trainer:
    def __init__(self, model, criterion, optimizer, data_loader, 
                logs, char_dict, writer_id_max, fixed_classifier, valid_data_loader=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.char_dict = char_dict
        self.writer_id_max = writer_id_max
        self.valid_data_loader = valid_data_loader
        self.nce_criterion = criterion['NCE']
        self.pen_criterion = criterion['PEN']
        self.cls_criterion = criterion['CLS']
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        
        self.fixed_classifier = fixed_classifier

    def _train_iter(self, data, step):
        self.model.train()
        prev_time = time.time()
        # prepare input
        coords, coords_len, character_id, writer_id, img_list, char_img = data['coords'].cuda(), \
            data['coords_len'].cuda(), \
            data['character_id'].long(), \
            data['writer_id'].long().cuda(), \
            data['img_list'].cuda(), \
            data['char_img'].cuda()
        #label_id = data['img_label']
        img_label = data['img_label'].long().cuda()
        # forward
        input_seq = coords[:, 1:-1]
        
        preds, nce_emb, nce_emb_patch, wc_nce_emb, w_nce_emb, flat_pro_meaning_fea = self.model(img_list, input_seq, char_img)
        
        #meaning feature positive pairs selection
        f_dim=512
        wm_nce_emb_part1 = torch.empty(0, 2, f_dim).cuda() #[len(label_id), 2, f_dim]
        content_count=0
        
        for i in range(len(img_label)):
            if True: # condiriton for negative samples 
            #correct_tensor[i]:
            #    continue
            #else:
                wm_nce_emb_row = torch.stack((flat_pro_meaning_fea[i], flat_pro_meaning_fea[i]), 0) # [2, C:f_dim]
                wm_nce_emb_row = wm_nce_emb_row.unsqueeze(0)#[1, 2, f_dim]
                wm_nce_emb_part1 = torch.cat((wm_nce_emb_part1, wm_nce_emb_row), dim=0)
                content_count+=1
       
        m_labels = torch.arange(self.writer_id_max, self.writer_id_max + content_count).cuda()
        wm_nce_emb = torch.cat((wm_nce_emb_part1, w_nce_emb), dim=0) # [content_count + B, 2, f_dim] B: 64, content_count: how many features meet the condition.
        wm_nce_emb = nn.functional.normalize(wm_nce_emb, p=2, dim=2)

        #character feature label = unique value
        c_max = wc_nce_emb.shape[0] - len(writer_id)
        increasing_values = torch.arange(self.writer_id_max, self.writer_id_max + c_max).cuda()
        
        writer_dummy_labels = torch.ones(len(writer_id)).cuda()
        wc_labels = torch.cat([writer_id, increasing_values])

        #content meaning
        wm_labels = torch.cat([m_labels, writer_id])
    
        # calculate loss
        gt_coords = coords[:, 1:, :]
        nce_loss_writer = self.nce_criterion(nce_emb, labels=writer_id)
        nce_loss_glyph = self.nce_criterion(nce_emb_patch)
        
        gamma_wc = 1
        gamma_wm = 1
        nce_loss_wc = gamma_wc * self.nce_criterion(wc_nce_emb, labels=wc_labels)
        nce_loss_wm = gamma_wm * self.nce_criterion(wm_nce_emb, labels=wm_labels)
        
        preds = preds.view(-1, 123)
        gt_coords = gt_coords.reshape(-1, 5)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits] = get_mixture_coef(preds)
        moving_loss_all, state_loss = self.pen_criterion(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, \
                                      o_corr, o_pen_logits, gt_coords[:,0].unsqueeze(-1), gt_coords[:,1].unsqueeze(-1), gt_coords[:,2:])
        
        moving_loss = torch.sum(moving_loss_all) / torch.sum(coords_len)
        pen_loss = moving_loss + 2*state_loss       

        loss = pen_loss + nce_loss_writer + nce_loss_glyph + nce_loss_wc + nce_loss_wm
        
        # backward and update trainable parameters
        self.model.zero_grad()
    
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()
        
        # log file
        loss_dict = {"total_loss": loss.item(), "pen_loss": pen_loss.item(), "moving_loss": moving_loss.item(),"state_loss": state_loss.item(), 
                     "nce_loss_writer": nce_loss_writer.item(), "nce_loss_glyph": nce_loss_glyph.item(), "nce_loss_wc": nce_loss_wc.item(), "nce_loss_wm": nce_loss_wm.item(),}
        self.tb_summary.add_scalars("loss", loss_dict, step)
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
        coords, coords_len, character_id, writer_id, img_list, char_img = test_data['coords'].cuda(), \
            test_data['coords_len'].cuda(), \
            test_data['character_id'].long().cuda(), \
            test_data['writer_id'].long().cuda(), \
            test_data['img_list'].cuda(), \
            test_data['char_img'].cuda()
         # forward
        with torch.no_grad():
            preds, flat_pro_meaning_fea, flat_pro_writer_fea, flat_pro_character_fea = self.model.inference(img_list, char_img, 120)
            bs = character_id.shape[0]
            SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
            preds = torch.cat((SOS, preds), 1)  # add the first token
            preds = preds.cpu().numpy()
            gt_coords = coords.cpu().numpy()  # [N, T, C]
            self._vis_genarate_samples(gt_coords, preds, character_id, step)

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

            if (step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
               self._save_checkpoint(step)
            else:
                pass
            if self.valid_data_loader is not None:
                if (step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
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

    def _vis_genarate_samples(self, gt_coords, preds, character_id, step):
        for i, _ in enumerate(gt_coords):
            gt_img = coords_render(gt_coords[i], split=True, width=64, height=64, thickness=1)
            pred_img = coords_render(preds[i], split=True, width=64, height=64, thickness=1)
            example_img = Image.new("RGB", (cfg.TEST.IMG_W * 2, cfg.TEST.IMG_H),
                                    (255, 255, 255))
            example_img.paste(pred_img, (0, 0)) # gererated character
            example_img.paste(gt_img, (cfg.TEST.IMG_W, 0)) # gt character
            character = self.char_dict[character_id[i].item()]
            save_path = os.path.join(self.save_sample_dir, 'ite.' + str(step//100000)
                 + '-'+ str(step//100000 + 100000), character + '_' + str(step) + '_.jpg')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                example_img.save(save_path)
            except:
                print('error. %s, %s' % (save_path, character))