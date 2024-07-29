import argparse
from data_loader.loader import Online_Dataset
import torch
import numpy as np
import tqdm
from fastdtw import fastdtw
import cv2
from utils.util import writeCache, dxdynp_to_list, coords_render
import os
#from tensorboardX import SummaryWriter

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok = True)
        print("makedir "+dir_path)    
    else:
        print(dir_path+" already exist, no need to makedir.")

def main(opt):
    """ vis or not"""
    #v_size = 100
    vis = False
    if len(opt.visualize_dir) > 0:
        vis = True
        mkdir(opt.visualize_dir)
        mkdir(os.path.join(opt.visualize_dir, 'pred'))
        mkdir(os.path.join(opt.visualize_dir, 'gt'))
    
#    writer = SummaryWriter('dtwlogs')
    
    """ set dataloader"""
    test_dataset = Online_Dataset(opt.data_path)
    print('loading generated samples, the total amount of samples is', len(test_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              num_workers=8)
    
    char_dict = test_dataset.char_dict
    """start test iterations"""
    euclidean = lambda x, y: np.sqrt(sum((x - y) ** 2))
    fast_norm_dtw_len, total_num = 0, 0

    for data in tqdm.tqdm(test_loader):
        preds, preds_len, character_id, writer_id, coords_gts, len_gts = data['coords'], \
            data['coords_len'].long(), \
            data['character_id'].long(), \
            data['writer_id'].long(), \
            data['coords_gt'], \
            data['len_gt'].long()
            
        preds2 = preds.detach().cpu().numpy() #only for img
        coords_gts2 = coords_gts.detach().cpu().numpy() #only for img
            
        for i, pred in enumerate(preds):
            pred_len,  gt_len= preds_len[i], len_gts[i]
            pred_valid, gt_valid = pred[:pred_len], coords_gts[i][:gt_len]

            # Convert relative coordinates into absolute coordinates
            seq_1 = torch.cumsum(gt_valid[:, :2], dim=0)
            seq_2 = torch.cumsum(pred_valid[:, :2], dim=0)
            
            # DTW between paired real and fake online characters
            fast_d, _ = fastdtw(seq_1, seq_2, dist= euclidean)
            fast_norm_dtw_len += (fast_d/gt_len)
            
            #writer.add_scalar(str(writer_id[i].item()) + '_' + char_dict[character_id[i].item()], fast_d/gt_len, i)
            
            if vis:
                sk_pil = coords_render(preds2[i], split=True, width=256, height=256, thickness=8, board=0)
                sk_pil_gt = coords_render(coords_gts2[i], split=True, width=256, height=256, thickness=8, board=0)
                character = char_dict[character_id[i].item()]
                save_path = os.path.join(opt.visualize_dir, 'pred',
                                str(writer_id[i].item()) + '_' + character+'.png')
                
                save_path_gt = os.path.join(opt.visualize_dir, 'gt',
                                str(writer_id[i].item()) + '_' + character+'.png')
                
                try:
                    sk_pil.save(save_path)
                    sk_pil_gt.save(save_path_gt)
                except:
                    print('error. %s, %s, %s' % (save_path, str(writer_id[i].item()), character))
                    print('error. %s, %s, %s' % (save_path_gt, str(writer_id[i].item()), character))
                
                '''
                # Visualize and save the character trajectory
                img = np.ones((v_size, v_size, 3), np.uint8) * 255  # White background
                seq_1_scaled = (seq_1 * 20).int().cpu().numpy()  # Scale coordinates for better visualization
                seq_2_scaled = (seq_2 * 20).int().cpu().numpy()
                for point in seq_1_scaled:
                    cv2.circle(img, tuple(point), 2, (0, 0, 255), -1)  # Red circles for ground truth
                for point in seq_2_scaled:
                    cv2.circle(img, tuple(point), 2, (255, 0, 0), -1)  # Blue circles for prediction
                
                for j in range(len(seq_1_scaled) - 1):
                    cv2.line(img, tuple(seq_1_scaled[j]), tuple(seq_1_scaled[j + 1]), (0, 0, 255), 1)
                    
                for j in range(len(seq_2_scaled) - 1):
                    cv2.line(img, tuple(seq_2_scaled[j]), tuple(seq_2_scaled[j + 1]), (255, 0, 0), 1)
                
                cv2.imwrite(os.path.join(opt.visualize_dir, f'char_{total_num}.png'), img)
                '''
                
        total_num += len(preds)
    #writer.close()
    avg_fast_norm_dtw_len = fast_norm_dtw_len/total_num
    print(f"the avg fast_norm_len_dtw is {avg_fast_norm_dtw_len}")
    

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default='Generated/Chinese',
                        help='dataset path for evaluating the DTW distance between real and fake characters')
    parser.add_argument('--visualize_dir', dest='visualize_dir', default='',
                        help='visualize pictures')
    opt =  parser.parse_args() 
    main(opt)