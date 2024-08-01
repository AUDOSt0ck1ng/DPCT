#torch.cuda.set_device(2)
import os 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import os
import numpy as np
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from fastdtw import fastdtw
import torch
from data_loader.loader import ScriptDataset
#import pickle
from models.encoder import Content_Cls
from models.model import DPCT_Generator
from models.o_model import SDT_Generator
import tqdm
from utils.util import writeCache, dxdynp_to_list, coords_render, corrds2xys
#import lmdb
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok = True)

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

def img_merge_for_vis(width, height, imgs):
    new_image = Image.new('RGB', (width * len(imgs), height))
    for i in range(len(imgs)):
        new_image.paste(imgs[i], (width * i, 0))        #test_img, char_img, sdt prediction, ground truth, my prediction, my prediction-character-wise
    return new_image

def cal_cls_correct_num(preds_cls, cls_gt):
    predicted_classes = torch.argmax(preds_cls, dim=1).float()
    correct_tensor = torch.eq(predicted_classes, cls_gt)
    return sum(correct_tensor), correct_tensor

def add_SOS_token(preds, bs):
    SOS = torch.tensor(bs * [[0, 0, 1, 0, 0]]).unsqueeze(1).to(preds)
    preds = torch.cat((SOS, preds), 1)  # add the SOS token like GT
    preds = preds.detach().cpu().numpy()
    return preds

def tsne_save(feature_set_1, feature_set_2, f_desc_1, f_desc_2, save_path):
    num_samples_1 = feature_set_1.size(0)
    
    combined_features = torch.vstack((feature_set_1, feature_set_2))
    combined_features = combined_features.cpu()
    combined_features_np = combined_features.numpy()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_features_np)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_results[:num_samples_1, 0], tsne_results[:num_samples_1, 1], color='r', label=f_desc_1, alpha=0.5)
    plt.scatter(tsne_results[num_samples_1:, 0], tsne_results[num_samples_1:, 1], color='b', label=f_desc_2, alpha=0.5)
    plt.legend()
    plt.title('t-SNE projection of '+f_desc_1+' and '+f_desc_2)
    #plt.xlabel('Component 1')
    #plt.ylabel('Component 2')

    # 保存图像到文件
    plt.savefig(save_path, format='png')  # 可以指定其他格式如 'pdf', 'svg' 等
    #plt.show()

def sdt_meaning(style_imgs, model_cls):
    batch_size, num_imgs, in_planes, h, w = style_imgs.shape
    style_imgs = style_imgs.view(-1, in_planes, h, w)
    sdt_meaning_emb = model_cls.feature_ext(style_imgs)  #[4, B*N, C:512]
    sdt_pro_meaning_fea = sdt_meaning_emb#self.pro_mlp_meaning(meaning_emb) #[4, B*N, C:256]
    sdt_flat_pro_meaning_fea = torch.mean(sdt_pro_meaning_fea, 0)
    return sdt_flat_pro_meaning_fea

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()

    """setup data_loader instances"""
    test_dataset = ScriptDataset(
       cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS)#, True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    char_dict = test_dataset.char_dict
    writer_dict = test_dataset.writer_dict

    os.makedirs(os.path.join(opt.save_dir, 'test'), exist_ok=True)

    """build model architecture"""
    model = SDT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
    if len(opt.pretrained_model) > 0:
        model_weight = torch.load(opt.pretrained_model)
        model.load_state_dict(model_weight)
        print('load pretrained model from {}'.format(opt.pretrained_model))
    else:
        raise IOError('incorrect model_1 checkpoint path')
    model.eval()

    model_2 = DPCT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
    if len(opt.pretrained_model_2) > 0:
        model_weight_2 = torch.load(opt.pretrained_model_2)
        model_2.load_state_dict(model_weight_2)
        print('load pretrained model from {}'.format(opt.pretrained_model_2))
        
    else:
        raise IOError('incorrect model_2 checkpoint path')
    model_2.eval()
    
    d_model = 512
    layer=2
    #model_cls = Content_Cls(d_model, layer) #(512, 2, 6764)
    model_cls = Content_Cls(d_model, layer).to('cuda') #(512, 2, 6764)
    
    
    if len(opt.pretrained_model_cls) > 0:
        model_cls_weight = torch.load(opt.pretrained_model_cls)
        #model_cls.load_state_dict(model_2.contentcls.state_dict())
        model_cls.load_state_dict(model_cls_weight)
        print('load pretrained model from {}'.format(opt.pretrained_model_cls))
    else:
        raise IOError('incorrect model_cls checkpoint path')
    model_cls.eval()
    
    """calculate the total batches of generated samples"""
    if opt.sample_size == 'all':
        batch_samples = len(test_loader)
    else:
        batch_samples = int(opt.sample_size)*len(writer_dict)//cfg.TRAIN.IMS_PER_BATCH

    batch_num = 0
    data_iter = iter(test_loader)
    with torch.no_grad():
        
        cls_counter=0 #最終應該要=資料比數
        cls_log_path = os.path.join(opt.save_dir, 'test/clslog.txt')
        dtw_log_path = os.path.join(opt.save_dir, 'test/dtwlog.txt')
        
        sdt_cls_correct_num = 0 
        gt_cls_correct_num = 0
        g1_cls_correct_num = 0
        
        for _ in tqdm.tqdm(range(batch_samples)):
            batch_num += 1
            if batch_num > batch_samples:
                break
            else:
                data = next(data_iter)
                # prepare input
                coords, coords_len, character_id, writer_id, img_list, char_img, char_gt = data['coords'].cuda(), \
                    data['coords_len'].cuda(), \
                    data['character_id'].long(), \
                    data['writer_id'].long().cuda(), \
                    data['img_list'].cuda(), \
                    data['char_img'].cuda(), \
                    data['character_id'].long().cuda(), \
                
                #sdt_flat_pro_meaning_fea = sdt_meaning(img_list, model_cls)
                
                #preds = model.inference(img_list, char_img, 120)
                preds, sdt_flat_pro_meaning_fea, sdt_flat_pro_writer_fea, sdt_flat_pro_character_fea = model.inference_tsne(img_list, char_img, 120)
                bs = character_id.shape[0]
                preds = add_SOS_token(preds, bs)
                
                #preds_2 = model_2.inference(img_list, char_img, 120)
                preds_2, flat_pro_meaning_fea, flat_pro_writer_fea, flat_pro_character_fea = model_2.inference(img_list, char_img, 120)
                preds_2 = add_SOS_token(preds_2, bs)
                
                #pred_2_characterless
                #pred_2_characterless, _, _, _ = model_2.inference_option(img_list, char_img, 120, False, True)
                #pred_2_characterless = add_SOS_token(pred_2_characterless, bs)
                
                #pred_2_writerless
                #pred_2_writerless, _, _, _ = model_2.inference_option(img_list, char_img, 120, True, False)
                #pred_2_writerless = add_SOS_token(pred_2_writerless, bs)
                
                #pred_2_disable_all, _, _, _ = model_2.inference_option(img_list, char_img, 120, True, True)
                #pred_2_disable_all = add_SOS_token(pred_2_disable_all, bs)
                
                ##tsne繪製特徵分布
                #todo
                if len(opt.tsne) > 0:
                    tsne_save(sdt_flat_pro_meaning_fea, sdt_flat_pro_writer_fea, 'sdt content features', 'sdt writer features', os.path.join(opt.save_dir, 'sdt_WM_tsne.png'))
                    tsne_save(sdt_flat_pro_meaning_fea, flat_pro_writer_fea, 'dpct content features', 'dpct writer features', os.path.join(opt.save_dir, 'dpct_WM_tsne.png'))
                    tsne_save(sdt_flat_pro_character_fea, sdt_flat_pro_writer_fea, 'sdt character features', 'sdt writer features', os.path.join(opt.save_dir, 'sdt_WC_tsne.png'))
                    tsne_save(flat_pro_character_fea, flat_pro_writer_fea, 'dpct character features', 'dpct writer features', os.path.join(opt.save_dir, 'dpct_WC_tsne.png'))    
                
                coords = coords.detach().cpu().numpy()

                euclidean = lambda x, y: np.sqrt(sum((x - y) ** 2))
                content_slices = torch.unbind(char_img, dim=0)
                #img size and word
                w = h = 64#256
                tn=2#8
                
                #sdt gt g1 img data chunk
                tensor_sdt_idc =  torch.empty(0, 1, w, h)
                tensor_gt_idc = torch.empty(0, 1, w, h)
                tensor_g1_idc = torch.empty(0, 1, w, h)
                
                dtw_v1 = [] # sdt gt
                dtw_v2 = [] # g1 gt
                dtw_v3 = [] # sdt g1
                
                #計算dtw: gt, sdt, g1
                for i, pred in enumerate(preds):    
                    character = char_dict[character_id[i].item()]
                    pred, _ = dxdynp_to_list(preds[i])
                    pred_2, _ = dxdynp_to_list(preds_2[i])
                    coord, _ = dxdynp_to_list(coords[i])
                    
                    pred = corrds2xys(pred)
                    pred_len = pred.shape[0]
                    pred_2 = corrds2xys(pred_2)
                    pred_2_len = pred_2.shape[0]
                    coord = corrds2xys(coord)
                    gt_len = coord.shape[0]
                    
                    pred_valid, pred_2_valid, gt_valid = pred[:pred_len], pred_2[:pred_2_len], coord[:gt_len]

                    # Convert relative coordinates into absolute coordinates
                    seq_gt = torch.cumsum(torch.tensor(gt_valid)[:, :2], dim=0)
                    seq_1 = torch.cumsum(torch.tensor(pred_valid)[:, :2], dim=0)
                    seq_2 = torch.cumsum(torch.tensor(pred_2_valid)[:, :2], dim=0)
                    
                    # DTW between paired real and fake online characters
                    fast_gt_1, _ = fastdtw(seq_gt, seq_1, dist= euclidean)
                    fast_gt_2, _ = fastdtw(seq_gt, seq_2, dist= euclidean)
                    fast_1_2, _ = fastdtw(seq_1, seq_2, dist= euclidean)
                    
                    dtw_v1.append(fast_gt_1/gt_len)
                    dtw_v2.append(fast_gt_2/gt_len)
                    dtw_v3.append(fast_1_2/pred_len)
                
                for i, pred in enumerate(preds):
                    character = char_dict[character_id[i].item()]

                    sk_pil = coords_render(preds[i], split=True, width=w, height=h, thickness=tn, board=0)
                    sk_pil_2 = coords_render(preds_2[i], split=True, width=w, height=h, thickness=tn, board=0)
                    #sk_pil_2_characterless = coords_render(pred_2_characterless[i], split=True, width=w, height=h, thickness=tn, board=0)
                    #sk_pil_2_writerless = coords_render(pred_2_writerless[i], split=True, width=w, height=h, thickness=tn, board=0)
                    #sk_pil_2_disable_all = coords_render(pred_2_disable_all[i], split=True, width=w, height=h, thickness=tn, board=0)
                    sk_pil_gt = coords_render(coords[i], split=True, width=w, height=h, thickness=tn, board=0)
                    sk_pil_content = to_pil_image(content_slices[i])
                    
                    
                    width, height = sk_pil.size
                    imgs = [sk_pil_content, sk_pil, sk_pil_gt, sk_pil_2]
                    #imgs.append(sk_pil_2_characterless)
                    #imgs.append(sk_pil_2_writerless)
                    #imgs.append(sk_pil_2_disable_all)
                    
                    new_image = img_merge_for_vis(width, height, imgs)
                                        
                    new_image_path = os.path.join(opt.save_dir, 'test')
                    new_image_path = os.path.join(new_image_path, str(writer_id[i].item()))
                    mkdir(new_image_path)
                    new_image_path = os.path.join(new_image_path, str(writer_id[i].item()) + '_' + character+ '_sdt_gt_g1.png')
                    #圖片比較大直接寫檔好了。
                    try:
                        new_image.save(new_image_path)
                    except:
                        print('error. %s, %s, %s' % (new_image_path, str(writer_id[i].item()), character))
                    
                    
                    #img resize and to tensor
                    transform = get_proper_transform(w)
                    
                    #build sdt gt g1 data chunk
                    tensor_sdt_idc = torch.cat((tensor_sdt_idc, transform(sk_pil).unsqueeze(0)), dim=0)
                    tensor_gt_idc = torch.cat((tensor_gt_idc, transform(sk_pil_gt).unsqueeze(0)), dim=0)
                    tensor_g1_idc = torch.cat((tensor_g1_idc, transform(sk_pil_2).unsqueeze(0)), dim=0)
               
                #cls
                #preds_sdt_cls = model_cls.forward(tensor_sdt_idc)
                #preds_gt_cls = model_cls.forward(tensor_gt_idc)
                #preds_g1_cls = model_cls.forward(tensor_g1_idc)
                preds_sdt_cls = model_cls.forward(tensor_sdt_idc.to('cuda'))
                preds_gt_cls = model_cls.forward(tensor_gt_idc.to('cuda'))
                preds_g1_cls = model_cls.forward(tensor_g1_idc.to('cuda'))
                
                num, sdt_result = cal_cls_correct_num(preds_sdt_cls, char_gt)
                sdt_cls_correct_num += num
                
                num, gt_result = cal_cls_correct_num(preds_gt_cls, char_gt)
                gt_cls_correct_num += num
                
                num, g1_result = cal_cls_correct_num(preds_g1_cls, char_gt)
                g1_cls_correct_num += num
                
                #data num in this iteration
                iter_num = len(gt_result)
                cls_counter += iter_num
                #純文字比較小每個iteration再一起寫。
                try:                              
                    with open(cls_log_path, 'a') as file:
                        for i in range(iter_num):
                            file.write('%s: sdt:%s, gt:%s, g1:%s\n' % (str(writer_id[i].item()) + '_' + str(char_dict[character_id[i].item()])+ '_sdt_gt_g1.png', str(sdt_result[i].item()), str(gt_result[i].item()), str(g1_result[i].item())))
                except:
                    print('error. fail to write cls result.') 
                
                try:                              
                    with open(dtw_log_path, 'a') as file:
                        for i in range(iter_num):
                            file.write('%s: gt-sdt:%s, gt-g1:%s, sdt-g1:%s\n' % (str(writer_id[i].item()) + '_' + str(char_dict[character_id[i].item()])+ '_sdt_gt_g1.png', str(dtw_v1[i]), str(dtw_v2[i]), str(dtw_v3[i])))
                except:
                    print('error. fail to write dtw result.') 

        try:
            print('total:%s, sdt_acc=%s, gt_acc=%s, g1_acc=%s\n' % (cls_counter, str(sdt_cls_correct_num.item()), str(gt_cls_correct_num.item()), str(g1_cls_correct_num.item())))
            with open(cls_log_path, 'a') as file:
                file.write('total:%s, sdt_acc=%s, gt_acc=%s, g1_acc=%s\n' % (cls_counter, str(sdt_cls_correct_num.item()), str(gt_cls_correct_num.item()), str(g1_cls_correct_num.item())))
        except:
            print('error. fail to write cls acc result.')         
                
if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_CASIA.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated/2sets', help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default='', required=True, help='continue train model')
    parser.add_argument('--pretrained_model_2', dest='pretrained_model_2', default='', required=True, help='continue train model')
    parser.add_argument('--pretrained_model_cls', dest='pretrained_model_cls', default='', required=True, help='continue train model')
#    parser.add_argument('--store_type', dest='store_type', required=True, default='online', help='online or not')
#    parser.add_argument('--store_img', dest='store_img', required=True, default=True, help='True or False')
    parser.add_argument('--sample_size', dest='sample_size', default='500', required=True, help='randomly generate a certain number of characters for each writer')
    parser.add_argument('--tsne', dest='tsne', default='', required=False, help='print tsne or not')
    opt = parser.parse_args()
    main(opt)