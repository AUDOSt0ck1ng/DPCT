import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from models.loss import SupConLoss, get_pen_loss, get_meaning_loss
from models.model import DPCT_Generator
from models.o_model import SDT_Generator
from utils.logger import set_log
from data_loader.loader import ScriptDataset
import torch
from trainer.trainer import Trainer
from models.encoder import Content_Cls

def is_weight_eq(model_a, model_b):
    for idx in model_a.state_dict().keys():
        if not torch.equal(model_a.state_dict()[idx], model_b.state_dict()[idx]):
            return 0
    return 1

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)
    """ set dataset"""
    train_dataset = ScriptDataset(
        cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TRAIN.ISTRAIN, cfg.MODEL.NUM_IMGS)
    print('number of training images: ', len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                               shuffle=True,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn_,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS)
    test_dataset = ScriptDataset(
       cfg.DATA_LOADER.PATH, cfg.DATA_LOADER.DATASET, cfg.TEST.ISTRAIN, cfg.MODEL.NUM_IMGS)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                              shuffle=True,
                                              sampler=None,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_fn_,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    char_dict = test_dataset.char_dict
    """ build model, criterion and optimizer"""
    model = DPCT_Generator(num_encoder_layers=cfg.MODEL.ENCODER_LAYERS,
            num_head_layers= cfg.MODEL.NUM_HEAD_LAYERS,
            wri_dec_layers=cfg.MODEL.WRI_DEC_LAYERS,
            gly_dec_layers= cfg.MODEL.GLY_DEC_LAYERS).to('cuda')
    
    d_model = 512
    layer=2
    fixed_classifier = Content_Cls(d_model, layer).to('cuda')   #(512, 2, 6763)
    
    ### load checkpoint
    if len(opt.pretrained_model) > 0:
        model.load_state_dict(torch.load(opt.pretrained_model))
        print('load pretrained model from {}'.format(opt.pretrained_model))
        
        fixed_classifier.load_state_dict(model.contentcls.state_dict())
        
        for param in model.contentcls.parameters():
            param.requires_grad = False
            
        for param in fixed_classifier.parameters():
            param.requires_grad = False
        
    elif len(opt.contentcls_pretrained) > 0:
        model_dict = load_specific_dict(model.contentcls, opt.contentcls_pretrained, "contentcls")
        model.contentcls.load_state_dict(model_dict)
    
        fixed_classifier.load_state_dict(model_dict)
    
        for param in model.contentcls.parameters():
            param.requires_grad = False
            
        for param in fixed_classifier.parameters():
            param.requires_grad = False
    
    #elif len(opt.content_pretrained) > 0:
    #    model_dict = load_specific_dict(model.contentcls.feature_ext, opt.content_pretrained, "feature_ext")
    #    model.contentcls.feature_ext.load_state_dict(model_dict)
        
        #for param in model.contentcls.feature_ext.parameters():
        #    param.requires_grad = False
    #    print('load content pretrained model from {}'.format(opt.content_pretrained))
    else:
        pass
    
    criterion = dict(NCE=SupConLoss(contrast_mode='all'), PEN=get_pen_loss, CLS=get_meaning_loss)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.SOLVER.BASE_LR)
    #cls_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR/2)
    """start training iterations"""
    writer_id_max = len(train_dataset.writer_dict)
    trainer = Trainer(model, criterion, optimizer, train_loader, logs, char_dict, writer_id_max, fixed_classifier, test_loader)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='',
                        dest='pretrained_model', required=False, help='continue to train model')
    parser.add_argument('--contentcls_pretrained', default='',
                        dest='contentcls_pretrained', required=False, help='continue to train content classifier')
    parser.add_argument('--content_pretrained', default='model_zoo/position_layer2_dim512_iter138k_test_acc0.9443.pth',
                        dest='content_pretrained', required=False, help='continue to train content encoder')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/CHINESE_CASIA.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--log', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    opt = parser.parse_args()
    main(opt)