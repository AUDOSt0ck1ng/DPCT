import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pickle
import argparse
from PIL import Image
from models.encoder import Content_Cls

def show_files(path, all_files):    
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            if not cur_path.endswith(('.png')):
                continue
            else:
                all_files.append(cur_path)
    return all_files

def show_files_in_test_dir(path):
    files = []
    for i in range(60):
        dirnum=str(i)
        dirpath = os.path.join(path, dirnum)
        show_files(dirpath, files)
        
    return files

def write_log_final(log_path, cls_counter, sdt_cls_correct_num, gt_cls_correct_num, g1_cls_correct_num)->None:
    try:
        print('total:%s, sdt_acc=%s, gt_acc=%s, g1_acc=%s\n' % (cls_counter, str(sdt_cls_correct_num.item()), str(gt_cls_correct_num.item()), str(g1_cls_correct_num.item())))
        with open(log_path, 'a') as file:
            file.write('total:%s, sdt_acc=%s, gt_acc=%s, g1_acc=%s\n' % (cls_counter, str(sdt_cls_correct_num.item()), str(gt_cls_correct_num.item()), str(g1_cls_correct_num.item())))
    except:
        print('error. fail to write cls acc result.')  

def write_log_iter(log_path, iter_num, file_names, sdt_result, gt_result, g1_result)->None:
    try:                              
        with open(log_path, 'a') as file:
            for i in range(iter_num):
                file.write('%s: sdt:%s, gt:%s, g1:%s\n' % (file_names[i], str(sdt_result[i].item()), str(gt_result[i].item()), str(g1_result[i].item())))
    except:
        print('error. fail to write cls result.') 

def cal_cls_correct_num(preds_cls, cls_gt):
    predicted_classes = torch.argmax(preds_cls, dim=1).float()
    correct_tensor = torch.eq(predicted_classes, cls_gt)
    return sum(correct_tensor), correct_tensor

def img_processing(image):
    height, width = image.shape
    #sub_image_height = height
    sub_image_width = width // 4
    image_SDT = image[:, sub_image_width:2 * sub_image_width]       #SDT
    image_GT = image[:, 2 * sub_image_width:3 * sub_image_width]   #GT    
    image_G1 = image[:, 3 * sub_image_width:4 * sub_image_width]   #G1  
      
    return image_SDT, image_GT, image_G1

def create_batches(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batches = []

    for batch in dataloader:
        batches.append(batch)

    return batches

# Define the transformations
def get_img_transform():
    transform = transforms.Compose([
    #transforms.Resize((64, 64)),  # Resize images to 64x64 
    transforms.ToTensor(),        # Convert images to tensor
    ])
    return transform

class CustomDataset(Dataset):
    def __init__(self, root_dir, char_dict_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        self.char_dict = pickle.load(open(char_dict_path, 'rb'))

        self.image_paths = show_files(root_dir,[])#show_files_in_test_dir(root_dir)
        #show_files(root_dir, [])

        #for file_name in os.listdir(root_dir):
        #    if file_name.endswith('.png'):
            #if file_name.endswith('.jpg') or file_name.endswith('.png'):
        #        self.image_paths.append(os.path.join(root_dir, file_name))
        
        self.img_nums = len(self.image_paths)

    def __len__(self):
        return self.img_nums

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        #image = Image.open(img_path).convert('L')  # Ensure image is in grayscale format
        image = cv2.imread(img_path, flags=0)
        image = image/255.
            
        image_SDT, image_GT, image_G1 = img_processing(image)
        
        if self.transform:
            image_SDT = self.transform(image_SDT).float()
            image_GT = self.transform(image_GT).float()
            image_G1 = self.transform(image_G1).float()

        label, img_name = self.extract_label_from_path(img_path)  # Extract label from image path

        return image_SDT, image_GT, image_G1, label, img_name

    def extract_label_from_path(self, path):
        # Placeholder for string processing logic to extract label from path
        # For example, if the label is part of the file name, you can extract it here
        # label = path.split('/')[-1].split('_')[0]  # Modify this line as needed
        #label = 0  # Replace with actual label extraction logic
        
        split_data = path.split('/')
        file_name = split_data[-1]
        word = file_name.split('_')[1]
        try:
            label = self.char_dict.find(word)
        except:
            raise Exception("dict fail, file_name: %s"%(file_name))
        
        return float(label), file_name

def main(opt):
    
    # Load cls model
    d_model = 512
    layer=2
    model_cls = Content_Cls(d_model, layer).to('cuda')   #(512, 2, 6763)
    
    if len(opt.pretrained_model_cls) > 0:
        model_cls.load_state_dict(torch.load(opt.pretrained_model_cls))
        print('load cls model from {}'.format(opt.pretrained_model_cls))
    
    model_cls.eval()
    
    # Init log_path
    log_path = os.path.join(opt.data_path, 'other_clslog.txt')
    
    # Initialize the dataset
    dataset = CustomDataset(root_dir=opt.data_path, char_dict_path=opt.char_dict_path, transform=get_img_transform())

    # Create batches
    batches = create_batches(dataset, batch_size=1920)

    sdt_cls_correct_num = 0 
    gt_cls_correct_num = 0
    g1_cls_correct_num = 0
    cls_counter = 0
    # Now you can use these batches
    it = 0
    
    for batch in batches:
        print('Iteration: %s' %(str(it)))
        it += 1
        
        images_SDT, images_GT, images_G1, labels, img_names = batch
    
        preds_sdt_cls = model_cls.forward(images_SDT.to('cuda'))
        preds_gt_cls = model_cls.forward(images_GT.to('cuda'))
        preds_g1_cls = model_cls.forward(images_G1.to('cuda'))
        
        #乒:55 乓:56 丸:37 九:62 黑:6709 黔:6710
        ans_index=6710
        sdt_probs = torch.softmax(preds_sdt_cls, dim=1)[:, ans_index]
        g1_probs = torch.softmax(preds_g1_cls, dim=1)[:, ans_index]
        
        labels = labels.to('cuda')
        
        num, sdt_result = cal_cls_correct_num(preds_sdt_cls, labels)
        sdt_cls_correct_num += num
        
        num, gt_result = cal_cls_correct_num(preds_gt_cls, labels)
        gt_cls_correct_num += num
        
        num, g1_result = cal_cls_correct_num(preds_g1_cls, labels)
        g1_cls_correct_num += num
        
        iter_num = len(gt_result)
        cls_counter += iter_num
        
        #717寫個分數就好。
        print(img_names)
        print(', '.join([f"{prob.item():.4f}" for prob in sdt_probs]))
        print(', '.join([f"{prob.item():.4f}" for prob in g1_probs]))
        
        #write iter    
        #write_log_iter(log_path, iter_num, img_names, sdt_result, gt_result, g1_result)
    
    #write final
    #write_log_final(log_path, cls_counter, sdt_cls_correct_num, gt_cls_correct_num, g1_cls_correct_num)
        

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_cls', default='/home/hhc102u/SDT/Saved/CHINESE_CASIA/pretrained_content_encoder/checkpoint-iter3999.pth',
                        dest='pretrained_model_cls', required=False, help='pretrained content encoder')
    parser.add_argument('--data_path', default='/home/hhc102u/SDT/Generated/temp/cal717', dest='data_path', required=False, help='data path')
    parser.add_argument('--char_dict_path', default='/home/hhc102u/SDT/data/CASIA_CHINESE/character_dict.pkl', dest='char_dict_path', required=False, help='char dict path')
    opt = parser.parse_args()
    main(opt)