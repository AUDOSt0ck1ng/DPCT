import os
import shutil

class MP:
    def __init__(self, key) -> None:
        self.gt_true = 0
        self.sdt_true = 0
        self.g1_true = 0
        self.total = 0
        self.key = key
    
    def add(self, sdt, gt, g1) -> None:
        self.total += 1
        
        if gt:
            self.gt_true += 1
        if sdt:
            self.sdt_true += 1
        if g1:
            self.g1_true += 1  
    
    def print(self):
        return self.key+': total:'+str(self.total)+', sdt:'+str(self.sdt_true)+', gt:'+str(self.gt_true)+', g1:'+str(self.g1_true)

    def isBigger(self): 
        if self.g1_true > self.sdt_true:
            return True
        else:
            return False

def parse_line(line):
    # 去除行末尾的換行符和多餘的空格
    line = line.strip()
    
    # 按照格式解析行內容
    filename, data = line.split(': ')
    
    key = filename.split('_')[1]
    
    sdt, gt, g1 = data.split(', ')
    
    # 提取每個標籤的值
    sdt_value = sdt.split(':')[1] == 'True'
    gt_value = gt.split(':')[1] == 'True'
    g1_value = g1.split(':')[1] == 'True'
    
    return {
        'filename': filename,
        'key': key,
        'sdt': sdt_value,
        'gt': gt_value,
        'g1': g1_value
    }

def read_txt_file(filepath, target, char_list):
    os.makedirs(target, exist_ok=True)
    if not os.path.exists(filepath+'/clslog.txt'):
        print(f"File {filepath+'/clslog.txt'} does not exist.")
        return
    
    with open(filepath+'/clslog.txt', 'r') as file:
        lines = file.readlines()
    
    lines = lines[:-1]
    data = {}
    
    for line in lines:
        parse_line_data = parse_line(line)
        filename = parse_line_data['filename']
        key = parse_line_data['key']
        sdt = parse_line_data['sdt']
        gt = parse_line_data['gt']
        g1 = parse_line_data['g1']
        
        
        #if key in char_list:  
        if True:
            item = data.get(key, False)
            if item:
                item.add(sdt, gt, g1)
            else:
                item = MP(key)
                item.add(sdt, gt, g1)
                data[key]=item
            '''
            if g1:
                continue
            else:
                folder = filename.split('_')[0]
                path = filepath+'/'+folder+'/'+filename
                tp = target+'/'+key
                os.makedirs(tp, exist_ok=True)
                shutil.copy(path, tp+'/'+filename)
            '''    
    return data

if __name__ == "__main__":
    filepath = '/home/hhc102u/SDT/Generated/2sets_test_wm-all+wc_redo(187999)/test'  # 替換為你的txt檔案路徑
    #filepath = '/home/hhc102u/SDT/Generated/2sets_test_June_bestcheck/test'
    #filepath = '/home/hhc102u/SDT/Generated/2sets_test_wc-only-redo(191999)/test'
    target = '/home/hhc102u/SDT/Generated/temp'
    char_list = ['免','兔']#]'乓','乒','兵','水','冰','丸','九','力','刀','刁','天','夫','中','串']
    parsed_data = read_txt_file(filepath, target, char_list)
    
    # 打印結果
    if True:
        for item in char_list:
            data = parsed_data.get(item, False)
            if data:
                print(data.print())            
    else:
        total = 0
        count = 0
        
        for item in parsed_data:
            data = parsed_data.get(item, False)
            if data:
                total+=1
                if(data.isBigger()):
                    count +=1
                #print(data.print()) 
        
        
        print('total: '+str(total))        
        print('G1 win: '+str(count))