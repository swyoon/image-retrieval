import h5py
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import json
import os
from PIL import Image
import numpy as np

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

DATASET = 'f30k'
assert DATASET in ('coco', 'f30k')

if DATASET == 'coco':
    coco_img_train = '/data/public/rw/datasets/coco/images/train2014/'
    coco_img_val = '/data/public/rw/datasets/coco/images/val2014/'
    coco_cap_train = '/data/public/rw/datasets/coco/annotations/captions_train2014.json'
    coco_cap_val = '/data/public/rw/datasets/coco/annotations/captions_val2014.json'


    coco_cap_train = json.load(open(coco_cap_train, 'r'))
    coco_cap_val = json.load(open(coco_cap_val, 'r'))

    l_img_path = [ coco_img_train + d_img['file_name'] for d_img in coco_cap_train['images']] + \
                 [ coco_img_val + d_img['file_name'] for d_img in coco_cap_val['images']]
    print(len(l_img_path))
    print(len(set(l_img_path)))
    l_id = [ d_img['id'] for d_img in coco_cap_train['images']] + \
            [d_img['id'] for d_img in coco_cap_val['images']]
    print(len(l_id))

elif DATASET == 'f30k':
    f = '/data/project/rw/CBIR/data/f30k/dataset_flickr30k.json'
    img_dir = '/data/project/rw/CBIR/data/f30k/images/'
    f30k = json.load(open(f))['images']
    l_img_path = [os.path.join(img_dir, t['filename']) for t in f30k]
    l_id = [t['imgid'] for t in f30k]

resnet = models.resnet152(pretrained=True).cuda()
feature_part = list(resnet.children())[:-1]
resnet = nn.Sequential(*feature_part)
resnet = resnet.cuda()
resnet.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
resize = transforms.Resize(224)
centercrop = transforms.CenterCrop(224)

l_feat = []
for img_path in tqdm(l_img_path):
    pil_img = Image.open(img_path, mode='r').convert('RGB')
    img_tensor = normalize(to_tensor(centercrop(resize(pil_img))))
    img_tensor = img_tensor.unsqueeze(0).cuda()
    feat = resnet(img_tensor)
    feat = feat.squeeze(3).squeeze(2)[0]
    l_feat.append(feat.detach().cpu().numpy())


# out_path = '/data/project/rw/mscoco_processed/resnet152_2.h5'
out_path = f'/data/project/rw/CBIR/data/{DATASET}/resnet152.h5'
f = h5py.File(out_path, mode='w')
f.create_dataset('resnet_feature', data=np.array(l_feat))
f.create_dataset('id', data=l_id)
