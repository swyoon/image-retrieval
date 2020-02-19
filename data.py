import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import pickle


def get_karpathy_split_light(verbose=True):
    """faster version"""
    karpathy_coco_path = '/data/project/rw/mscoco_processed/karphathy_split.pkl'
    d_split = pickle.load(open(karpathy_coco_path, 'rb'))

    if verbose:
        for key, val in d_split.items():
            print(f'{key}: {len(val)}')
    d_split['train'] = d_split['train'] + d_split['restval']
    return d_split


class BERTSimilarity:
    """"""
    def __init__(self, similarity_file, id_file):
        self.sims = np.load(similarity_file)
        self.l_id = np.load(id_file)
        self.id2idx = {img_id: idx for idx, img_id in enumerate(self.l_id)}

    def get_similarity(self, img_id_1, img_id_2):
        img_idx_1 = self.id2idx[int(img_id_1)]
        img_idx_2 = self.id2idx[int(img_id_2)]
        return self.sims[img_idx_1][img_idx_2]


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """update by woong.ssang"""
    def __init__(self, root='/data/project/rw/CBIR/data/coco',
                 vocab=None, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.ann_dir = os.path.join(root, 'annotations')
        train_cap_path = os.path.join(self.ann_dir, 'captions_train2014.json')
        val_cap_path = os.path.join(self.ann_dir, 'captions_val2014.json')
        self.d_split = get_karpathy_split_light()
        self.train_coco = COCO(train_cap_path)
        self.val_coco = COCO(val_cap_path)

        self.vocab = vocab
        self.transform = transform

    def _get_coco_by_imgid(self, img_id):
        """returns appropriate coco object"""
        if img_id in self.train_coco.imgs:
            return self.train_coco
        elif img_id in self.val_coco.imgs:
            return self.val_coco
        else:
            raise ValueError(f'Invalid COCO Image ID')

    def _get_coco_by_capid(self, cap_id):
        if cap_id in self.train_coco.anns:
            return self.train_coco
        elif cap_id in self.val_coco.anns:
            return self.val_coco
        else:
            raise ValueError(f'Invalid COCO Image ID')

    def get_img_path(self, img_id):
        coco = self._get_coco_by_imgid(img_id)
        img_filename = coco.loadImgs(ids=[img_id])[0]['file_name']

        if 'train' in img_filename:
            dirname = 'train2014'
        else:
            dirname = 'val2014'
        return os.path.join(self.img_dir, dirname, img_filename)

    def imgid2capid(self, img_id):
        coco = self._get_coco_by_imgid(img_id)
        l_cap_ids = coco.getAnnIds(imgIds=[img_id])
        return l_cap_ids

    def capid2imgid(self, cap_id):
        coco = self._get_coco_by_capid(cap_id)
        img_id = coco.loadAnns(cap_id)[0]['image_id']
        return img_id

    def get_caption(self, cap_id):
        coco = self._get_coco_by_capid(cap_id)
        return coco.loadAnns(cap_id)[0]['caption']

    # def __getitem__(self, index):
    #     """This function returns a tuple that is further passed to collate_fn
    #     """
    #     vocab = self.vocab
    #     root, caption, img_id, path, image = self.get_raw_item(index)

    #     if self.transform is not None:
    #         image = self.transform(image)

    #     # Convert caption (string) to word ids.
    #     tokens = nltk.tokenize.word_tokenize(
    #         str(caption).lower())
    #     caption = []
    #     caption.append(vocab('<start>'))
    #     caption.extend([vocab(token) for token in tokens])
    #     caption.append(vocab('<end>'))
    #     target = torch.Tensor(caption)
    #     return image, target, index, img_id

    # def get_raw_item(self, index):
    #     if index < self.bp:
    #         coco = self.coco[0]
    #         root = self.root[0]
    #     else:
    #         coco = self.coco[1]
    #         root = self.root[1]
    #     ann_id = self.ids[index]
    #     caption = coco.anns[ann_id]['caption']
    #     img_id = coco.anns[ann_id]['image_id']
    #     path = coco.loadImgs(img_id)[0]['file_name']
    #     image = Image.open(os.path.join(root, path)).convert('RGB')

    #     return root, caption, img_id, path, image

    # def __len__(self):
    #     return len(self.ids)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root='/data/project/rw/CBIR/data/f30k', 
                 vocab=None, transform=None):
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.vocab = vocab
        self.transform = transform
        json_path = os.path.join(root, 'dataset_flickr30k.json')
        self.dataset = jsonmod.load(open(json_path, 'r'))['images']
        self.ids = []
        # for i, d in enumerate(self.dataset):
        #     if d['split'] == split:
        #         self.ids += [(i, x) for x in range(len(d['sentences']))]

        self.d_imgid2capid = {t['imgid']: t['sentids'] for t in self.dataset}
        self.d_imgid2filename = {t['imgid']: t['filename'] for t in self.dataset}
        self.d_capid2imgid = {s['sentid']: s['imgid'] for t in self.dataset
                              for s in t['sentences']}
        self.d_captions = {s['sentid']: s['raw'] for t in self.dataset
                           for s in t['sentences']}
        self.d_split = {split: [t['imgid'] for t in self.dataset
                                if t['split'] == split]
                        for split in ['train', 'val', 'test']}

    def get_img_path(self, img_id):
        return os.path.join(self.img_dir, self.d_imgid2filename[img_id])

    def imgid2capid(self, img_id):
        return self.d_imgid2capid[img_id]

    def capid2imgid(self, cap_id):
        return self.d_capid2imgid[cap_id]

    def get_caption(self, cap_id):
        return self.d_captions[cap_id]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[int(img_id)])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if 'coco' in data_name:
        # COCO custom dataset
        dataset = CocoDataset(root=root,
                              json=json,
                              vocab=vocab,
                              transform=transform, ids=ids)
    elif 'f8k' in data_name or 'f30k' in data_name:
        dataset = FlickrDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                        batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, 'train', opt)
        train_loader = get_loader_single(opt.data_name, 'train',
                                         roots['train']['img'],
                                         roots['train']['cap'],
                                         vocab, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=workers,
                                         collate_fn=collate_fn)

        transform = get_transform(data_name, 'val', opt)
        val_loader = get_loader_single(opt.data_name, 'val',
                                       roots['val']['img'],
                                       roots['val']['cap'],
                                       vocab, transform, ids=ids['val'],
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name,
                                        roots[split_name]['img'],
                                        roots[split_name]['cap'],
                                        vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=workers,
                                        collate_fn=collate_fn)

    return test_loader
