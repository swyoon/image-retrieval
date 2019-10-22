from utils import load_files, save_pickle, clean_str, save_json

fname_new_sg = '/data/public/rw/datasets/visual_genome/filtered_scene_graphs_with_adj.pkl'

sg = load_files(fname_new_sg)
img_id_list = list(sg.keys())

fname_split_info = '/data/public/rw/datasets/visual_genome/stanford_filtered/img_split.json'
split_info = load_files(fname_split_info)

train_id = []
test_id = []
not_in_stanford_filtered = []
new_split = {}
for img_id in img_id_list:
    if str(img_id) in split_info.keys():
        split = split_info[str(img_id)]
        if split == 'train':
            train_id.append(img_id)
            new_split[img_id] = 'train'
        elif split == 'train_all':
            train_id.append(img_id)
            new_split[img_id] = 'train'
        elif split == 'test':
            test_id.append(img_id)
            new_split[img_id] = 'test'
        else:
            print('unavailable split: {}'.format(split))
    else:
        not_in_stanford_filtered.append(img_id)

num_total = len(img_id_list)
num_not = len(not_in_stanford_filtered)
num_train = len(train_id)
num_test = len(test_id)

print("train: {}/{}, {}".format(num_train, num_total, num_train/num_total))
print("test: {}/{}, {}".format(num_test, num_total, num_test/num_total))
print("not in stanford filted: {}".format(num_not))

fname_new_split = "/data/project/rw/CBIR/img_split.json"
save_json(new_split, fname_new_split)