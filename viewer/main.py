import os
from skimage.io import imread
import pandas as pd
import json
from flask import Flask, render_template

# global configuration
visual_genome_img_dir = 'vg_images/'
with open('/data/public/rw/datasets/visual_genome/vg_coco_caption.json', 'r') as f:
    vg_coco = json.load(f)
    print('caption loaded')
    id2caption = {str(data['vg_id']):data['caption'] for data in vg_coco}


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello_world'


@app.route('/show/', methods=['GET'])
def show_method_list():
    l_methods = sorted(os.listdir('results'))

    return render_template('show_methods.html', l_methods=l_methods)


@app.route('/show/<method>', methods=['GET'])
def show_list(method):
    files = os.listdir(os.path.join('results', method))
    imgids = sorted([f.split('.')[0] for f in files if f.lower().endswith('.tsv')])
    n_test_images = len(imgids)

    return render_template('show_index.html', imgids=imgids, method=method,
                           n_test_images=n_test_images)


@app.route('/show/<method>/<query_img_id>', methods=['GET'])
def show(method, query_img_id):
    query_img_path = os.path.join(visual_genome_img_dir, f'{query_img_id}.jpg')
    query_caption = id2caption[query_img_id]
    query_caption = ' / '.join([cap.strip() for cap in query_caption])

    # read result file
    sim_score_file = os.path.join('results', method, f'{query_img_id}.tsv')
    sim_score = pd.read_csv(sim_score_file, header=None, sep='\t')
    sim_score = sim_score.sort_values(1, ascending=False).reset_index()
    data = []  # (img_id, img_path, similarity_score)
    top_K = 20
    for i, row in sim_score.iterrows():
        img_id = str(int(row[0]))
        img_path = os.path.join(visual_genome_img_dir, f'{int(row[0])}.jpg')
        score = row[1]
        img_cap = id2caption[img_id][0].strip()
        data.append((img_id, img_path, score, img_cap))
        if i >= top_K:
            break

    return render_template('show.html',
                           query_img_path=query_img_path,
                           query_img_id=query_img_id,
                           query_caption=query_caption,
                           data=data,
                           method=method)
