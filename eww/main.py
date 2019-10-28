import os
from skimage.io import imread
import pandas as pd
from flask import Flask, render_template

# global configuration
visual_genome_img_dir = 'vg_images/'

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello_world'


@app.route('/show/<method>', methods=['GET'])
def show_list(method):
    files = os.listdir(os.path.join('results', method))
    print(files)
    imgids = [f.split('.')[0] for f in files if f.lower().endswith('.tsv')]
    print(imgids)

    return render_template('show_index.html', imgids=imgids, method=method)


@app.route('/show/<method>/<query_img_id>', methods=['GET'])
def show(method, query_img_id):
    query_img_path = os.path.join(visual_genome_img_dir, f'{query_img_id}.jpg')

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
        data.append((img_id, img_path, score))
        if i >= top_K:
            break

    return render_template('show.html',
                           query_img_path=query_img_path,
                           query_img_id=query_img_id,
                           data=data,
                           method=method)
