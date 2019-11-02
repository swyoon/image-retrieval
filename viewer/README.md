# CBIR Viewer Web App

A web-based (barely functional) viewer application for visualizing CBIR results.  
Mainly based on Flask and Bootstrap.

## Access

* [http://gpu-cloud-vnode33.dakao.io:6481/show/resnet](http://gpu-cloud-vnode33.dakao.io:6481/show/resnet)


## Run


`FLASK_APP=main.py FLASK_ENV=development flask run --port 6481 --host 0.0.0.0`


## Make prediction result files

```python
import os
import pandas as pd

# configuration
result_dir = '/data/project/rw/viewer_CBIR/viewer/results'
method_name = 'my_model'
out_dir = os.path.join(result_dir, method_name)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    print(f'creating {out_path}...')


# example data
query_img_id = 2315353
data = {'img_id': [713389, 2344690],
        'sim': [1.0, 0.1]}

# save data
data = pd.DataFrame(data)
output_path = os.path.join(out_dir, f'{query_img_id}.tsv')
data[['img_id', 'sim']].to_csv(output_path,
                               sep='\t', header=False, index=False)
print(f'Successfully saved at {output_path}')
```
