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
result_path = '/data/rw/project/viewer_CBIR/viewer/results'
method_name = 'my_model'

# example data
query_img_id = 2315353
data = {'img_id': [713389, 2344690],
        'sim': [1.0, 0.1]}

# save data
data = pd.DataFrame(result)
output_path = os.path.join(result_path, f'{method_name}/{query_img_id}.tsv')
data[['img_id', 'sim']].to_csv(output_path,
                               sep='\t', header=False, index=False)
```
