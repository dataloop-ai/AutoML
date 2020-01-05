import dtlpy as dl
import os

path = '/home/noam/data'

dl.login_token(
    'insert_token_here')
dl.setenv('dev')
project = dl.projects.get(project_id='buffs_project')
dataset_obj = project.datasets.get('my_data')
# dataset_obj.items.upload(local_path='/Users/noam/tiny_coco', remote_path='')
dataset_obj.items.download(local_path=path)
os.rename(os.path.join(path, 'items', 'tiny_coco'), os.path.join(path, 'tiny_coco'))