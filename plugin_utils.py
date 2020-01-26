import os
import json
import logging
logger = logging.getLogger(__name__)

def get_dataset_obj():
    import dtlpy as dl
    with open("dataloop_configs.json") as f:
        dataloop_configs = json.load(f)
    project_name = dataloop_configs['project']
    dataset_name = dataloop_configs['dataset']
    project = dl.projects.get(project_name=project_name)
    dataset_obj = project.datasets.get(dataset_name)
    return dataset_obj


def maybe_download_data(dataset_obj):
    # check if data is downloaded if not then download
    name = 'tiny_coco'
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path_to_put_data = os.path.join(parent_dir, 'data')
    if not os.path.exists(path_to_put_data):
        os.mkdir(path_to_put_data)

    if os.path.exists(os.path.join(path_to_put_data, name)):
        logger.info(name + ' already exists, no need to download')
    else:
        dataset_obj.items.download(local_path=path_to_put_data)
        logger.info('downloaded dataset to ' + path_to_put_data)
        os.rename(os.path.join(path_to_put_data, 'items', name), os.path.join(path_to_put_data, name))
        os.rmdir(os.path.join(path_to_put_data, 'items'))
