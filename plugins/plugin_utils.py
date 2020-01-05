import os


def get_dataset_obj():
    import dtlpy as dl
    dl.login_token(
        'insert_token_here')
    dl.setenv('dev')
    project = dl.projects.get(project_name='buffs_project')
    dataset_obj = project.datasets.get('my_data')
    return dataset_obj

def download_data(dataset_obj):
    # check if data is downloaded if not then download
    name = 'tiny_coco'
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path_to_put_data = os.path.join(parent_dir, 'data')
    if not os.path.exists(path_to_put_data):
        os.mkdir(path_to_put_data)

    if len(os.listdir(path_to_put_data)) > 0:
        print('data already in data path')
    else:
        dataset_obj.items.download(local_path=path_to_put_data)
        os.rename(os.path.join(path_to_put_data, 'items', name), os.path.join(path_to_put_data, name))
        os.rmdir(os.path.join(path_to_put_data, 'items'))
