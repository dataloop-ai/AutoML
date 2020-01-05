import os


def get_dataset_obj():
    import dtlpy as dl
    dl.login_token(
        'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6Ik5qZERNMFV3TVRJMlJUVTVPVGcwTVRnd01ESTVRelpHTUVJek0wSkdORVl6TXpJek1qYzVRUSJ9.eyJodHRwczovL2RhdGFsb29wLmFpL2F1dGhvcml6YXRpb24iOnsiZ3JvdXBzIjpbInBpcGVyIl0sInJvbGVzIjpbXSwicGVybWlzc2lvbnMiOltdfSwiZW1haWwiOiJub2FtckBkYXRhbG9vcC5haSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJpc3MiOiJodHRwczovL2RhdGFsb29wLWRldmVsb3BtZW50LmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDExNDM1NDY2OTA1OTk5MDc3ODQxNiIsImF1ZCI6Ikk0QXJyOWl4czVSVDRxSWpPR3RJWjMwTVZYekVNNHc4IiwiaWF0IjoxNTc4MjEzNTM1LCJleHAiOjE1NzgyNzgzMzV9.F4ETp6VqhW2WokJhfcdh85eCNcbmOOHR7moai5iL_-ScWqCDjzg6ZL7_qK2sH8akgDIGVP9UkCM6dFotWuI3M7-pggsy9pd0Uo1H5uprUzsQhELpsU7UbF504VRXAQX1pVmpEDGFqs-TG9gvWX7T8KsbyP8PVN2M2aj7V5rwGtGqQoWSVYMpVkT7C_f9BzxQvH2txyzkBnQs-uaGmCX91qRingcps_f6AqN5xeWo7BUm8X51oHguUWw8MMjiGma48Xv2nmdeIPM9BeOeb5MlkJk9Iz35Gq_Sk1wWwImynw3zn6ps238Eawhgi93akdJZ7qTYVHPaKya_dYCLMflhoQ')
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
