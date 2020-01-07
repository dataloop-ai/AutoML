import os


def get_dataset_obj():
    import dtlpy as dl
    dl.login_token(
        'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6Ik5qZERNMFV3TVRJMlJUVTVPVGcwTVRnd01ESTVRelpHTUVJek0wSkdORVl6TXpJek1qYzVRUSJ9.eyJodHRwczovL2RhdGFsb29wLmFpL2F1dGhvcml6YXRpb24iOnsiZ3JvdXBzIjpbInBpcGVyIl0sInJvbGVzIjpbXSwicGVybWlzc2lvbnMiOltdfSwiZW1haWwiOiJub2FtckBkYXRhbG9vcC5haSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJpc3MiOiJodHRwczovL2RhdGFsb29wLWRldmVsb3BtZW50LmF1dGgwLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDExNDM1NDY2OTA1OTk5MDc3ODQxNiIsImF1ZCI6Ikk0QXJyOWl4czVSVDRxSWpPR3RJWjMwTVZYekVNNHc4IiwiaWF0IjoxNTc4Mzg4Mzk5LCJleHAiOjE1Nzg0NTMxOTl9.So1sMldoyt3-B2XWctAtE1Dtu3fx1vkgqEtc585BHXTezmTE2DBWkODFQ-p3ZZCG8RzW8iszse8XVgndCSxHui5QUbUpoOZH5PTj_n7EuTE9NEnZMik7-EJ46QuGOytXmYmXwCWHWJ68_mZPgRe21xgbXcFvLijOW4muTa-rG-z0Ga-xt-Yi1l9ZcrHlMsaXnyoZKviTvAjtEaD7klmDtZkB_-cSe-NMzH9oVPIIV4k3EnrWewWdTxKwIejtVlfaexoZbZ9NCcbz9eQ3RJ2lxwtCZ96GgGnw4Ms2XjOU1I4frNDDCOYRcuOiNno5P7bNFtEks2RT8OY6Ccto2M_Fcg')
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
