import os
import logging
logger = logging.getLogger(__name__)

def get_dataset_obj():
    import dtlpy as dl
    dl.login_token(
        'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik5qZERNMFV3TVRJMlJUVTVPVGcwTVRnd01ESTVRelpHTUVJek0wSkdORVl6TXpJek1qYzVRUSJ9.eyJodHRwczovL2RhdGFsb29wLmFpL2F1dGhvcml6YXRpb24iOnsiZ3JvdXBzIjpbImFkbWlucyIsInBpcGVyIl0sInJvbGVzIjpbImFkbWluIl0sInBlcm1pc3Npb25zIjpbXX0sImVtYWlsIjoibm9hbXJAZGF0YWxvb3AuYWkiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6Ly9kYXRhbG9vcC1kZXZlbG9wbWVudC5hdXRoMC5jb20vIiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMTQzNTQ2NjkwNTk5OTA3Nzg0MTYiLCJhdWQiOiJJNEFycjlpeHM1UlQ0cUlqT0d0SVozME1WWHpFTTR3OCIsImlhdCI6MTU3ODk5NDQ2NCwiZXhwIjoxNTc5MDU5MjY0fQ.L15gcALu3_4aim4YoRWkhhmfDrnlinclY7FEM6VMA8VwbndCXNypPsqHBy-TgiQS_4_2QYjkraM5spmo6aK5c5P7PmPusAtC5XJ65E9M3plJ_D1DuenpLjlaH9vMYV3bc0rGCqggMMa3XIGrF0EF06-6zY_KwN_IsfZEBu98QibdoUXi3K56rxDKLdxdGwX--qk7Soy0Dx-kTwq-ugP1NIf5Q7xRyM1fQbQUhj2jZQGFVy6J5GIsbI_KpQT-5dU6WNiHsTRMZy-qYdUofbCIurZSqHiUmBrgbNw2NVAsWRt0ZNHsKt8eWRoru0PZzFlp0cURf0CZOGStEuclarM0MA')
    # dl.login()
    dl.setenv('dev')
    project = dl.projects.get(project_name='buffs_project')
    dataset_obj = project.datasets.get('my_data')
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
        logger.info('downloaded dataset to ', path_to_put_data)
        os.rename(os.path.join(path_to_put_data, 'items', name), os.path.join(path_to_put_data, name))
        os.rmdir(os.path.join(path_to_put_data, 'items'))
