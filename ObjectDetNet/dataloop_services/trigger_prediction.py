import logging
logger = logging.getLogger(__name__)

import dtlpy as dl
dl.setenv('prod')

def create_trigger():
    predict_service = dl.services.get('predict')
    project = dl.projects.get('golden_project')
    dataset = project.datasets.get('predict_rodent')
    dataset_id = dataset.id

    # TODO: where does dataset id go?
    trigger = predict_service.triggers.create(
        service_id=predict_service.id,
        resource=dl.TriggerResource.ITEM,
        actions=dl.TriggerAction.CREATED,
        name='predict',
        filters={'$and': [
        {'datasetId': dataset_id},
        {'metadata': {'system': {'mimetype': 'image/*'}}}]},
        execution_mode=dl.TriggerExecutionMode.ONCE,
        function_name='predict_single_item',
        project_id=project.id
    )
    logger.info('trigger created')

