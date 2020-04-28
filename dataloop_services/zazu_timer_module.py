import logging
import dtlpy as dl
import json
import torch
import os
from time import sleep
from dataloop_services.plugin_utils import maybe_download_pred_data, download_and_organize
from eval import precision_recall_compute
logger = logging.getLogger(__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, configs, time, test_dataset, query):
        self.configs_input = dl.FunctionIO(type='Json', name='configs', value=configs)
        self.service = dl.services.get('zazu')
        project_name = configs['dataloop']['project']
        self.project = dl.projects.get(project_name)

        maybe_download_pred_data(dataset_obj=test_dataset, val_query=query)

        filters = dl.Filters()
        filters.custom_filter = query
        dataset_name = test_dataset.name
        path_to_dataset = os.path.join(os.getcwd(), dataset_name)
        download_and_organize(path_to_dataset=path_to_dataset, dataset_obj=test_dataset, filters=filters)

        json_file_path = os.path.join(path_to_dataset, 'json')

        self.compute = precision_recall_compute()
        self.compute.add_dataloop_local_annotations(json_file_path)

        self._circle(time)

    def _circle(self, time_lapse):
        while 1:
            logger.info("running new execution")
            execution_obj = self.service.execute(function_name='search', execution_input=[self.configs_input],
                                                 project_id='fcdd792b-5146-4c62-8b27-029564f1b74e')
            while execution_obj.latest_status['status'] != 'success':
                sleep(5)
                execution_obj = dl.executions.get(execution_id=execution_obj.id)
                if execution_obj.latest_status['status'] == 'failed':
                    raise Exception("plugin execution failed")
            logger.info("execution object status is successful")
            self.project.artifacts.download(package_name='zazuml',
                                            execution_id=execution_obj.id,
                                            local_path=os.getcwd())
            logger.info(str(os.listdir('.')))
            new_checkpoint_name = 'checkpoint_' + str(execution_obj.id) + '.pt'
            os.rename('checkpoint0.pt', new_checkpoint_name)

            new_checkpoint = torch.load(new_checkpoint_name, map_location=torch.device('cpu'))
            model_name = new_checkpoint['model_specs']['name']

            model_obj = dl.models.get(model_name=model_name)
            adapter_temp = model_obj.build(local_path=os.getcwd())
            adapter_temp.load_inference(checkpoint_path=new_checkpoint_name)
            output_path = adapter_temp.predict(output_dir='new_checkpoint')
            self.compute.add_path_detections(output_path, model_name='new_checkpoint')
            new_checkpoint_mAP = self.compute.get_metric(model_name='new_checkpoint', precision_to_recall_ratio=1.)

            best_checkpoint = model_obj.checkpoints.get('checkpoint0')
            check0_path = best_checkpoint.download(local_path=os.getcwd())
            adapter = model_obj.build(local_path=os.getcwd())
            adapter.load_inference(checkpoint_path=check0_path)
            output_path = adapter.predict(output_dir=best_checkpoint.name)
            self.compute.add_path_detections(output_path, model_name=best_checkpoint.name)
            best_checkpoint_mAP = self.compute.get_metric(model_name=best_checkpoint.name, precision_to_recall_ratio=1.)

            if new_checkpoint_mAP > best_checkpoint_mAP:
                model_obj.checkpoints.upload(checkpoint_name='checkpoint0',
                                             local_path=new_checkpoint_name)

                logger.info('switching with new checkpoint')

            self.compute.save_plot_metrics()

            sleep(time_lapse)
