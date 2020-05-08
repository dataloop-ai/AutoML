import logging
import dtlpy as dl
import json
import torch
import os
from time import sleep
from dataloop_services.plugin_utils import maybe_download_pred_data, download_and_organize
from eval import precision_recall_compute
from logging_utils import logginger, init_logging



class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, configs, time, test_dataset_id, query):
        time = int(time)
        dl.setenv('prod')
        configs = json.loads(configs)
        query = json.loads(query)
        self.configs_input = dl.FunctionIO(type='Json', name='configs', value=configs)
        self.service = dl.services.get('zazu')
        project_name = configs['dataloop']['project']
        self.project = dl.projects.get(project_name)
        test_dataset = self.project.datasets.get(dataset_id=test_dataset_id)
        maybe_download_pred_data(dataset_obj=test_dataset, val_query=query)

        # add gt annotations
        filters = dl.Filters()
        filters.custom_filter = query
        dataset_name = test_dataset.name
        path_to_dataset = os.path.join(os.getcwd(), dataset_name)
        # only download if doesnt exist
        if not os.path.exists(path_to_dataset):
            download_and_organize(path_to_dataset=path_to_dataset, dataset_obj=test_dataset, filters=filters)

        json_file_path = os.path.join(path_to_dataset, 'json')
        i = 0
        while 1:
            logs_file_name = 'timer_logs_' + str(i) + '.conf'
            graph_file_name = 'precision_recall_' + str(i) + '.png'

            logger = init_logging(__name__, filename=logs_file_name)
            self.compute = precision_recall_compute()
            self.compute.add_dataloop_local_annotations(json_file_path)
            logger.info("running new execution")
            execution_obj = self.service.execute(function_name='search', execution_input=[self.configs_input],
                                                 project_id='72bb623f-517f-472b-ad69-104fed8ee94a')
            while execution_obj.latest_status['status'] != 'success':
                sleep(5)
                execution_obj = dl.executions.get(execution_id=execution_obj.id)
                if execution_obj.latest_status['status'] == 'failed':
                    raise Exception("plugin execution failed")
            logger.info("execution object status is successful")
            self.project.artifacts.download(package_name='zazuml',
                                            execution_id=execution_obj.id,
                                            local_path=os.getcwd())

            logger.info('artifact download finished')
            logger.info(str(os.listdir('.')))
            new_checkpoint_name = 'checkpoint_' + str(execution_obj.id) + '.pt'
            logger.info(str(os.listdir('.')))
            os.rename('checkpoint0.pt', new_checkpoint_name)
            logger.info(str(os.listdir('.')))
            new_checkpoint = torch.load(new_checkpoint_name, map_location=torch.device('cpu'))
            model_name = new_checkpoint['model_specs']['name']
            new_home_path = new_checkpoint['model_specs']['data']['home_path']

            model_obj = dl.models.get(model_name=model_name)
            adapter_temp = model_obj.build(local_path=os.getcwd())
            adapter_temp.load_inference(checkpoint_path=new_checkpoint_name)
            output_path = adapter_temp.predict(output_dir='new_checkpoint')
            logger.info('predictions in : ' + output_path)
            logger.info(os.listdir(output_path))
            self.compute.add_path_detections(output_path, model_name='new_checkpoint')
            logger.info(str(self.compute.by_model_name.keys()))
            if len(self.compute.by_model_name.keys()) < 2:
                # if the model cant predict anything then just skip it
                raise Exception('''model couldn't make any predictions''')

            new_checkpoint_mAP = self.compute.get_metric(model_name='new_checkpoint', precision_to_recall_ratio=1.)
            if 'check0' not in [checkp.name for checkp in model_obj.checkpoints.list()]:
                new_checkpoint = model_obj.checkpoints.upload(checkpoint_name='check0',
                                                              local_path=new_checkpoint_name)
                logger.info('uploaded this checkpoint as the new check0 : ' + new_checkpoint_name[:-3])
                from ObjectDetNet.prediction_deployment_stuff import do_deployment_stuff, create_trigger
                do_deployment_stuff(model_id=model_obj.id, checkpoint_id=new_checkpoint.id)
                logger.info('deployed prediction service')
                create_trigger()
                logger.info('deployed prediction trigger')
            else:
                new_checkpoint = model_obj.checkpoints.upload(checkpoint_name=new_checkpoint_name[:-3],
                                                              local_path=new_checkpoint_name)
                logger.info('uploaded this checkpoint : ' + new_checkpoint_name[:-3])
                best_checkpoint = model_obj.checkpoints.get('check0')
                check0_path = best_checkpoint.download(local_path=os.getcwd())
                adapter = model_obj.build(local_path=os.getcwd())
                adapter.load_inference(checkpoint_path=check0_path)
                output_path = adapter.predict(output_dir=best_checkpoint.name, home_path=new_home_path)
                self.compute.add_path_detections(output_path, model_name=best_checkpoint.name)
                best_checkpoint_mAP = self.compute.get_metric(model_name=best_checkpoint.name, precision_to_recall_ratio=1.)
                logger.info('best checkpoint: ' + str(best_checkpoint_mAP))
                logger.info('new checkpoint: ' + str(new_checkpoint_mAP))

            if new_checkpoint_mAP > best_checkpoint_mAP:
                predict_service = dl.services.get('predict')
                predict_service.input_params = {'model_id': model_obj.id,
                                                'checkpoint_id': new_checkpoint.id}
                predict_service.update()

                logger.info('switching with new checkpoint')

            self.compute.save_plot_metrics(save_path=graph_file_name)

            self.project.artifacts.upload(filepath=logs_file_name,
                                          package_name='zazuml',
                                          execution_id=execution_obj.id)
            self.project.artifacts.upload(filepath=graph_file_name,
                                          package_name='zazuml',
                                          execution_id=execution_obj.id)
            i += 1
        # logger.info('waiting ' + str(time_lapse) + ' for next execution . . . .')
