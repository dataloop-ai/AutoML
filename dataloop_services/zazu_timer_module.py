import logging
import dtlpy as dl
import json
import torch
import os
from time import sleep
from dataloop_services.plugin_utils import maybe_download_pred_data, download_and_organize
from dataloop_services import deploy_predict_item, create_trigger
from eval import precision_recall_compute
from logging_utils import logginger, init_logging
import logging

logger = logging.getLogger(__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, configs, time, test_dataset_id, query):
        logger.info('dtlpy version: ' + str(dl.__version__))
        logger.info('dtlpy info: ' + str(dl.info()))
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
        self.model_obj = self.project.models.get(model_name='object_detection')
        self.adapter = self.model_obj.build(local_path=os.getcwd())
        logger.info('model built')
        while 1:

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
            logs_file_name = 'timer_logs_' + str(execution_obj.id) + '.conf'
            graph_file_name = 'precision_recall_' + str(execution_obj.id) + '.png'
            self.cycle_logger = init_logging(__name__, filename=logs_file_name)
            logger.info('artifact download finished')
            logger.info(str(os.listdir('.')))

            # load new checkpoint and change to unique name
            new_checkpoint_name = 'checkpoint_' + str(execution_obj.id) + '.pt'
            logger.info(str(os.listdir('.')))

            os.rename('checkpoint0.pt', new_checkpoint_name)
            new_model_name = new_checkpoint_name[:-3]
            logger.info(str(os.listdir('.')))
            new_checkpoint = torch.load(new_checkpoint_name, map_location=torch.device('cpu'))
            # self.model_obj = self.project.models.get(model_name=new_checkpoint['model_specs']['name'])
            # self.adapter = self.model_obj.build(local_path=os.getcwd())
            # logger.info('model built')
            self.new_home_path = new_checkpoint['model_specs']['data']['home_path']

            self._compute_predictions(checkpoint_path=new_checkpoint_name,
                                      model_name=new_model_name)

            if len(self.compute.by_model_name.keys()) < 2:
                # if the model cant predict anything then just skip it
                logger.info('''model couldn't make any predictions, trying to train again''')
                continue

            # if previous best checkpoint doesnt exist there must not be a service, launch prediction service with new
            # new_checkpoint and create trigger

            if 'check0' not in [checkp.name for checkp in self.model_obj.checkpoints.list()]:
                logger.info('there is no check0, will add upload new checkpoint as check0 and '
                            'deploy prediction service')
                new_checkpoint_obj = self.model_obj.checkpoints.upload(checkpoint_name='check0',
                                                                       local_path=new_checkpoint_name)
                logger.info('uploaded this checkpoint as the new check0 : ' + new_checkpoint_name[:-3])

                self._maybe_launch_predict(new_checkpoint_obj)
                continue
            logger.info('i guess check0 does exist')
            best_checkpoint = self.model_obj.checkpoints.get('check0')
            check0_path = best_checkpoint.download(local_path=os.getcwd())
            logger.info('downloading best checkpoint')
            logger.info(str(os.listdir('.')))
            logger.info('check0 path is: ' + str(check0_path))
            self._compute_predictions(checkpoint_path=check0_path, model_name=best_checkpoint.name)

            # compute metrics
            new_checkpoint_mAP = self.compute.get_metric(model_name=new_model_name, precision_to_recall_ratio=1.)
            best_checkpoint_mAP = self.compute.get_metric(model_name=best_checkpoint.name, precision_to_recall_ratio=1.)
            logger.info('best checkpoint: ' + str(best_checkpoint_mAP))
            logger.info('new checkpoint: ' + str(new_checkpoint_mAP))

            # if new checkpoint performs better switch out prediction
            if new_checkpoint_mAP > best_checkpoint_mAP:
                logger.info('new checkpoint is better')
                logger.info('uploading old best checkpoint under new name')
                self.model_obj.checkpoints.upload(checkpoint_name='checkpoint_' + check0_path.split('_')[-1][:-3],
                                                  local_path=check0_path)
                logger.info('deleting old best checkpoint')
                best_checkpoint.delete()
                logger.info('uploading new best checkpoint as check0')
                new_best_checkpoint_obj = self.model_obj.checkpoints.upload(checkpoint_name='check0',
                                                                            local_path=new_checkpoint_name)
                if 'predict' not in [s.name for s in dl.services.list()]:
                    self._maybe_launch_predict(new_best_checkpoint_obj)
                else:
                    self._update_predict_service(new_best_checkpoint_obj)
                logger.info('switched with new checkpoint')

            self.compute.save_plot_metrics(save_path=graph_file_name)

            self.project.artifacts.upload(filepath=logs_file_name,
                                          package_name='zazuml',
                                          execution_id=execution_obj.id)
            self.project.artifacts.upload(filepath=graph_file_name,
                                          package_name='zazuml',
                                          execution_id=execution_obj.id)
            logger.info('waiting ' + str(time) + ' seconds for next execution . . . .')
            sleep(time)


    def _compute_predictions(self, checkpoint_path, model_name):

        self.adapter.load_inference(checkpoint_path=checkpoint_path)
        logger.info('checkpoint loaded')
        output_path = self.adapter.predict(output_dir=model_name, home_path=self.new_home_path)
        logger.info('predictions in : ' + output_path)
        logger.info(os.listdir(output_path))
        self.compute.add_path_detections(output_path, model_name=model_name)
        logger.info(str(self.compute.by_model_name.keys()))

    def _maybe_launch_predict(self, new_checkpoint_obj):
        if 'predict' not in [s.name for s in dl.services.list()]:
            logger.info('predict service doesnt exist, about to deploy prediction service')
            package_obj = dl.packages.get('zazuml')
            deploy_predict_item(package=package_obj,
                                model_id=self.model_obj.id,
                                checkpoint_id=new_checkpoint_obj.id)
            logger.info('service deployed')
            logger.info('deployed prediction service')
            create_trigger()
            logger.info('created prediction trigger')
        else:
            logger.info('predict service exists, no reason to relaunch')

    def _update_predict_service(self, new_best_checkpoint_obj):
        logger.info('update predict service')
        predict_service = dl.services.get('predict')
        logger.info('service: ' + str(predict_service))
        predict_service.input_params = {'model_id': self.model_obj.id,
                                        'checkpoint_id': new_best_checkpoint_obj.id}
        predict_service.update()
        logger.info('service: ' + str(predict_service))
