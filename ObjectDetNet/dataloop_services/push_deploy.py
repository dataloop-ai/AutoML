import dtlpy as dl
import os


def deploy_predict_item(package, model_id, checkpoint_id):
    input_to_init = {'model_id': model_id,
                     'checkpoint_id': checkpoint_id}

    service_obj = package.services.deploy(service_name='predict',
                                          module_name='predict_item_module',
                                          package=package,
                                          runtime={'gpu': True,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3',
                                                   'podType': 'gpu-k80-m'
                                                   },
                                          is_global=True,
                                          execution_timeout=60 * 60 * 1e10,
                                          init_input=input_to_init)

    return service_obj


def push_package(project):
    item_input = dl.FunctionIO(type='Item', name='item')
    model_input = dl.FunctionIO(type='Json', name='model_id')
    checkpoint_input = dl.FunctionIO(type='Json', name='checkpoint_id')

    predict_item_function = dl.PackageFunction(name='predict_single_item', inputs=[item_input], outputs=[],
                                               description='')
    load_checkpoint_function = dl.PackageFunction(name='load_new_inference_checkpoint',
                                                  inputs=[model_input, checkpoint_input], outputs=[],
                                                  description='')

    predict_item_module = dl.PackageModule(entry_point='prediction_module.py',
                                           name='predict_item_module',
                                           functions=[predict_item_function, load_checkpoint_function],
                                           init_inputs=[model_input, checkpoint_input])
    module_path = os.path.join(os.getcwd(), 'dataloop_services')
    package_obj = project.packages.push(
        package_name='ObDetNet',
        src_path=module_path,
        modules=[predict_item_module])

    return package_obj
