import dtlpy as dl
import os
import logging

logger = logging.getLogger(__name__)


def deploy_predict(package):
    input_to_init = {
        'package_name': package.name,
    }

    logger.info('deploying package . . .')
    service_obj = package.services.deploy(service_name='predict',
                                          module_name='predict_module',
                                          package=package,
                                          runtime={'gpu': True,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                   },
                                          execution_timeout=60 * 60 * 1e10,
                                          init_input=input_to_init)

    return service_obj

def deploy_model(package):
    input_to_init = {
        'package_name': package.name,
    }

    logger.info('deploying package . . .')
    service_obj = package.services.deploy(service_name='trial',
                                          module_name='models_module',
                                          package=package,
                                          runtime={'gpu': True,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                   },
                                          execution_timeout=60 * 60 * 1e10,
                                          init_input=input_to_init)

    return service_obj


def deploy_zazu(package):
    input_to_init = {
        'package_name': package.name
    }

    logger.info('deploying package . . .')
    service_obj = package.services.deploy(service_name='zazu',
                                          module_name='zazu_module',
                                          package=package,
                                          runtime={'gpu': False,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                   },
                                          execution_timeout=60 * 60 * 1e10,
                                          init_input=input_to_init)

    return service_obj

def deploy_zazu_timer(package, init_inputs):

    logger.info('deploying package . . .')
    service_obj = package.services.deploy(service_name='timer',
                                          module_name='zazu_timer_module',
                                          package=package,
                                          runtime={'gpu': False,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                   },
                                          execution_timeout=60*60*1e10,
                                          init_input=init_inputs)

    return service_obj

def push_package(project):
    dataset_input = dl.FunctionIO(type='Dataset', name='dataset')
    train_query_input = dl.FunctionIO(type='Json', name='train_query')
    val_query_input = dl.FunctionIO(type='Json', name='val_query')
    hp_value_input = dl.FunctionIO(type='Json', name='hp_values')
    model_specs_input = dl.FunctionIO(type='Json', name='model_specs')
    checkpoint_path_input = dl.FunctionIO(type='Json', name='checkpoint_path')
    package_name_input = dl.FunctionIO(type='Json', name='package_name')

    configs_input = dl.FunctionIO(type='Json', name='configs')
    time_input = dl.FunctionIO(type='Json', name='time')
    test_dataset_input = dl.FunctionIO(type='Json', name='test_dataset_id')
    query_input = dl.FunctionIO(type='Json', name='query')

    predict_inputs = [dataset_input, val_query_input, checkpoint_path_input, model_specs_input]
    model_inputs = [dataset_input, train_query_input, val_query_input, hp_value_input, model_specs_input]
    zazu_inputs = [configs_input]

    predict_function = dl.PackageFunction(name='run', inputs=predict_inputs, outputs=[], description='')
    model_function = dl.PackageFunction(name='run', inputs=model_inputs, outputs=[], description='')
    zazu_search_function = dl.PackageFunction(name='search', inputs=zazu_inputs, outputs=[], description='')
    zazu_predict_function = dl.PackageFunction(name='predict', inputs=zazu_inputs, outputs=[], description='')
    timer_update_function = dl.PackageFunction(name='update_time', inputs=time_input, outputs=[], description='')

    predict_module = dl.PackageModule(entry_point='dataloop_services/predict_module.py',
                                     name='predict_module',
                                     functions=[predict_function],
                                     init_inputs=[package_name_input])

    models_module = dl.PackageModule(entry_point='dataloop_services/trial_module.py',
                                     name='models_module',
                                     functions=[model_function],
                                     init_inputs=[package_name_input])

    zazu_module = dl.PackageModule(entry_point='dataloop_services/zazu_module.py',
                                   name='zazu_module',
                                   functions=[zazu_search_function, zazu_predict_function],
                                   init_inputs=package_name_input)

    zazu_timer_module = dl.PackageModule(entry_point='dataloop_services/zazu_timer_module.py',
                                         name='zazu_timer_module',
                                         functions=[timer_update_function],
                                         init_inputs=[configs_input, time_input, test_dataset_input, query_input])

    package_obj = project.packages.push(
        package_name='zazuml',
        src_path=os.getcwd(),
        modules=[predict_module, models_module, zazu_module, zazu_timer_module])

    return package_obj


def update_service(project, service_name):
    package_obj = project.packages.get('zazuml')
    service = project.services.get(service_name)
    service.package_revision = package_obj.version
    service.update()

