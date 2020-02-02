import dtlpy as dl
import os
import logging

logger = logging.getLogger(__name__)


def deploy_model(package, service_name):
    input_to_init = {
        'package_name': package.name,
        'service_name': service_name

    }

    logger.info('deploying package . . .')
    service_obj = package.services.deploy(service_name=service_name,
                                          module_name='models_module',
                                          agent_versions={
                                              'dtlpy': '1.9.7',
                                              'runner': '1.9.7.latest',
                                              'proxy': '1.9.7.latest',
                                              'verify': True
                                          },
                                          package=package,
                                          runtime={'gpu': True,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                   },
                                          init_input=input_to_init)

    return service_obj


def deploy_zazu(package):
    input_to_init = {
        'package_name': package.name
    }

    logger.info('deploying package . . .')
    service_obj = package.services.deploy(service_name='zazu',
                                          module_name='zazu_module',
                                          agent_versions={
                                              'dtlpy': '1.9.7',
                                              'runner': '1.9.7.latest',
                                              'proxy': '1.9.7.latest',
                                              'verify': True
                                          },
                                          package=package,
                                          runtime={'gpu': False,
                                                   'numReplicas': 1,
                                                   'concurrency': 2,
                                                   'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                   },
                                          init_input=input_to_init)

    return service_obj


def push_package(project):
    dataset_input = dl.FunctionIO(type='Dataset', name='dataset')
    hp_value_input = dl.FunctionIO(type='Json', name='hp_values')
    model_specs_input = dl.FunctionIO(type='Json', name='model_specs')

    package_name_input = dl.FunctionIO(type='Json', name='package_name')
    service_name_input = dl.FunctionIO(type='Json', name='service_name')

    configs_input = dl.FunctionIO(type='Json', name='configs')

    model_inputs = [dataset_input, hp_value_input, model_specs_input]
    zazu_inputs = [configs_input]

    model_function = dl.PackageFunction(name='run', inputs=model_inputs, outputs=[], description='')
    train_function = dl.PackageFunction(name='train', inputs=zazu_inputs, outputs=[], description='')
    search_function = dl.PackageFunction(name='search', inputs=zazu_inputs, outputs=[], description='')

    models_module = dl.PackageModule(entry_point='dataloop_services/service_executor.py',
                                     name='models_module',
                                     functions=[model_function],
                                     init_inputs=[package_name_input, service_name_input])

    zazu_module = dl.PackageModule(entry_point='dataloop_services/zazu_module.py',
                                   name='zazu_module',
                                   functions=[train_function, search_function],
                                   init_inputs=package_name_input)

    package_obj = project.packages.push(
        package_name='zazuml',
        src_path=os.getcwd(),
        modules=[models_module, zazu_module])

    return package_obj


def update_service(project, service_name):
    package_obj = project.packages.get('zazuml')
    service = project.services.get(service_name)
    service.package_revision = package_obj.version
    service.update()

