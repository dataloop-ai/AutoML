import dtlpy as dl
import os
import logging
logger = logging.getLogger(__name__)


def push_and_deploy_package(self, project, package_name):
    logger.info('dtlpy version:', dl.__version__)
    dataset_input = dl.FunctionIO(type='Dataset', name='dataset')
    hp_value_input = dl.FunctionIO(type='Json', name='hp_values')
    model_specs_input = dl.FunctionIO(type='Json', name='model_specs')
    init_specs_input = dl.FunctionIO(type='Json', name='package_name')
    input_to_init = {
        'package_name': package_name
    }

    inputs = [dataset_input, hp_value_input, model_specs_input]
    function = dl.PackageFunction(name='run', inputs=inputs, outputs=[], description='')
    module = dl.PackageModule(entry_point='dataloop_services/service_executor.py', name='service_executor',
                              functions=[function],
                              init_inputs=init_specs_input)

    package = project.packages.push(
        package_name=package_name,
        src_path=os.getcwd(),
        modules=[module])

    logger.info('deploying package . . .')
    self.service = package.services.deploy(service_name=package.name,
                                           module_name='service_executor',
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
