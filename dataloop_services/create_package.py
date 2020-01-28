import dtlpy as dl
import os
import logging
logger = logging.getLogger(__name__)

project = dl.projects.get()

dataset_input = dl.FunctionIO(type='Dataset', name='dataset')
hp_value_input = dl.FunctionIO(type='Json', name='hp_values')
model_specs_input = dl.FunctionIO(type='Json', name='model_specs')
input_to_init = {
    'package_name': package_name
}

inputs = [dataset_input, hp_value_input, model_specs_input]

training_module = dl.PackageModule(entry_point='dataloop_services/service_executor.py',
                                   name='service_executor',
                                   functions=[dl.PackageFunction(name='run', inputs=inputs, outputs=[], description='')],
                                   init_inputs=input_to_init)
zazu_module = dl.PackageModule(entry_point='dataloop_services/zazu_module.py',
                               name='zazu_executer',
                               functions=[dl.PackageFunction(name='search', inputs=inputs, outputs=[], description=''),
                                          dl.PackageFunction(name='train', inputs=inputs, outputs=[], description=''),
                                          dl.PackageFunction(name='predict', inputs=inputs, outputs=[], description='')],
                               init_inputs=input_to_init)
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
                                       init_params=input_to_init)
