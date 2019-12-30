import dtlpy as dl
import pprint
import time
import os


def create_tuner():
    dl.setenv('prod')

    plugin_name = 'tuner'
    project_name = 'ZazuProject'

    project = dl.projects.get(project_name=project_name)

    ###############
    # push plugin #
    ###############
    plugin = project.plugins.push(plugin_name=plugin_name,
                                  src_path=os.getcwd(),
                                  inputs=[{"type": "Json",
                                           "name": "configs"}])

    plugin = project.plugins.get(plugin_name=plugin_name)

    #####################
    # create deployment #
    #####################
    deployment = plugin.deployments.deploy(deployment_name=plugin.name,
                                           plugin=plugin,
                                           runtime={'gpu': False,
                                                    'numReplicas': 1,
                                                    'concurrency': 2,
                                                    'image': 'gcr.io/viewo-g/piper/agent/runner/gpu/main/zazu:latest'
                                                    },
                                           bot=None)
    deployment = plugin.deployments.get(deployment_name=plugin.name)

    ##############
    # for update #
    ##############
    deployment.pluginRevision = plugin.version
    deployment.update()

    ####################
    # invoke a session #
    ####################
    inputs = [dl.PluginInput(type='Json',
                             value={'annotation_type': 'binary',
                                    'confidence_th': 0.50,
                                    'output_action': 'annotations'},
                             name='config')]
    session = deployment.sessions.create(deployment_id=deployment.id,
                                         session_input=inputs)

    # check updates
    for i in range(5):
        _session = deployment.sessions.get(session_id=session.id)
        pprint.pprint(_session.status[-1])
        time.sleep(2)


def create_trainer():
    dl.setenv('dev')

    plugin_name = 'trainer'
    project_name = 'ZazuProject'

    project = dl.projects.get(project_name=project_name)
    dl.projects.checkout(project_name)
    dl.plugins.checkout(plugin_name)

    ###############
    # push plugin #
    ###############
    plugin = project.plugins.push(plugin_name=plugin_name,
                                  src_path=os.getcwd())

    plugin = project.plugins.get(plugin_name=plugin_name)

    #####################
    # create deployment #
    #####################
    deployment = plugin.deployments.deploy(deployment_name=plugin.name,
                                           plugin=plugin,
                                           runtime={'gpu': False,
                                                    'numReplicas': 1,
                                                    'concurrency': 2,
                                                    })
    deployment = plugin.deployments.get(deployment_name=plugin.name)

    ##############
    # for update #
    ##############
    deployment.pluginRevision = plugin.version
    deployment.update()
