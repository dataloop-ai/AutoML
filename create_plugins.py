import dtlpy as dl
import os


def create_tuner():
    dl.setenv('dev')

    plugin_name = 'tuner'
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
                                           config={'project_id': project.id,
                                                   'plugin_name': plugin.name},
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
