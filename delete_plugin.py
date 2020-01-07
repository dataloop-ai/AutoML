import dtlpy as dl
dl.setenv('dev')
deployment = dl.projects.get(project_name='buffs_project').deployments.get(deployment_name="trial")
deployment.delete()

# or

plugin = dl.projects.get(project_name='buffs_project').plugins.get(plugin_name="trial")
plugin.delete()
print("erased . . . ")