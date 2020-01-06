import dtlpy as dl
dl.setenv('dev')
deployment = dl.projects.get(project_name='buffs_project').deployments.get(deployment_name="trial")
d = dl.plugins.delete()
if d:
    print("erased . . . ")