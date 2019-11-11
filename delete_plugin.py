import dtlpy as dl
deployment = dl.projects.get(project_id="fcdd792b-5146-4c62-8b27-029564f1b74e").deployments.get(deployment_name="thisdeployment")
d = dl.plugins.delete()
if d:
    print("erased . . . ")