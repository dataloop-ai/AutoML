# *Running ZazuML remotely via our Dataloop cloud service*

### *Set Configs*
```
{
  "max_trials": 1,
  "max_instances_at_once": 1,
  "model_priority_space": [
    10,
    0,
    0
  ],
  "task": "detection",
  "dataloop": {
    "project": "buffs_project",
    "dataset": "tiny_mice_data"
  }
}
```

## *Run in TERMINAL*

### *model & hyper-parameter search*
```
python zazu.py --search --remote
```
### *train*
```
python zazu.py --train --remote
```
### *predict*
```
python zazu.py --predict --remote
```

## *Run in PYTHON*
```
import json
import dtlpy as dl


# Load the configs file
with open('configs.json', 'r') as fp:
    configs = json.load(fp)

# Load configs input
configs_input = dl.FunctionIO(type='Json', name='configs', value=configs)
inputs = [configs_input]

# Get the remote Zazu service
zazu_service = dl.services.get('zazu')


# Run model & hyper-parameter search
zazu_service.execute(function_name='search', execution_input=[configs_input])


# Train
zazu_service.execute(function_name='train', execution_input=inputs)


# Predict
zazu_service.execute(function_name='predict', execution_input=inputs)
```

Check out more details [@Dataloop AI](https://dataloop.ai/)!