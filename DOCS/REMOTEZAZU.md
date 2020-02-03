## *RUNNING ZAZUML REMOTELY VIA OUR DATALOOP CLOUD SERVICE*

## Run in TERMINAL

### *Model & Hyper-Parameter Search*
```
python zazu.py --search --remote
```
### *Train*
```
python zazu.py --train --remote
```
### *Predict*
```
python zazu.py --predict --remote
```

## Run in PYTHON
```
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