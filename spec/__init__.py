import os.path
import json


class SpecBase:
    def __init__(self):
        self._specData = {}

    def load(self, specData):
        if os.path.isfile(specData):
            with os.open(specData) as f:
                specData = json.load(f)
        self._specData = specData
        # validate indeed object was recieved


class Metric:
    pass


class hpValues:
    pass


class Trial:
    pass


class Oracle:
    pass


class ongoingTrial:
    pass
