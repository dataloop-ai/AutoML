from .spec_base import Spec
from .recipe_spec import RecipeSpec


class ModelSpec(RecipeSpec):

    def __init__(self, spec_data=None):
        if not spec_data:
            spec_data = {"model_space": (10, 0, 0)}
        super().__init__(spec_data)

    def validate(self):
        if 'model_space' not in self.__dir__():
            raise Exception("Model spec must have a model_space field")

    def add_attr(self, value, name):
        setattr(self, name, value)

    def add_attr_from_obj(self, obj, name):
        setattr(self, name, getattr(obj, name))