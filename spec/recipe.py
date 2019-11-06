from .spec_base import Spec


class RecipeSpec(Spec):

    def validate(self):
        if 'task' not in self.spec_data:
            raise Exception("Recipe must have a task field")

    @property
    def task(self):
        return self.spec_data['task']


