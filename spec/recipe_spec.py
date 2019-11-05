from .spec_base import Spec


class RecipeSpec(Spec):

    def validate(self):
        if 'task' not in self.__dir__():
            raise Exception("Recipe must have a task field")



