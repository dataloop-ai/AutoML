from .spec_base import Spec


class ConfigSpec(Spec):

    def validate(self):
        if 'max_instances_at_once' not in self.spec_data:
            raise Exception("Configs must have a max instances at once field")



