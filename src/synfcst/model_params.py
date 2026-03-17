import json


class ModelParams:
    """ class wraps the model params file used to run the generation process

        represents a single fitted model from which generation can be done; intended to be used for multiple purposes
    """
    def __init__(self, filename:str):
        """ constructor to read model params file
        """
        data = json.load(open(filename, 'r'))
        # TODO: add validation, remove this block!
        for k,v in data.items():
            # set fields, simple for now, but we should replace this with some validation steps
            setattr(self, k, v)