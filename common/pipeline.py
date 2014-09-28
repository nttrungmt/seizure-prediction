import numpy as np

class Pipeline(object):
    """
    A Pipeline is an object representing the data transformations to make
    on the input data, finally outputting extracted features.

    gen_ictal: Whether ictal data generation should be used for this pipeline

    pipeline: List of transforms to apply one by one to the input data
    """
    def transforms_name(self, transforms):
        names = [t.get_name() for t in transforms]
        if self.gen_ictal:
            if isinstance(self.gen_ictal,bool) or self.gen_ictal==1:
                names = ['gen'] + names
            else:
                names = ['gen%g'%self.gen_ictal] + names
        return 'empty' if len(names) == 0 else '_'.join(names)

    def __init__(self, gen_ictal, pipeline):
        self.transforms = pipeline
        self.gen_ictal = gen_ictal
        self.name = self.transforms_name(self.transforms)

    def get_name(self):
        return self.name

    def apply(self, data):
        """this method is called for every window in the data (unless it is cached)"""
        for transform in self.transforms:
            data = transform.apply(data)
        return data

class UnionPipeline(Pipeline):
    def get_name(self):
        return 'union_' + self.name
