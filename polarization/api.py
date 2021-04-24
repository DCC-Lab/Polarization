# Changes to implement and expected API

# JonesVector k dependence, done.
# JonesMatrix dependent on k, done: any product with JonesVector will work
    
class TissueLayer:

    def propagateThrough(self, vector) -> 'JonesVector':
        pass

    def propagateManyThrough(self, vectors) -> ['JonesVector']:
        pass

    def backscatter(self, vector) -> 'JonesVector':
        pass

    def backscatterMany(self, vectors) -> ['JonesVector']:
        pass

class TissueStack:

    self.tissueLayers = []

    def propagateThrough(self, vector) -> 'JonesVector':
        pass

    def propagateManyThrough(self, vectors) -> ['JonesVector']:
        pass

    def backscatter(self, vector) -> 'JonesVector':
        pass

    def backscatterMany(self, vectors) -> ['JonesVector']:
        pass

    def __iter__(self):
        return iter(self.tissueLayers)


class Pulse:
    def __init__(self, vectors = None, centerWavelength = None):
        self.vectors = vectors
    # must behave list-like for JonesVectors

    def __iter__(self):
        return iter(self.vectors)

    def __next__(self):
        pass

    def __len__(self):
        pass




