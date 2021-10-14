from ._continuous_entropy import ContinuousEntropy


class ContinuousIndexedEntropy(ContinuousEntropy):
    @property
    def probability_tables(self):
        raise NotImplementedError

    def compress(self, **kwargs):
        raise NotImplementedError

    def decompress(self, **kwargs):
        raise NotImplementedError
