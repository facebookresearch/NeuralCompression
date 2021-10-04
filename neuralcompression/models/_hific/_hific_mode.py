import enum


class _HiFiCMode(enum.Enum):
    TRAIN = "train"
    VALIDATE = "validate"
    EVALUATE = "evaluate"
    TEST = "test"
