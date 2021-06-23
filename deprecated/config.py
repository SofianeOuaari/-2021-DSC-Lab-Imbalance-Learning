import warnings

# THIS FILE CONTAINS THE DEPRECATED CODE FOR TRAINING THE MODELS
warnings.warn("This module is deprecated in favour of machine_learning", DeprecationWarning, stacklevel=2)

# BEST
EPOCH = 5
STEP = 15
BATCH_SIZE = 128

baseline = dict(
    lr=[0.0001],
    decay=[0.2],
    optimizer=['Adam'],
    loss=['CE'],
    ratios=[1],
    model_type=['MLP1'],
    strategy=['under'],
)
