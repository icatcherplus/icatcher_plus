import pytest
import numpy as np
import icatcher.draw

def test_imports():
    """
    tests that the environment has the dependencies installed
    @return:
    """
    import torch
    import numpy as np
    import pooch
    import PIL

def test_mask():
    image = np.random.random((256, 512, 3))
    masked = icatcher.draw.mask_regions(image, 0, 128, 0, 256)
    assert masked[:128, 256:, :].all() == 0
