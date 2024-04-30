# When LTC is disabled in Torch-MLIR build, we will generate a dummy module to
# ensure that no import errors occur.

import sys
import os

if __name__ == "__main__":
    path = sys.argv[1]  # dummy script path
    file_name = sys.argv[2]  # dummy script

    contents = """
# This file was automatically generated due to LTC being disabled in build.

class LazyTensorCoreTestConfig:
    def __init__(self):
        assert False, "LTC is not enabled. Check the value of `TORCH_MLIR_ENABLE_LTC`"
    """

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file_name + ".py"), "w") as file:
        file.write(contents)
