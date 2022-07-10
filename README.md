# CPP/CUDA extension of 3DLUT Interpolation operators In Pytorch

## Declaration
An example of 3Dlut operator extensin for Pytorch. This repository inherits some code from repository [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT). The original repository contain some bugs, and they are fixed here.

## Usage
### Requirements
Python >= 3.8

Pytorch <= 1.11.0 (Don't use version >= 1.12, there are some incompatibility from the compiler. Ignore if you can solve this problem)

### Build
Two interpolation operators implemented here - trilinear and tetrahedral interpolation. Compile them before using with:

```
cd trilinear_interp  
./setup.sh
cd ../tetrahedral_interp
./setup.sh
```

### Run demo
test.py shows how to apply 3Dlut to the demo image with this implementation.
```
python3 test.py
```
test.ipynb jupyter notebook shows how to train a custom 3Dlut, you can use it to verify the correctness of the implementation interactively.