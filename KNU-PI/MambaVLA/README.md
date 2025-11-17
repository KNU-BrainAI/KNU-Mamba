## MambaVLA

This repository packages the MambaVLA training and inference stack.

### Setup (conda)
```
conda create -n mambavla python=3.10 -y
conda activate mambavla
pip install --upgrade pip
pip install -r requirements.txt
```
or
```
pip install -e .
```

### Test command
```
python run.py
```

### API Entry Points
Alternatively you can call the Python APIs directly:
```
from mambavla import train_vla, infer_vla
```

