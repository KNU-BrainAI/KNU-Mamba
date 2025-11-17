## API Usage

### Training (`train_vla`)
- Mirrors `python run.py --benchmark_type libero_object` but can be called directly from Python.
- Example:
```
from MambaVLA import train_vla

train_vla(
    config_file="MambaVLA/configs/config.yaml",
    dataset_path="/abs/path/to/datasets/libero_object",
    output_path="./runs/libero_object/latest/final_model.pth",
)
```
- `config_file` points to the dataclass-based YAML; override only the fields you need.
- Only a `.pth` checkpoint is saved, matching the behavior of `run.py`.

### Inference (`infer_vla`)
- Generates actions from a trained checkpoint and a language instruction.
```
from MambaVLA import infer_vla

action = infer_vla(
    pth="./runs/libero_object/latest/final_model.pth",
    language="pick up the red block",
    config_file="MambaVLA/configs/config.yaml",
)
print(action)
```
- Pass real observation tensors via `observations` for online control; if omitted, dummy batches are used.