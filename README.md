To install:

```python
uv venv
source .venv/bin/activate && uv pip install -e .
```

Preprocess data:

```
python arcagi/preprocess.py --data_dir=/Users/chris/code/ARC-AGI/data 
```

Check the dataloader works (which uses the preprocessed data)

```
python arcagi/data_loader.py
```