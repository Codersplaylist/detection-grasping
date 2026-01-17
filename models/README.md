# Model Directory

This directory should contain your trained YOLO model weights.

## Required File

Place your trained model file here:
- **Filename**: `my_model.pt` (or update the path in `config.yaml`)
- **Source**: Download from your Google Colab training session

## How to Get Your Model

From your Colab notebook (`untitled26.py`), the model was saved as:
```
/content/my_model.zip
```

This zip file contains:
- `my_model.pt` - The trained model weights (this is what you need)
- `train/` - Training results and metrics

### Steps:
1. In your Colab notebook, locate the "Files" panel (left sidebar)
2. Navigate to `/content/`
3. Download `my_model.zip`
4. Extract the zip file
5. Copy `my_model.pt` to this directory

## Verify Model

Once you've placed the model file, you can verify it works:

```bash
python test_inference.py
```

This will test model loading and run inference on your test images.
