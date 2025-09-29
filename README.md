<div align="center">

# Towards Generalized Video Quality Assessment: A Weak-to-Strong Learning Paradigm


<div align="left">

## Installation

```bash
conda create -n w2s_vqa python=3.10
conda activate w2s_vqa
pip install -r requirements.txt
```

## Training
- **SlowFast feature extraction**

```bash
cd slowfast_feature
python extract_slowfast_feature.py --feature_save_folder path/to/save_features --videos_dir path/to/videos
```

- **Train the model**

📄 **Note**: The format of the dataset JSON file should follow the example in  
[`example.json`](example.json)

Before training, make sure to **modify the file paths** in the script according to your local environment.
```bash
bash scripts/train/finetune_onevision.sh
```

## Quick Inference

```shell
python infer_pair.py --model-path path/to/save_weights
```

