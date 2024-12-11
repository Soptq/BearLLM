<div align="center">
<a href="https://www.python.org/">
<img src="../docs/images/logo.svg" width="200" alt="logo"/>
</a>
<h1>Multi-modal Rotating Components Health Management Dataset</h1>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/Pytorch-latest-orange"></a>
<a href="https://arxiv.org/abs/2408.11281"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-B31B1B"></a>
<a href="https://huggingface.co/datasets/RotCHM/MRCHM"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-🤗-FFFDF5"></a>
![GitHub Repo stars](https://img.shields.io/github/stars/RotCHM/RotCHM)
</div>

## ⚡️ Download

Due to the capacity limitation of GitHub, please download the data file on [huggingface](https://huggingface.co/datasets/RotCHM/MRCHM).

## 📚 Introduction
The [MRCHM](https://github.com/RotCHM/RotCHM/tree/main/MRCHM) dataset is the first multimodal dataset designed for the study of rotating equipment health management. It is divided into two parts: vibration signals and a health management corpus. The vibration signals and condition information are derived from 13 publicly available datasets, covering rotating components such as bearings and gears, and are still under continuous updating and improvement. The thousands of working conditions pose more difficult challenges for the identification model and better represent real-world usage scenarios.

In the dataset, vibration signals from different datasets have been converted to the same length (24000) by Discrete Cosine Normalization (DCN). For more information about the implementation of DCN, please refer to the [paper](https://arxiv.org/abs/2408.11281) or [code](https://github.com/RotCHM/RotCHM/blob/main/MRCHM/dcn.py).

## 💻 Demo

We provide a demo script to show how to load the MRCHM dataset and output the data shape. Please check the [demo](https://github.com/RotCHM/RotCHM/blob/main/MRCHM/demo.py) for more details.

## 📖 Citation
Please cite the following paper if you use this dataset in your research:

```
@misc{peng2024bearllmpriorknowledgeenhancedbearing,
      title={BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation}, 
      author={Haotian Peng and Jiawei Liu and Jinsong Du and Jie Gao and Wei Wang},
      year={2024},
      eprint={2408.11281},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.11281}, 
}
```