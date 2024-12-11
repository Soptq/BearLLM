<div align="center">
<a href="https://www.python.org/">
<img src="./docs/images/logo.svg" width="200" alt="logo"/>
</a>
<h1>Rotating Components Health Management</h1>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/Pytorch-latest-orange"></a>
<a href="https://arxiv.org/abs/2408.11281"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-B31B1B"></a>
<a href="https://huggingface.co/datasets/RotCHM/MRCHM"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-🤗-FFFDF5"></a>
![GitHub Repo stars](https://img.shields.io/github/stars/RotCHM/RotCHM)
</div>

## 🔥 NEWS
- **[2024-12-11]** ⏫ We are now working on making the code of BearLLM public. Stay tuned!
- **[2024-12-10]** 🎉 The BearLLM paper is accepted by the Thirty-Ninth AAAI Conference on Artificial Intelligence ([AAAI-25](https://aaai.org/conference/aaai/aaai-25/)).
- **[2024-11-20]** 🔼 The MBHM dataset is upgraded to the MRCHM dataset. Check the [dataset page](https://huggingface.co/datasets/RotCHM/MRCHM) for more details.
- **[2024-08-21]** 📝 The preprint of the BearLLM paper is available on arXiv. Check the [paper page](https://arxiv.org/abs/2408.11281) for more details.

## 📅 TODO
- [ ] Upload the health management corpus of the MRCHM dataset.
- [x] Collect the codes for pre-training and fine-tuning BearLLM.
- [x] Collect the codes of BearLLM's classification network and other comparison models.
- [x] Upload the vibration signal portion of the MRCHM dataset.

## 📚 Introduction
The [MRCHM](https://github.com/RotCHM/RotCHM/tree/main/MRCHM) dataset is the first multimodal dataset designed for the study of rotating equipment health management. It is divided into two parts: vibration signals and a health management corpus. The vibration signals and condition information are derived from 13 publicly available datasets, covering rotating components such as bearings and gears, and are still under continuous updating and improvement. The thousands of working conditions pose more difficult challenges for the identification model and better represent real-world usage scenarios.

[BearLLM](https://github.com/RotCHM/RotCHM/tree/main/BearLLM) is a prior knowledge-enhanced bearing health management framework with a unified vibration signal representation. This framework transforms the signal to be tested into the frequency domain, enabling effective identification of spectral differences compared to the vibration signal under fault-free conditions. By aligning the vibration signal with the fault semantic embedding, we achieve a unified natural language response for various health management tasks through a fine-tuned language model with low computational overhead. Experiments demonstrate that this framework achieves leading performance under thousands of working conditions.

[RoHMF](https://github.com/RotCHM/RotCHM/tree/main/RoHMF) is the next-generation rotating component health management framework we developed. This framework expands the application scenarios of BearLLM, capable of handling more component types and more health management tasks without the need for fault-free signals as a reference. More relevant information will be made public after the paper is accepted.

## 📖 Citation
Please cite the following paper if you use this study in your research:

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