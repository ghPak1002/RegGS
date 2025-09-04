<p align="center">

  <h2 align="center">RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration</h2>
  <p align="center">
    <strong>Chong Cheng</strong><sup>1*</sup>
    ·
    <strong>Yu Hu</strong><sup>1*</sup>
    ·
    <strong>Sicheng Yu</strong><sup>1</sup>
    ·
    <strong>Beizhen Zhao</strong><sup>1</sup>
    ·
    <strong>Zijian Wang</strong><sup>1</sup>
    ·
    <strong>Hao Wang</strong><sup>1†</sup>
</p>

<p align="center"><strong>International Conference on Computer Vision, ICCV 2025</strong></a>
<p align="center">
    <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)
</p>
   <h3 align="center">

   [![arXiv](https://img.shields.io/badge/arXiv-2507.08136-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2507.08136) [![ProjectPage](https://img.shields.io/badge/Project_Page-RegGS-blue)](https://3dagentworld.github.io/reggs/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  <div align="center"></div>
</p>

## 🛠️ Setup
The code has been tested on systems with:
- Ubuntu 22.04 LTS
- Python 3.10.18
- CUDA 11.8
- NVIDIA GeForce RTX 3090 or A6000

## 📦 Repository

Clone the repo with `--recursive` because we have submodules:

```
git clone https://github.com/3DAgentWorld/RegGS.git --recursive
cd RegGS
```

## 💻 Installation

### Python Environment

This codebase has been successfully tested with Python 3.10, CUDA 11.8, and PyTorch 2.5.1. We recommend installing the dependencies in a virtual environment such as Anaconda.

1.  Install main libraries: 
    ```bash
    conda env create -f environment.yaml

    conda activate reggs

    pip install -r requirements.txt
    ```

2.  Install thirdparty submodules:
    ```bash
    pip install thirdparty/diff-gaussian-rasterization-w-pose

    pip install thirdparty/gaussian_rasterizer

    pip install thirdparty/simple-knn
    ```

3.  Compile the cuda kernels for RoPE (as in CroCo v2):
    ```bash
    cd src/noposplat/model/encoder/backbone/croco/curope
    python setup.py build_ext --inplace
    ```

4. If you encounter cannot import torch. add option `--no-build-isolation` to `pip install`

### Downloading Pretrained Checkpoints

Download NoPoSplat [re10k](https://huggingface.co/botaoye/NoPoSplat/resolve/main/re10k.ckpt) checkpoints and [acid](https://huggingface.co/botaoye/NoPoSplat/resolve/main/acid.ckpt) checkpoints to `./pretrained_weights` directory, run:

  ```bash
  wget -c https://huggingface.co/botaoye/NoPoSplat/resolve/main/re10k.ckpt -P ./pretrained_weights

  wget -c https://huggingface.co/botaoye/NoPoSplat/resolve/main/acid.ckpt -P ./pretrained_weights
  ```

### Run RegGS on re10k sample

The preprocessed re10k data is placed in the directory `./sample_data`. To run RegGS on sample data, run:
1. The inference stage: \
`CUDA_VISIBLE_DEVICES=0 python3 run_infer.py config/re10k.yaml`
2. The refinement stage: \
`CUDA_VISIBLE_DEVICES=0 python3 run_refine.py --checkpoint_path output/re10k/000c3ab189999a83`
3. The evaluation stage: \
`CUDA_VISIBLE_DEVICES=0 python3 run_metric.py --checkpoint_path output/re10k/000c3ab189999a83`

## ✏️ TODO

- [x] create codebase
- [x] add evaluation script
- [x] prepare sample data
- [x] write installation guide
- [ ] add data preprocessing script
- [ ] implement GPU-optimized k-means
- [ ] add Gradio visualization

## 🎓 Citation

```
@inproceedings{cc2025_reggs,
  title     = {{RegGS}: Unposed Sparse Views Gaussian Splatting with {3DGS} Registration},
  author    = {Cheng, Chong and Hu, Yu and Yu, Sicheng and Zhao, Beizhen and Wang, Zijian and Wang, Hao},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
