# BRIDGE

This repository will contain the implementation of **BRIDGE** (Bayesian Reconstruction and Inference of Data-driven Generative Environments), a GPU-accelerated, differentiable framework written in [`jax`](https://docs.jax.dev/en/latest/index.html) for field-level inference in large-scale structure cosmology.

The BRIDGE code leverages efficient structure formation techniques (ALPT; [Kitaura & Heß 2013](https://academic.oup.com/mnrasl/article/435/1/L78/1097681)) and incorporates the HICOBIAN bias model ([Coloma-Nadal et al. 2024](https://iopscience.iop.org/article/10.1088/1475-7516/2024/07/083); [Sinigaglia et al. 2024](https://www.aanda.org/articles/aa/abs/2024/02/aa46931-23/aa46931-23.html)) for a physically motivated nonlocal bias based on a differentiable cosmic-web classification, all within a fully Bayesian framework

For more details, refer to our paper:
[**Rosselló, P. et al. (2025)**](https://arxiv.org/abs/2506.03969).

The code will be made publicly available soon.


<small>This work has been supported by the Spanish Ministry of Science and Innovation under the Big Data of the Cosmic-Web project (PID2020-120612GB-I00 / AEI / 10.13039/501100011033, PI: Francisco-Shu Kitaura).</small>