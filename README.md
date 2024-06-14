<a href="https://arxiv.org/abs/2105.05874" alt="Citation"><img src="https://img.shields.io/badge/cite-citation-blue" /></a>
<a href="https://twitter.com/FeTS_Challenge" alt="Citation"><img src="https://img.shields.io/twitter/follow/fets_challenge?style=social" /></a>

```
SimpleITK==2.2.1
Pillow==8.4.0

git clone https://github.com/intel/openfl.git && cd openfl && git checkout f4b28d710e2be31cdfa7487fdb4e8cb3a1387a5f
setup.py 에서 open utf-8 로 수정
pip install .
cd ..


git clone https://github.com/CBICA/GaNDLF.git && cd GaNDLF && git checkout e4d0d4bfdf4076130817001a98dfb90189956278
git submodule update --init --recursive
pip install .
cd ..


git clone https://github.com/FETS-AI/Algorithms.git fets && cd fets && git checkout fets_challenge
.gitmodules 에서 GANDLF 로 폴더명 변경 (url 의 대소문자는 구분하지 않아서 문제없음)
git submodule update --init --recursive
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install .
cd ..

setup.py 에 있는 설치패키지 모두 주석처리 후

    install_requires=[
        # 'openfl @ git+https://github.com/intel/openfl.git@f4b28d710e2be31cdfa7487fdb4e8cb3a1387a5f',
        # 'GANDLF @ git+https://github.com/CBICA/GaNDLF.git@e4d0d4bfdf4076130817001a98dfb90189956278',
        # 'fets @ git+https://github.com/FETS-AI/Algorithms.git@fets_challenge',
    ],

pip install .
export CUDA_VISIBLE_DEVICES=0
```



# Federated Tumor Segmentation Challenge

The repo for the FeTS Challenge: The 1st Computational Competition on Federated Learning.

## Website

https://www.synapse.org/#!Synapse:syn28546456

## Challenge Tasks

### Task 1

The first task of the challenge involves customizing core functions of a baseline federated learning system implementation. The goal is to improve over the baseline consensus models in terms of robustness in final model scores to data heterogeneity across the simulated collaborators of the federation. For more details, please see [Task_1](./Task_1).

### Task 2

This task utilizes federated testing across various sites of the FeTS initiative in order to evaluate model submissions across data from different medical institutions, MRI scanners, image acquisition parameters and populations. The goal of this task is to find algorithms (by whatever training technique you wish to apply) that score well across these data. For more details, please see [Task_2](./Task_2).

## Documentation and Q&A

Please visit the [challenge website](https://synapse.org/fets) and [forum](https://www.synapse.org/#!Synapse:syn28546456/discussion/default).

<!-- ## Frequently asked questions

Please see [FAQ](https://fets-ai.github.io/Challenge/faq/). -->

## Citation

Please cite [this paper](https://arxiv.org/abs/2105.05874) when using the data:

```latex
@misc{pati2021federated,
      title={The Federated Tumor Segmentation (FeTS) Challenge}, 
      author={Sarthak Pati and Ujjwal Baid and Maximilian Zenk and Brandon Edwards and Micah Sheller and G. Anthony Reina and Patrick Foley and Alexey Gruzdev and Jason Martin and Shadi Albarqouni and Yong Chen and Russell Taki Shinohara and Annika Reinke and David Zimmerer and John B. Freymann and Justin S. Kirby and Christos Davatzikos and Rivka R. Colen and Aikaterini Kotrotsou and Daniel Marcus and Mikhail Milchenko and Arash Nazer and Hassan Fathallah-Shaykh and Roland Wiest and Andras Jakab and Marc-Andre Weber and Abhishek Mahajan and Lena Maier-Hein and Jens Kleesiek and Bjoern Menze and Klaus Maier-Hein and Spyridon Bakas},
      year={2021},
      eprint={2105.05874},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
