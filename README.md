# InvarNet
This is the demo implementation of our paper: [Towards Invariant Time Series Forecasting in Smart Cities](https://dl.acm.org/doi/abs/10.1145/3589335.3651897).


## Introduction
In the transformative landscape of smart cities, the integration of the cutting-edge web technologies into time series forecasting presents a pivotal opportunity to enhance urban planning, sustainability, and economic growth. The advancement of deep neural networks has significantly improved forecasting performance. However, a notable challenge lies in the ability of these models to generalize well to out-of-distribution (OOD) time series data. The inherent spatial heterogeneity and domain shifts across urban environments create hurdles that prevent models from adapting and performing effectively in new urban environments. To tackle this problem, we propose a solution to derive invariant representations for more robust predictions under different urban environments instead of relying on spurious correlation across urban environments for better generalizability. Through extensive experiments on both synthetic and real-world data, we demonstrate that our proposed method outperforms traditional time series forecasting models when tackling domain shifts in changing urban environments. The effectiveness and robustness of our method can be extended to diverse fields including climate modeling, urban planning, and smart city resource management.


## Citation
If you find it useful, please cite our paper. Thank you!
```
@inproceedings{zhang2024towards,
  title={Towards invariant time series forecasting in smart cities},
  author={Zhang, Ziyi and Ren, Shaogang and Qian, Xiaoning and Duffield, Nick},
  booktitle={Companion Proceedings of the ACM on Web Conference 2024},
  pages={1344--1350},
  year={2024}
}
```
