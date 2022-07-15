# 基于GauGAN的风景图片生成器
![主要结果](https://s3.bmp.ovh/imgs/2022/04/19/440f015864695c92.png)

## 简介
本项目包含了第二届计图挑战赛中Benchmark小组对赛道一风景生成比赛的代码实现。

本项目的特点是：基于GauGAN的方法，添加了VAN的结构与UNet的想法在jittor框架下给出代码实现，并运用在清华大学计算机系图形学实验室从Flickr官网收集的1万两千张高清（宽1024、高768）的风景图片及其语义图上进行训练和测试。评测时各项分数为：

| Mask Accuracy | 美学评分   | FID     | 综合分数  |
| ------------- | ------ | ------- | ----- |
| 0.8479        | 5.0436 | 43.8223 | 0.452 |

## 配置环境 

本项目可在 3090 上运行，训练时间约为 6 天。

#### 运行环境
- ubuntu 20.04 LTS
- python >= 3.8
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

## 训练&测试
可以运行`python train.py --input_path train数据路径`执行训练

可以运行`python test.py --input_path train数据路径 --output_path ./results`执行测试

## 致谢
此项目GauGAN部分基于论文 *[Semantic Image Synthesis with Spatially-Adaptive Normalization](http://arxiv.org/pdf/1903.07291)* 实现，VAN部分基于论文 *[Visual Attention Network](https://arxiv.org/abs/2202.09741)* 实现。UNet部分代码以及代码主框架参考了提供的[baseline](https://github.com/Jittor/JGAN/tree/master/competition#%E8%B5%9B%E9%A2%98%E4%B8%80%E9%A3%8E%E6%99%AF%E5%9B%BE%E7%89%87%E7%94%9F%E6%88%90%E8%B5%9B%E9%A2%98)。此外，也感谢清华大学2022年春季计算机图形学教学团队提供的帮助与在算力上的支持。