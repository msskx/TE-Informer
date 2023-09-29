## Air pollution prediction based on deep learning models: a case study applied to the Yan'an city of China

### Notes and Precautions:
*This repository provides experimental data and relevant model code.
It should be noted that different models do not have universal training code, so if you want to run the program, you need to make corresponding code adjustments based on your actual situation (assuming you have basic knowledge of deep learning and the PyTorch framework).
If you want to train with your own experimental data, you need to write the corresponding dataloader.
More details depend on your actual running environment.*
### Data:AQI time series of Yan 'an from 2014 to 2019
在data文件夹下放置了实验需要的延安市空气污染数据集

### Models folder:
- RNN
- LSTM
- TCN
- Transformer Encoder
- TE-Informer

![modelframework.png](img%2Fmodelframework.png)


### Utils folder
在该文件夹下存放了评估函数，绘图函数等相关工具