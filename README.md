## Air pollution prediction based on deep learning models: a case study applied to the Yan'an city of China

### Notes and Precautions:
*This repository provides experimental data and relevant model code.
It should be noted that different models do not have universal training code, so if you want to run the program, you need to make corresponding code adjustments based on your actual situation (assuming you have basic knowledge of deep learning and the PyTorch framework).
If you want to train with your own experimental data, you need to write the corresponding dataloader.
More details depend on your actual running environment.*

### Model architecture
![modelframework.png](img%2Fmodelframework.png)
### Data:AQI time series of Yan 'an from 2014 to 2019
The air pollution data set of Yan 'an city required by the experiment is placed under the data folder

### Models folder:
- Informer folder
- RNN
- LSTM
- TCN
- MLP
- former
- other modules

### Utils folder
In this folder are stored time coding function, evaluation function, drawing function and other related tools