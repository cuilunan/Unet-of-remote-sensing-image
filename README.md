# Unet-of-remote-sensing-image

  针对高分辨率遥感卫星进行地物识别，主要有15类的地物类型，包括各种农作物，工业用地，河流，水源，建筑物等。利用Unet结构进行语义分割，得到各个地物类型的场景分割图像，Unet结构和官方论文不太一样，自己根据理解进行了一些微调，改变了输出通道的数量，和上采样层后通道数量，每个巻积层后面加了batchNromalize层，正确率有一定的提高，最后finetune的15类分割准确率达到82%。
  
  数据集：主要采用的landsat多通道图像，根据美国官方网站提供的地物标签制作卫星图像的groundTruth,得到23000多张训练图像，每张224×224
美国卫星数据官网：https://nassgeodata.gmu.edu/CropScape/

  代码：基于tensorflow的网络结构，其中process.py是制作训练数据用的，将一张7000×8000的大卫星图片根据经纬度分割成2万多张224×224的小卫星图和相应的groundTruth.

  数据量过大，如果有需要数据的朋友，可以联系我，qq:153323967

  groundTruth:
  ![error](https://github.com/cuilunan/Unet-of-remote-sensing-image/raw/master/result/groundTruth.png)
  
  input_image:
  ![error](https://github.com/cuilunan/Unet-of-remote-sensing-image/raw/master/result/input_image.png)
  prediction:
  ![error](https://github.com/cuilunan/Unet-of-remote-sensing-image/raw/master/result/prediction.png)
  train loss:
  ![error](https://github.com/cuilunan/Unet-of-remote-sensing-image/raw/master/result/loss_train.png)
  validation loss:
  ![error](https://github.com/cuilunan/Unet-of-remote-sensing-image/raw/master/result/loss_val.png)
  train accuracy:
  ![error](https://github.com/cuilunan/Unet-of-remote-sensing-image/raw/master/result/train_acc.png)
  validation accuracy:
  ![error](https://github.com/cuilunan/Unet-of-remote-sensing-image/raw/master/result/val_acc.png)



