# fire-detection-without-lamp
 

### 问题由来及推理：
![image](https://github.com/kailaisun/fire-detection-without-lamp/blob/master/data/1.png)

### 方法：
![image](https://github.com/kailaisun/fire-detection-without-lamp/blob/master/data/2.png)

### Inceptionv4网络结构：
![image](https://github.com/kailaisun/fire-detection-without-lamp/blob/master/data/4.bmp)


## 安装：

### 环境依赖：
```
pip install -r requirements.txt
```

### 运行：
```
python demo.py --save-txt --savetxt --FLremove
```
在output文件夹里，查看灯光和火灾的检测结果（不显示黑色抑制）。根目录result.txt显示火灾判定概率。


```
python demo.py --save-txt --savetxt --saveimg --FLremove
```
在output文件夹里，查看灯光和火灾的检测结果（显示黑色抑制）。

### 参考：
 [Firenet](https://github.com/tobybreckon/fire-detection-cnn)， 
 [YoloV3](https://github.com/ultralytics/yolov3)
