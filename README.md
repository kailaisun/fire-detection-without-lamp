# fire-detection-without-lamp
 

<!-- ### Problems：
![image](https://github.com/kailaisun/fire-detection-without-lamp/blob/master/data/1.png)

### Methods：
![image](https://github.com/kailaisun/fire-detection-without-lamp/blob/master/data/2.png)

### Inceptionv4：
![image](https://github.com/kailaisun/fire-detection-without-lamp/blob/master/data/4.bmp) -->


## Installation：

### Packages：
```
pip install -r requirements.txt
```

### Run：
```
python demo.py --save-txt --savetxt --FLremove
```
The results of the lamp and fire detection (suppression is not showed) is in the output folder. The file result.txt shows the fire probability.


```
python demo.py --save-txt --savetxt --saveimg --FLremove
```
The results of the lamp and fire detection (suppression is showed) is in the output folder.

### Reference：
 [Firenet](https://github.com/tobybreckon/fire-detection-cnn)， 
 [YoloV3](https://github.com/ultralytics/yolov3)
