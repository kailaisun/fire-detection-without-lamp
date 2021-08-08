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
The results of the lamp and fire detection (Rectangular mask) is in the output folder.
```
python demo.py --save-txt --savetxt --saveimg --grabcut --FLremove
```
The results of the lamp and fire detection (Segmentation mask) is in the output folder.

### How to download data：
1.Fire data
Download fire data from https://collections.durham.ac.uk/files/r2d217qp536#.X5F5G2gzZnK

2.Lamp data
download lamp data from openimageV4：
```
git clone https://github.com/EscVM/OIDv4_ToolKit.git 
```
following its instruction,  then use
```
 'python3 main.py downloader --classes Lamp --type_csv train(test,validation)' to get data of lamps. 
```
### How to train yolov3：
put all the pictures and labels in a same folder(data/JPEGImages)
If you want to train Fire-only weights:

```
 'python3 train.py --epochs n(epochs you want) --weights weights/yolov3.pt --cfg cfg/yolov3-fireonly.cfg --data data/fire.data --name data/fire.names --single-cls' 
```
If you want to train Fire-lamp weights:
```
 'python3 train.py --epochs n(epochs you want) --weights weights/yolov3.pt --cfg cfg/yolov3.cfg --data data/fires.data --name data/fires.names' 
 ```

### Reference：
 [Firenet](https://github.com/tobybreckon/fire-detection-cnn)， 
 [YoloV3](https://github.com/ultralytics/yolov3)
