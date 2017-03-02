#PYTORCH implementation of *CNNs for Sentence Classification*
##Data Preprocessing
Data preprocessing is borrowed from https://github.com/yoonkim/CNN_sentence.

##Requirements
1. pytorch
2. climate (for logging)

## Running the models

Example script
```
sh sh_py_cnn_txt_sentence.sh
```

Details of the architecture as well as the parameter settings are borrowed from https://github.com/yoonkim/CNN_sentence.

###Example output:
Using sh sh_py_cnn_txt_sentence.sh

*i is the index for 10-fold cross-validation.*
### Training
```
I 2017-03-02 12:19:33 __main__:275 i = 3, epoch = 9, mini-b = 0/136, loss = 0.3454
I 2017-03-02 12:20:48 __main__:275 i = 3, epoch = 9, mini-b = 10/136, loss = 0.1774
I 2017-03-02 12:22:03 __main__:275 i = 3, epoch = 9, mini-b = 20/136, loss = 0.2944
I 2017-03-02 12:23:17 __main__:275 i = 3, epoch = 9, mini-b = 30/136, loss = 0.2454
I 2017-03-02 12:24:32 __main__:275 i = 3, epoch = 9, mini-b = 40/136, loss = 0.4117
I 2017-03-02 12:25:47 __main__:275 i = 3, epoch = 9, mini-b = 50/136, loss = 0.3266
I 2017-03-02 12:27:02 __main__:275 i = 3, epoch = 9, mini-b = 60/136, loss = 0.4027
I 2017-03-02 12:28:17 __main__:275 i = 3, epoch = 9, mini-b = 70/136, loss = 0.2635
I 2017-03-02 12:29:32 __main__:275 i = 3, epoch = 9, mini-b = 80/136, loss = 0.3072
I 2017-03-02 12:30:47 __main__:275 i = 3, epoch = 9, mini-b = 90/136, loss = 0.3126
I 2017-03-02 12:32:02 __main__:275 i = 3, epoch = 9, mini-b = 100/136, loss = 0.2749
I 2017-03-02 12:33:17 __main__:275 i = 3, epoch = 9, mini-b = 110/136, loss = 0.2964
I 2017-03-02 12:34:32 __main__:275 i = 3, epoch = 9, mini-b = 120/136, loss = 0.2080
I 2017-03-02 12:35:46 __main__:275 i = 3, epoch = 9, mini-b = 130/136, loss = 0.2463
```
###Validation set
```
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.400, accuracy = 81.250
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.351, accuracy = 81.250
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.335, accuracy = 87.500
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.511, accuracy = 78.125
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.347, accuracy = 85.938
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.468, accuracy = 78.125
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.559, accuracy = 71.875
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.418, accuracy = 84.375
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.586, accuracy = 70.312
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.378, accuracy = 82.812
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.304, accuracy = 87.500
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.403, accuracy = 87.500
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.464, accuracy = 79.688
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.515, accuracy = 75.000
I 2017-03-02 12:36:24 __main__:296 i = 3, loss = 0.559, accuracy = 73.438
```
###Testing
```
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.299, accuracy = 90.625
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.463, accuracy = 81.250
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.378, accuracy = 79.688
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.433, accuracy = 82.812
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.317, accuracy = 85.938
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.424, accuracy = 79.688
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.455, accuracy = 73.438
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.380, accuracy = 87.500
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.387, accuracy = 79.688
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.332, accuracy = 85.938
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.413, accuracy = 79.688
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.441, accuracy = 82.812
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.440, accuracy = 78.125
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.483, accuracy = 76.562
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.468, accuracy = 81.250
I 2017-03-02 12:36:24 __main__:316 testing, i = 3, loss = 0.581, accuracy = 73.438
```
Average accuracy is 81.1525 for the above example output (i=3).
