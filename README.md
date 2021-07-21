# TLA - Twitter Linguistic Analysis
## Tool for linguistic analysis of communities 

TLA is built using PyTorch, Transformers and several other State-of-the-Art machine learning
techniques and it aims to expedite and structure the cumbersome process of collecting, labeling, and analyzing data
from Twitter for a corpus of languages while providing detailed labeled datasets
for all the languages. The analysis
provided by TLA will also go a long way in understanding the sentiments of
different linguistic communities and come up with new and innovative solutions
for their problems based on the analysis.
List of languages our library provides support for are  listed as follows:<br>
1>English
<br>
2>Russian
<br>
3>Romainain
<br>
4>Persian
<br>
5>Spanish
<br>
6>Hindi
<br>
7>Chinese
<br>
8>French
<br>
9>Portuguese
<br>
10>Indonesian
<br>
11>Urdu
<br>
12>Turkish
<br>
13>Japanese
<br>
14>Dutch
<br>
15>Thai
<br>
16>Swedish
<br>



## Features

- Provides 16 labeled Datasets for different languages for analysis.
- Implements Bert based architecture to identify languages.
- Provides Functionalities to Extract,process and label tweets from twitter.
- Provides a Random Forest classifier to implement sentiment analysis on any string.

---


### Installation :
```
pip install --upgrade https://github.com/tusharsarkar3/TLA.git
```
---

## <div align="center">Overview </div>

<details>
<summary>Extract data</summary>


```
from TLA.Data.get_data import store_data

store_data('en',False)
```
This will extract and store the unlabeled data in a new directory inside data named 
datasets.
</details>

<details>
<summary>Label data</summary>


```
from TLA.Datasets.get_lang_data import language_data

df = language_data('en')
print(df)
```
This will print the labeled data that we have already collected.
</details>

<details>
<summary>Classify languages</summary>

<details>
<summary>Training </summary>

Training can be done in the following way:

```
from TLA.Lang_Classify.train import train_lang

train_lang(path_to_dataset,epochs)
```
</details>

<details>
<summary>Prediction </summary>

Inference is done in the following way:

```
from TLA.Lang_Classify.predict import predict

model = get_model(path_to_weights)
preds = predict(dataframe_to_be_used,model)
```
</details>


</details>


<details>
<summary>Analyse</summary>

<details>
<summary>Training </summary>

Training can be done in the following way:

```
from TLA.Analyse.train_rf import train_rf

train_rf(path_to_dataset)
```
This will store all the vectorizers and models in a seperate directory named
saved_rf and saved_vec and they are present inside Analysis directory.
Further instructions for training multiple languages is given in the next section which 
shows how to run the commands using CLI

</details>

<details>
<summary>Final Analysis </summary>

Analysis is done in the following way:

```
from TLA.Analysis.analyse import analyse_data 

analyse_data(path_to_weights)
```

This will store the final analysis as .csv inside a new directory named
analysis.

</details>


</details>


## <div align="center">Overview with Git</div>
<details> 
<summary>Installation another method</summary>

```
git clone https://github.com/tusharsarkar3/TLA.git
```
</details>
<details>
<summary>Extract data</summary>

Navigate to the required directory

```
cd Data
```

Run the following command:
```
python get_data.py --lang en --process True
```
Lang flag is used to input the language of the dataset that is required and
process flag shows where pre-processing should be done before returning the data.
Give the following codes in the lang flag wrt the required language:

| Language | Code  | Language | Code |
| ---------------- | ---------------- | ---------------- | ---------------- |
| Iris  | <b>100</b>  | 97.7 |
| Breast Cancer  | <b>96.49</b>  | 96.47 |
| Wine  | <b>97.22</b>  | <b>97.22</b> |
| Diabetes  | <b>78.78</b>  | 77.48 |
| Titanic  | 79.85  | <b>80.5</b> |
| German Credit  | 71.33  | <b>77.66</b> |

</details>

<details>
<summary>Training</summary>
 
Run commands below to reproduce results on [Drone Dataset](https://www.kaggle.com/dasmehdixtr/drone-dataset-uav) dataset..
```bash
$ python train.py --img 640 --batch 16 --epochs 15 --data coco128.yaml --weights yolov5s.pt

```

 Check out <a href="https://github.com/ultralytics/yolov5">YOLOv5</a> for more information.
</details>  

<details open>
<summary>Inference </summary>

```bash
$ python detect.py --weights 'path to the best set of weights' --source 0  # webcam       
                                                                        file.jpg  # image 
                                                                        file.mp4  # video
                                                                        path/  # directory
                                                                        path/*.jpg  # glob
                                                                        'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                                                                        'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
 The results will be stored in a new directory named run which will be on the same level as the root directory.
 
 Check out <a href="https://github.com/ultralytics/yolov5">YOLOv5</a> for more information.
</details>



---
### Results:

![img](exp27/test_batch0_pred.jpg)
![img](exp27/results.png)
![img](exp27/P_curve.png) 
![img](exp27/R_curve.png)
![img](exp27/PR_curve.png)

---

<h3 align="center"><b>Developed with :heart: by <a href="https://github.com/tusharsarkar3">Tushar Sarkar</a>



                                                         
                                          
```
The output will be a  
```
---
### Output images :

![img](screenshots/Results_metrics.png)  
![img](screenshots/results_graph.png)
---

### Reference
If you make use of this software for your work, we would appreciate it if you would cite us:
```
@misc{sarkar2021xbnet,
      title={XBNet : An Extremely Boosted Neural Network}, 
      author={Tushar Sarkar},
      year={2021},
      eprint={2106.05239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
---
 #### Features to be added :
- Metrics for different requirements
- Addition of some other types of layers

---

<h3 align="center"><b>Developed with :heart: by <a href="https://github.com/tusharsarkar3">Tushar Sarkar</a> and <a href="https://github.com/nishant42491">Nishant Rajadhyaksha</a>
