# TLA - Twitter Linguistic Analysis
## Tool for linguistic analysis of communities 


[![](https://img.shields.io/badge/Made_with-PyTorch-res?style=for-the-badge&logo=pytorch)](https://pytorch.org/ "PyTorch")


TLA is built using PyTorch, Transformers and several other State-of-the-Art machine learning
techniques and it aims to expedite and structure the cumbersome process of collecting, labeling, and analyzing data
from Twitter for a corpus of languages while providing detailed labeled datasets
for all the languages. The analysis
provided by TLA will also go a long way in understanding the sentiments of
different linguistic communities and come up with new and innovative solutions
for their problems based on the analysis.
List of languages our library provides support for are  listed as follows:<br>

| Language | Code   | Language | Code |
| ----------------  | ---------------- | ---------------- | ---------------- |
| English |   en    | Hindi    |   hi  |
| Swedish |   sv    | Thai     |   th  |
| Dutch   |   nl   | Japanese |   ja  |
 | Turkish  |   tr  | Urdu     |  ur   |
 | Indonesian | id   |Portuguese | pt  |
 | French    | fr   | Chinese |  zn-ch |
 | Spanish  | es    | Persian |   fa   |
 | Romainain | ro  | Russian | ru |



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



 <summary>Loading Dataset</summary>

To load a dataset run the following command in python.
 
```
df= pd.read_csv("TLA/TLA/Datasets/get_data_en.csv")
 
```
The command will return a dataframe consisting of the data for the specific language requested.
 
In the phrase get_data_en, en can be sunstituted by the desired language code to load the dataframe for the specific language.
 
  <summary>Pre-Processing</summary>
 
 To preprocess a given string run the following command.
 
 In your terminal use code 
 
 ```
 cd Data
 ```
 
 then run the command in python
 
 ```
 from TLA.Data import Pre_Process_Tweets
 
 df=Pre_Process_Tweets.pre_process_tweet(df)
 ```
 
 Here the function pre_process_tweet takes an input as a dataframe of tweets and returns an output of a dataframe with the list of preprocessed words
 for a particular tweet next to the tweet in the dataframe.
 
 
 
 
</details>




<details>
<summary>Analysis</summary>
 
 <summary> Training </summary>
 To train a random forest classifier for the purpose of sentiment analysis run the following command in your terminal.
 
 ```  
 cd Analysis
 ```
 then 
 
 ```
 python train.rf --path "path to your datafile" --train_all_datasets False
 ```
 
 here the --path flag represents the path to the required dataset you want to train the Random Forest Classifier on
 the --train_all_datasets flag is a boolean which can be used to train the model on multiple datasets at once.
 
 The output is a file with the a .pkl file extention saved in the folder at location "TLA\Analysis\saved_rf\{}.pkl"
 The output for vectorization of is stored in a .pkl file in the directory  "TLA\Analysis\saved_vec\{}.pkl"
 
 <summary> Get Sentiment </summary>
 
 To get the sentiment of any string use the following code.
 
 In your terminal type
 
 ```
 cd Analysis
 ```
 then in your terminal type
 
 ```
 python get_sentiment.py --prediction "Your string for prediction to be made upon" --lang "en"
 ```
 
 here the --prediction flag collects the string for which you want to get the sentiment for.
 the --lang represents the language code representing the language you typed your string in.
 
 The output is a sentiment which is either positive or negative depending on your string.
 
 
 <summary>Statistics</summary>
 
 To get a comprehensive statistic on sentiment of datasets run the following command.
 
 In your terminal type
 
 ```
 cd Analysis
 ```
 
 then
 
 ```
 python analyse.py 
 ```
 
 This will give you an output of a table1.csv file at the location 'TLA\Analysis\analysis\table1.csv' comprising of statistics relating to the
 percentage of positive or negative tweets for a given language dataset.
 
 It will also give a table2.csv file at 'TLA\Analysis\analysis\table2.csv' comprising of statistics for all languages combined.
 
 
 </details>  






<details>
<summary>Language Classification </summary>
 <summary>Training</summary>
 To train a model for language classfication on a given dataset run the following commands.
 
 In your terminal run
 
 ```
cd Lang_Classify
 ```
 then run
 
 ```
 python train.py --data "path for your dataset" --model "path to weights if pretrained" --epochs 4
 ```
 
The --data flag requires the path to your training dataset.
 
 The --model flag requires the path to the model you want to implement
 
 The --epoch flag represents the epochs you want to train your model for.
 
 The output is a file with a .pt extention named saved_wieghts_full.pt where your trained wieghst are stored.
 
 
 <summary>Prediction</summary>
 To make prediction on any given string Us ethe following code.
 
 In your terminal type
 
 ```
 cd Lang_Classify
 ```
 then run the code
 
 ```
 python predict.py --predict "Text/DataFrame for language to predicted" --weights " Path for the stored weights of your model " 
 ```
 
 The --predict flag requires the string you want to get the language for.
 
 The --wieghts flag is the path for the stored wieghts you want to run your model on to make predictions.
 
 
The outputs is the language your string was typed in.



</details>
 



---
### Results:

![img](ss/performance.png)

Performance of TLA ( Loss vs epochs)



---
 #### Features to be added :
- Access to more languag
- Addition of some other types of layers

---

<h3 align="center"><b>Developed by <a href="https://github.com/tusharsarkar3">Tushar Sarkar</a> and <a href="https://github.com/nishant42491">Nishant Rajadhyaksha</a>
