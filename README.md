## Team 5 Final Project


### Running the code it will get the plot for confusion matrix plot and also store the low confidence data into new csv file. 

### Dataset
### train, test: https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=testdata.manual.2009.06.14.csv
### Yelp: https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset
### 10000 Restaurant Reviews: https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews
### IMDB: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

```bash
# Run poetry for environment
poetry shell

# run the main code
python main.py
```

# If you want to combine all data and run once
```bash
python main.py --bigdata
```
A Random Forest model will be trained first and it's model stored in ```models/big_data.pkl```, subsequent runs will skip Random Forest training if a model has been trained. Use the ```--overwrite``` flag to force retraining and starts predicting using the saved model. The model will then goes on to train a distilBERT model using the low confidence data. To skip Random Forest altogether, use the ```--skiprf``` flag.
