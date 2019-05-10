# DataMiningProject

# Visualization
- Heatmap based on lexica like this one https://docs.google.com/presentation/d/1C7U5vLUF38gbC5fHnAnEWsLgHLS9iEzR38S1pAKKhCs/edit#slide=id.g5070c672b6_0_11
- Maybe bubbles https://www.google.com/search?q=data+visualization+bubbles&oq=visualization+bubbles+&aqs=chrome.1.69i57j0l2.6848j1j7&sourceid=chrome&ie=UTF-8 and https://docs.google.com/presentation/d/1C7U5vLUF38gbC5fHnAnEWsLgHLS9iEzR38S1pAKKhCs/edit#slide=id.g5070c672b6_0_5
- Maybe some plots like tsne_lot like this one https://cdn-images-1.medium.com/max/800/1*9e_PVh0wGy99EwVPZtyDAg.png and url https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
- Maybe tsne

# Preprocessing

  - Adding extra stop words like + from web
    ```
    #make stop words
    my_additional_stop_words=['said','say','just','it','says','It']
    stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    ```
  - General tests if we need more features at preprocessing
  - Stemming and not stemming

# Vectorization / Classification

  - Maybe use  KeyedVectors for the word embeddings model
  - Kolovou said max features should be between 1000 to 3000
  - Check about max_features at vectorizers. Useful links: 
    https://stackoverflow.com/questions/40731271/test-and-train-dataset-has-different-number-of-features?fbclid=IwAR2NKF1aCx4BRqZb4Hd04TrV8JZxDv3vsprlqwHpybOd13nXWzvL8XfZH88 
    and https://stackoverflow.com/questions/46118910/scikit-learn-vectorizer-max-features?fbclid=IwAR2NKF1aCx4BRqZb4Hd04TrV8JZxDv3vsprlqwHpybOd13nXWzvL8XfZH88
  - Maybe to make a plot to see what happens if we use numberOne < max_features < numberTwo  at vectorizers
  - (Maybe not) To make a pipeline like this one : https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
  - Maybe word embeddings from web
  - Study each of BOW, tfidf, word embeddings and see what they expect for input
  - Use of lexica
  
# General

  - (???) When we don't need something anymore we "free" it from the memory as it may be big enough (Python most likely does it on its own)
  - Do some general tests (με το μάτι) so to check if all is OK. 
  - Before final version remove all #printToBeRemoved
  - Note that neutral tweets are difficult to be discovered
  
# Extra (maybe impossible to be implemented)

  - Bonus classifier
  - Correct dictation of tweets
