# DataMiningProject

# Visualization
- Heatmap based on lexica like this one https://docs.google.com/presentation/d/1C7U5vLUF38gbC5fHnAnEWsLgHLS9iEzR38S1pAKKhCs/edit#slide=id.g5070c672b6_0_11
- Maybe bubbles https://www.google.com/search?q=data+visualization+bubbles&oq=visualization+bubbles+&aqs=chrome.1.69i57j0l2.6848j1j7&sourceid=chrome&ie=UTF-8 and https://docs.google.com/presentation/d/1C7U5vLUF38gbC5fHnAnEWsLgHLS9iEzR38S1pAKKhCs/edit#slide=id.g5070c672b6_0_5
- Maybe some plots like tsne_lot like this one https://cdn-images-1.medium.com/max/800/1*9e_PVh0wGy99EwVPZtyDAg.png and url https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
- Wordclouds for sure for Positives, Negatives, Neutrals, maybe for all together using the 
code below:
    ```
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from wordcloud import WordCloud
    
    test_string = 'ok ok ok aha wow ok ok ok ok ok ok LOL lel ok ok'

    #make wordcloud object
    wc = WordCloud(background_color = 'white', stopwords = ENGLISH_STOP_WORDS)

    #generate a word cloud
    wc.generate(test_string)

    #store to file
    wc.to_file('test_wordCloud4.png')
    ```
- All visualizations like wordclouds, heatmaps etc should be generated one time as they need some time to be generated and present them at the notebook
- 

# Preprocessing

  - Adding extra stop words like 
    ```
    #make stop words
    my_additional_stop_words=['said','say','just','it','says','It']
    stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    ```
  - General tests if we need more features at preprocessing
  - Maybe we should not do stemming as we will use lexicas

# Vectorization / Classification

  - We need to implement word embeddings vectorization
  - We need to use the correct train and test set 
  - BOW and tfIdf run them one time and then save the results at notebook as they will be the same all the time
  - We convert labels (positive,negative,neutral) into numbers but this may not be necessary
    
# General

  - Organize step-by-step with comments, images etc at Jupyter Notebook
  - When we don't need something anymore we "free" it from the memory as it may be big enough
  - Do some general tests (με το μάτι) so to check if all is OK. 