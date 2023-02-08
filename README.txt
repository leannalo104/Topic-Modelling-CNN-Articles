# Unsupervised Learning of News Articles Using Topic Modelling 

This project explores the creation of a custom topic modelling method to label a database of CNN news articles using word vectorization and clustering, and evaluate it against the existing classification 

There are 2 python notebooks for my project, which should be viewed in the following order:
1. Capstone_CNN_EDA+Pre-processing.ipynb
- Contains EDA, pre-processing, preliminary word2vec model

2. Capstone_CNN_Modelling+Evaluation.ipynb
- Contains word2vec model, HDBSCAN model, cluster identification and labelling, and final assessment using cosine similarity score 

A pdf file containing slides summarizing the project methodology and findings is also attached. 

The initial dataset can be downloaded here: https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning

## Topic Modelling

Topic modelling is commonly used unsupervised ML technique in the industry to process, organize, and categorize vast quantities of textual information, with common methods being LDA (Latent Dirichlet Allocation (LDA) and Latent Semantic Analysis (LSA). My project will not make use of these popular techniques, but instead will be carrying out a custom method where I will oversee all the individual steps (vectorization, clustering, identification/labelling) in the process. 

In this age where many read the news online, knowing the main topics of a large amount of text can help a business:
- Determine the most relevant tags to use, thus helping improve a website’s SEO (search engine optimization)
- Inform website hierarchy, increasing user experience for site visitors

## Dataset 

The dataset used was obtained from Kaggle and covers articles from from August 2011 to April 2022. It is structured, containing 37,949 rows (representing each article) and 11 columns.
The main columns used in my analysis were 1) Article text, which was the basis of my unsupervised learning method, which I then compared to the 2) Section column. 

Data Preprocessing + Cleaning
Key steps included removal of rows containing null values, conversion of column datatype (date time format for dates, string format for text) and dropping any redundant/irrelevant columns (i.e. headline, description, keywords, etc.). 
Additional cleaning was required for the ‘Article text’ column, where special characters/punctuation, stop words, and common non-specific words appearing in the articles (i.e. CNN, photo, caption, etc.) were removed. The text was then lemmatized and tokenized for processing in my word model. 

## EDA 

I explored the breakdown of articles based on the following columns:
- Category: 9 total categories. The most common are News and Sports.  
- Section: 55 unique sections. The most popular Section is Europe, followed by Sport, and then Football. 
- Date: Articles generally show an increase with year. Broken down by month, we see an increase in articles in early 2022, with most from March 2022. 

## Model Summary
The following models were used:
1. Word2Vec - The vocabulary was built using the tokenized article text column as well as bigram search using gensim’s Phrases model. The average vector for each document was then calculated by taking the average of the individual token vectors. 
2. HDBSCAN (Clustering) - A pipeline was run using the validity score to determine the best hyperparameters for the model. The embedded word documents and associated clusters were all visualized using UMAP. 
3. Cluster Identification/Labelling - A Bag of Words was run on each cluster to determine the top 30 words. Using each word bank as a guide, each cluster was labelled with a topic based on the prominent themes/topics dictated by the words. The noise cluster was simply classified as ‘news’. A dictionary was created with the cluster ID key and unsupervised topic value, and mapped back to each article in the dataset.
4. Cosine Similarity Score - The cosine similarity score was calculated between the unsupervised label determined by my model and the ‘Section’ column. For cluster labels containing more than 1 word, the average vector of all the words was used in the calculation. If a label was not in my Word2Vec dictionary, this score could not be determined and was calculated as Nan. 

## Results
- HDBSCAN:
The validity score for my clustering model was 37.8%, not extremely high but the best possible score determined by my pipeline. This being said, the HDBSCAN documentation FAQ indicates that performance is best on 50-100 dimensional data and significantly decreases beyond that. My data had 300 dimensions.
In total, my model created 61 clusters, including the noise cluster. The size of the noise cluster in my model was the largest one, comprising 28.2% of the articles. This was just slightly higher than my second biggest cluster at 27%. 
- Cluster Identification and Labelling: Overall this process was quite smooth. I found that most of the clusters had clearly defined topics (i.e. tennis, royal family, religion), but some did have overlapping themes- perhaps suggesting the HDBSCAN hyperparameters were too strict- while a small number were ambiguous in theme based on the top 30 words.
- Cosine Similarity Score between Model Label vs Article label: Just under 3% of the documents did not have a calculated cosine similarity score. Excluding articles in the noise cluster, my model classified 26886 rows, or 70% of the initial dataset. As seen in Figure 6, There was a 100% similarity score for 30.9% of the documents. Over ⅔ of the articles had a score higher than 50%. The average score was 62.7%. The 50th percentile had a score of 64%, with the 25th percentile at a score of 40% or below. 
- Qualitative Assessment of Cluster vs Section label: In my final evaluation step, I did a visual scan of the dataframe to compare the 2 label columns with the original article to assess the ‘fit’ of the label. I did find many instances where my model label (‘cluster category’ column) was more granular than the original label (‘Section’ column), which appeared to be more geographical than topic-oriented. 
I also noticed some instances where the vectors in my word model, which were used to calculate cosine similarity score, were questionable. For example, the cosine similarity score between the words ‘tennis’ and ‘sport’ revealed a surprisingly low score of 0.5.
That being said, I did find some articles (although far fewer) that were not properly clustered in my model. The last article mentioning the New Orleans Saints was likely clustered with religion-related articles due to the word ‘saint’ in the text, as word embeddings in Word2vec are context-independent.

## Final comments
Overall, I do think there is some success in this method and would suggest some tweaks for project improvement:
- Word model: There are advantages to exploring another word model, such as BERT, which takes context into account and can perhaps yield more accurate vectorization, clustering, and cosine similarity score.
- HDBSCAN Clustering model: Reduce to 150 dimensions, Optimize hyperparameters, finding ideal balance between a high validity score, less dense noise cluster (resulting in less data ‘lost’), and cluster number




