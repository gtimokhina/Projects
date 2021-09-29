# Recommendation Systems using MMoE


NOTE: ADD c++ libraries (check for same in Docker Container first)  
NOTE: For all image references and Code references check bottom of the section  

## 1. Purpose:
  
The primary purpose of this project is to create a recommendation engine using cosine similarity for generating candidate list and MMoE (Multi gate Mixture of Experts) for generating ranking. This is followed up with a web application which can use the recommendation engine model to recommend YouTube videos for users.


![Application Overview. Consumes User queries and transforms the input using BERT Sends input to Recommendation model to predict two categories of user behaviors, i.e., engagement and satisfaction](https://github.com/GPSV-Project/Deep_Learning_Recommendation_System/blob/master/9-%20Images%20and%20Figures/Recommendation%20System-Deployment%20Architecture.png)

## 2. Objectives:
  
Following are the three objectives targeted to achieve from this project:
    *	Create a web application for recommending Videos
    *	Colab to train the MMoE model to give scores for ranking
  
![Single Task recommendation Engine](https://github.com/GPSV-Project/Deep_Learning_Recommendation_System/blob/master/9-%20Images%20and%20Figures/Single%20Task%20Recommendation%20Engine.jpg)

## 3. PROBLEM STATEMENT:
  
Designing and developing a real-world large-scale video recommendation system is full of challenges, including:
      
    *	There are often different and sometimes conflicting objectives which we want to optimize for. For example, we may want to recommend videos that users rate highly and share with their friends, in addition to watching.
      
    *	There is often implicit bias in the system. For example, a user might have clicked and watched a video simply because it was being ranked high, not because it was the one that the user liked the most. Therefore, models trained using data generated from the current system will be biased, causing a feedback loop effect. 
  
![Comparison of (a) Shared-Bottom model. (b) One-gate MoE model. (c) Multi-gate MoE model](https://github.com/GPSV-Project/Deep_Learning_Recommendation_System/blob/master/9-%20Images%20and%20Figures/Comparision%20of%20Shared%20Bottom%20Model%20One%20Gate%20model%20Multi%20Gate%20model.png)

## 4. Dataset:
  
The dataset that has been taken is from Kaggle and can be accessed from this [link](https://www.kaggle.com/datasnaek/youtube-new). The dataset is called trending YouTube video statistics. This dataset was collected using the YouTube API.
  
This dataset includes several months (and counting) of data on daily trending YouTube videos. Data is included for the US, GB, DE, CA, RU, MX, KR, JP, IN and FR regions (USA, Great Britain, Germany, Canada, Russia, Mexico, South Korea, Japan, India, and France, respectively), with up to 200 listed trending videos per day. Each region’s data is in a separate file. Data includes the video title, channel title, publish time, tags, views, likes, and dislikes, description, and comment count
  
## 5. Data Preprocessing:
  
The data also includes a category_id field, which varies between regions. To retrieve the categories for a specific video, find it in the associated JSON. One such file is included for each of the five regions in the dataset.
  
The dataset that we are using for the project is for only united states region. We are focusing on following features:
    *	video titles
    *	channel title 
    *	publish time
    *	tags 
    *	views
    *	likes 
    *	dislikes
    *	description 
    *	comment count

## 6. Data Enrichment:
Apart from this we are also generating some user specific data which can help us with simulation. We were unable to find any user specific data since all user details, user id, user clicks, spend duration etc. is confidential information and internal to google and hence, not publicly available.
  
Following are the features that we are generating to simulate user experience:
    *	user click
    *	user rating
    *	time spend
    *	position
    *	position bias
    *	device information
    *	video embedding
    *	user embedding

The video embedding and user embedding is generated using BERT tokenizer and BERT model, whereas other features are generated using random choice and random integers

## 6. Model Architecture:
  
![Model Architecture](https://github.com/GPSV-Project/Deep_Learning_Recommendation_System/blob/master/9-%20Images%20and%20Figures/Model%20Architecture.png)
In our model we are using the dataset corpus that we have created from previous stage. The dataset is provided to our model for training. As in case of any recommendation engine we have two components:
    *	Candidate List Generation
    *	Generate Ranking


### 6.1. Candidate List Generation:
Our video recommendation system uses cosine similarity for multiple candidate generation, each of which captures one aspect of similarity between query from user and candidate video. In many large-scale systems and organizations this algorithm can be replaced with any other state of the art technique used for candidate generation. The model we selected is content based model (rather than collaborative or hybrid based) and hence dependent on features like user clicks and user profile which we have generated in data preprocessing for ease of simulation

![Candidate Generation](https://github.com/GPSV-Project/Deep_Learning_Recommendation_System/blob/master/9-%20Images%20and%20Figures/Recommendation%20System-Candidate%20List.png)


#### 6.1.1.	Alternative Approaches
Candidate Generation as stated earlier can be created using any techniques or even combination of techniques. For example, one algorithm generates candidates by matching topics of query video. Another algorithm retrieves candidate videos based on how often the video has been watched together with the query video. We can also construct a sequence model similar for generating personalized candidate given user history. We can also use techniques to generate context-aware high recall relevant candidates.

#### 6.1.2.	Current Approach
We chose Cosine similarity for candidate generation simply for the sake of simplicity as the project was specific to academic environment and was not part of any state-of-the-art production model. Also, time to run and execute the model to generate candidate is very quick which is helpful in web application used for demo. At the end, all candidates are pooled into a set and subsequently scored by the ranking system.


### 6.2. Ranking Generation:
Since the primary purpose of the project is to create a recommendation engine for a web application for sorting videos using MMoE as a ranking algorithm, hence, we are implementing MMoE (Multi gate Mixture of Experts) as the ranking algorithm. MMoE address and eliminate the challenges of scalability and implicit bias as we will see below

![Multi gate mixture of experts Model architecture. (a) Shallow Tower: It is used to handle Implicit biad. (b) Mixture of experts managed by a gating using two objective functions (Engagement and Satisfaction)](https://github.com/GPSV-Project/Deep_Learning_Recommendation_System/blob/master/9-%20Images%20and%20Figures/6.2.%20Ranking%20Generation.png)

#### 6.2.1.	MMoE Overview
MMoE structure is a combination of Multi-Layer Perceptrons followed by ReLU activations. There are experts in the MMoE layer which each of them is to learn a different feature of the input data.
  
The output of the Mixture of Experts (MoE) layer goes to a Gating Network. The output of the Gating Networks and the output of the shared hidden layer are inputs for the objective functions such as engagement and satisfaction. A sigmoid activation function represents each objective function.
  
Since users can have different types of behaviors towards recommended items, our ranking system can support multiple objectives. Each objective is to predict one type of user behavior related to user utility. The objectives are separate into two categories: engagement objectives and satisfaction objectives. 
    
*	Engagement objectives capture user behaviors such as clicks and watches. The prediction of these behaviors can be formulated into two types of tasks: binary classification task for behaviors such as clicks, and regression task for behaviors related to time spent. 
    
*	Similarly, for satisfaction objectives, the prediction of behaviors can be related to user satisfactions into either binary classification task or regression task. For example, behavior such as clicking like for a video is formulated as a binary classification task, and behavior such as rating is formulated as regression task. 
  
For binary classification tasks, we are computing cross entropy loss. And for regression tasks, we compute squared loss.
  
#### 6.2.2.	Shallow Tower Overview
We are using implicit feedback data for training because explicit feedback data which are ideal for training are not available or are expensive, might not be ideal as there is a usual bias in this data which can increase with feedback. To manage the bias, a shallow tower is introduced into the model architecture. It is addressing the challenge as below:
    
*	The shallow tower is trained using features that contribute to the bias like position of the recommendation and tries to predict whether there is a bias component involved in the current instance. 
    
*	It is removing the selection bias, takes input related to the selection bias, e.g., ranking order decided by the current system, and outputs a scalar serving as a bias term to the final prediction of the main model.

*	The Shallow tower factorizes the label in training data in two parts the unbiased user utility learned from the main model, and the estimated propensity score learned from the shallow tower.

## 7. Experimentation
### 7.1. Challenges:
The challenges of Recommendation system as stated earlier:

*	There are often different and sometimes conflicting objectives which we want to optimize for. For example, we may want to recommend videos that users rate highly and share with their friends, in addition to watching.
*	There is often implicit bias in the system. For example, a user might have clicked and watched a video simply because it was being ranked high, not because it was the one that the user liked the most. Therefore, models trained using data generated from the current system will be biased, causing a feedback loop effect.

### 7.2. Addressing Multi Objective Function:
There are two common objective functions that may create a conflict and should be optimized:
*	Engagement Objectives: These objectives can be measured using dataset on clicks, time spent of the user while watching the recommended video, etc.
*	Satisfaction Objective: These objectives can be measured by dataset in likes, shares, comments, rating, etc.

Both these objectives contain:
*	binary classification tasks (click or not, like or not, etc.)
*	regression tasks (time spent, rating given, etc.)

### 7.3. Removal of Implicit Bias
For training the model, the used dataset contains some implicit bias: 
  
The reason is because a user historically might have clicked and watched a video simply because it was being ranked high, not because it was the one that the user liked the most. 
  
So, if the model is trained using such data, it will produce biased non-optimal recommendations which the user might not like.

## 8. Conclusion:
  
In this project, we are trying to replicate the results of a Recommendation Engine using MMoE as ranking generation. We try to address the challenges in designing and developing industrial recommendation systems, especially ranking systems. These challenges include the presence of multiple competing ranking objectives, as well as implicit selection biases in user feedback. 
  
To tackle these challenges, we have used a large-scale multi-objective ranking system and applied it to the problem of recommending what video to watch next. To efficiently optimize multiple ranking objectives, we extended Multi-gate Mixture-of-Experts model architecture to utilize soft-parameter sharing.
  
Below are the results from tensor board for the training of recommendation engine and we can see with each epoch the loss in context of user rating is reducing. What it means is that Users are giving more positive rating# for the model
  
NOTE: # We are simulating the user experience as real-world data was unavailable.

## 9. References:
Below is the list of references used for this project:
  
1.	Zhao, Z., Hong, L., Wei, L., Chen, J., Nath, A., Andrews, S., . . . Chi, E. (2019). Recommending what video to watch next. Proceedings of the 13th ACM Conference on Recommender Systems. doi:10.1145/3298689.3346997

2.	Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., &amp; Chi, E. H. (2018). Modeling Task Relationships in Multi-task Learning with Multi-Gate-Mixture-of-Experts. Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining. doi:10.1145/3219819.3220007

3.	Bhatia, S. (2020, February 22). A Multitask Ranking System: How YouTube recommends the Next Videos. Retrieved December 11, 2020, from https://medium.com/@bhatia.suneet/a-multitask-ranking-system-how-youtube-recommends-the-next-videos-a23a63476073

4.	Mitchell, J. (2017). Trending YouTube Video Statistics. Retrieved 2019, from https://www.kaggle.com/datasnaek/youtube-new.

5.	Jonathan Baxter et al. 2000. A model of inductive bias learning. J. Artif. Intell. Res.(JAIR) 12, 149-198 (2000), 3.

6.	Rich Caruana. 1998. Multitask learning. In Learning to learn. Springer, 95–133.

7.	Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098 (2017).

8.	R Caruna. 1993. Multitask learning: A knowledge-based source of inductive bias. In Machine Learning: Proceedings of the Tenth International Conference. 41–48.

9.	Geoffrey Hinton, Oriol Vinyals, and JeDean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015).

10.	David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. 2013. Learning factored representations in a deep mixture of experts. arXiv:1312.4314 (2013).

11.	Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Georey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).

12.	Chrisantha Fernando, Dylan Banarse, Charles Blundell, . . .  Daan Wierstra. 2017. Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint arXiv:1701.08734 (2017).

13.	Jeffrey Dean, Greg Corrado, Rajat Monga,  . . .  , Quoc V Le, et al. 2012. Large scale distributed deep networks. In Advances in neural information processing systems. 1223–1231.

14.	Trapit Bansal, David Belanger, and Andrew McCallum. 2016. Ask the gru: Multitask learning for deep text recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 107–114.

15.	Melvin Johnson, Mike Schuster, Quoc V Le, . . . , Greg Corrado, et al. 2016. Google’s multilingual neural machine translation system: enabling zeroshot translation. arXiv preprint arXiv:1611.04558 (2016)

16.	Xia Ning and George Karypis. 2010. Multi-task learning for recommender system. In Proceedings of 2nd Asian Conference on Machine Learning. 269–284.

17.	Zhe Zhao, Zhiyuan Cheng, Lichan Hong, and EdHChi. 2015. Improving user topic interest profiles by behavior factorization. In Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 1406–1416.
