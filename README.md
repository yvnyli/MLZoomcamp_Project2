# Steam Game Critic Score
Give ~80K games a score similar to Metacritic. See what the model says about your favorite games. Check out a game's score before you buy.

ML Zoomcamp 2025: If you are here for peer review, please check out [Guide_for_evaluators.md](https://github.com/yvnyli/MLZoomcamp_Project2/blob/main/Guide_for_evaluators.md) for a map of where things are. Thank you.

## Dataset:

### **Steam Games Dataset** ([link](https://huggingface.co/datasets/FronkonGames/steam-games-dataset))

This dataset contains information about games published on the largest PC game platform, Steam. 

There are **83560 unique games** (rows) with their basic information such as name, publisher, language, and platform; marketing information such as description, genres, tags, header image; as well as rating and reviews.

An interesting thing is that only **a very small portion (5%) of the games received a Metacritic score**. Some of this might be due to a mismatch in data scraping. But given that there are only ~6K game ratings on Metacritic, vs. ~83K games in the dataset, it is true that most games have not gained attention of game critics. 

However, not being rated does not necessarily mean that a game is low quality or not worth playing. It might be due to low budget in marketing or the publisher being a smaller profile one. In other words, there could be **hidden gems** in the vast majority of games that flew under the radar.

## Modeling Problem:

Therefore, the goal of the project is to build a model that can **predict Metacritic score**, in the hopes to **find underrated great games**.

## Modeling Strategy:

The **3910 games with ground-truth Metacritic scores** are used for training (validation and tuning) and testing. I will try different classes of models and tune their hyperparameter. The winning model based on test accuracy will be used to predict Metacritic scores on games that don't have it. The accuracy of the predictions should be similar to the accuracy on the test set.

Bonus: I also train a classifier for probability of having Metacritic score. The idea is, predicted score might be reliable for games similar to the training games, but unreliable for games that are more different. The probability measures this similarity. I then labeled games into 4 **confidence tiers**: very low, low, medium, and high, which puts a grain of salt on the interpretation of the scores.


## Results: 

Data processing: 
- Numerics: There are 15 numerical columns. In all of them, the dominant mode is 0, making the rest of the distribution hard to see/take into effect. So I made two features out of each column. One is a binary indicating whether the value is zero (zero actually means missing data in many of them). The other is the value, with log transformation applied for performance.

  ![Histograms of numerics](https://www.markdownlang.com/markdown-logo.png)
  
- Release date: The only datetime column, which I turned into a numeric feature by subtracting the Epoch (days since 1/1/1970).
- Multi-hot: There are 5 columns containing lists of categorical labels, which can be represented by multi-hot vectors. These columns are categories, genres, tags, languages, and audio languages. Word clouds below. The problem is that there are 732 multi-hot features out of the 5 columns, which is too many for our training data size (732:3910 is roughly 1:5). So I used SVD to reduce dimensionality down to 70 features. This also eliminated colinearity.

Models: I trained linear (Elastic net), tree based (XGBoost), and neural network (multi-layer perceptron) regressors. Tuned hyperparameter for each model class on 5-fold CV of 85% data. The best XGBoost model had higher performance than the other two on the 15% testing data, and was used to make ~80K predictions.


## Try it out:

Cloud deployment: 





## Next ideas:
There are some columns containing a lot of information that could not be used directly or by multi-hot encoding. These include description and reviews which are long form text, and header image, screenshots, and movies which are media. In the next step to improve my model, I would use a pretrained neural network to turn the text and media into embeddings, which are vectors (multiple numerical columns) that our models can use. I might use [CLIP](https://github.com/openai/CLIP), which can embed image and text into similar representation.
