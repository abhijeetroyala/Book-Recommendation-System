ratings = sqlContext.read.csv('/FileStore/tables/ratings.csv', header=True, inferSchema=True)


# Explore the data with Spark APIs
ratings.show(truncate=False)


#Print the schema of the DataFrame:
ratings.printSchema()


books = spark.read.load("/FileStore/tables/books.csv",
                     format="csv", inferSchema="true", header="true")


# Explore the data with Spark APIs
books.show(truncate=False)


#Print the schema of the DataFrame:
books.printSchema()


ratings.describe().show()


print('Number of different users: {}'.format(ratings.select('user_id').distinct().count()))
print('Number of different books: {}'.format(ratings.select('book_Id').distinct().count()))
print('Number of books with at least one rating strictly higher than 4: {}'.format(ratings.filter('rating > 4').select('book_Id').distinct().count()))


ratings.createOrReplaceTempView('ratings')
spark.sql('SELECT COUNT(DISTINCT(book_id)) AS nb FROM ratings WHERE rating > 4').show()


import pandas as pd
ratings.toPandas().head()


import seaborn as sns
ratingsPandas = ratings.toPandas()
sns.lmplot(x='user_id', y='book_id', data=ratingsPandas, fit_reg=False);


display()


sns.palplot(sns.diverging_palette(10, 133, sep=80, n=5))
display()


lm = sns.lmplot(x='user_id', y='book_id', hue='rating', data=ratingsPandas, fit_reg=False, aspect=2, palette=sns.diverging_palette(10, 133, sep=80, n=5))
axes = lm.axes
axes[0, 0].set_ylim(0, 10000) # max book_id is 163949
axes[0, 0].set_xlim(0,  53424) # max userId is 671
lm;


display()


sns.violinplot([ratingsPandas.rating]);


display()


# Train the Model
from pyspark.ml.recommendation import ALS
model = ALS(userCol='user_id', itemCol='book_id', ratingCol='rating').fit(ratings)


#Run the Model
predictions = model.transform(ratings)
predictions.toPandas().head()


#Evaluate the Model
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
print('The root mean squared error for our model is: {}'.format(evaluator.evaluate(predictions)))


#Split the dataset 
(trainingRatings, testRatings) = ratings.randomSplit([80.0, 20.0])


als = ALS(userCol='user_id', itemCol='book_id', ratingCol='rating')
model = als.fit(trainingRatings)
predictions = model.transform(testRatings)


predictions.toPandas().head()


evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
print('The root mean squared error for our model is: {}'.format(evaluator.evaluate(predictions)))


#Handle  NAN value
avgRatings = ratings.select('rating').groupBy().avg().first()[0]
print ('The average rating in the dataset is: {}'.format(avgRatings))

evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
print ('The root mean squared error for our model is: {}'.format(evaluator.evaluate(predictions.na.fill(avgRatings))))


evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
print ('The root mean squared error for our model is: {}'.format(evaluator.evaluate(predictions.na.drop())))


#improve the performance score
def repeatALS(data, k=3, userCol='user_id', itemCol='book_id', ratingCol='rating', metricName='rmse'):
    evaluations = []
    for i in range(0, k):  
        (trainingSet, testingSet) = data.randomSplit([k - 1.0, 1.0])
        als = ALS(userCol=userCol, itemCol=itemCol, ratingCol=ratingCol)
        model = als.fit(trainingSet)
        predictions = model.transform(testingSet)
        evaluator = RegressionEvaluator(metricName=metricName, labelCol='rating', predictionCol='prediction')
        evaluation = evaluator.evaluate(predictions.na.drop())
        print('Loop {}: {} = {}'.format(i + 1, metricName, evaluation))
        evaluations.append(evaluation)
    return sum(evaluations) / float(len(evaluations))


print('RMSE = {}'.format(repeatALS(ratings, k=4)))


def kfoldALS(data, k=3, userCol='user_id', itemCol='book_id', ratingCol='rating', metricName='rmse'):
    evaluations = []
    weights = [1.0] * k
    splits = data.randomSplit(weights)
    for i in range(0, k):  
        testingSet = splits[i]
        trainingSet = spark.createDataFrame(sc.emptyRDD(), data.schema)
        for j in range(0, k):
            if i == j:
                continue
            else:
                trainingSet = trainingSet.union(splits[j])
        als = ALS(userCol=userCol, itemCol=itemCol, ratingCol=ratingCol)
        model = als.fit(trainingSet)
        predictions = model.transform(testingSet)
        evaluator = RegressionEvaluator(metricName=metricName, labelCol='rating', predictionCol='prediction')
        evaluation = evaluator.evaluate(predictions.na.drop())
        print('Loop {}: {} = {}'.format(i + 1, metricName, evaluation))
        evaluations.append(evaluation)
    return sum(evaluations) / float(len(evaluations))


#Compute the average performance score for 4 folds:
print('RMSE = {}'.format(kfoldALS(ratings, k=4)))


#Now compute the average performance score for 10 folds:
print('RMSE = {}'.format(kfoldALS(ratings, k=10)))


#improve the model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

(trainingRatings, validationRatings) = ratings.randomSplit([80.0, 20.0])
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

#paramGrid = ParamGridBuilder().addGrid(als.rank, [1, 5, 10]).addGrid(als.maxIter, [20]).addGrid(als.regParam, [0.05, 0.1, 0.5]).build()
paramGrid = ParamGridBuilder().build()

crossval = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
cvModel = crossval.fit(trainingRatings)
predictions = cvModel.transform(validationRatings)

print('The root mean squared error for our model is: {}'.format(evaluator.evaluate(predictions.na.drop())))


#Create a recommendBooks function:

from pyspark.sql.functions import lit

def recommendBooks(model, user, nbRecommendations):
    # Create a Spark DataFrame with the specified user and all the books listed in the ratings DataFrame
    dataSet = ratings.select('book_id').distinct().withColumn('user_id', lit(user))

    # Create a Spark DataFrame with the books that have already been rated by this user
    booksAlreadyRated = ratings.filter(ratings.user_id == user).select('book_id', 'user_id')

    # Apply the recommender system to the data set without the already rated books to predict ratings
    predictions = model.transform(dataSet.subtract(booksAlreadyRated)).dropna().orderBy('prediction', ascending=False).limit(nbRecommendations).select('book_id', 'prediction')

    # Join with the books DataFrame to get the books titles and genres
    recommendations = predictions.join(books, predictions.book_id == books.book_id).select(predictions.book_id, books.title, predictions.prediction)

#     recommendations.show(truncate=False)
    return recommendations


#Now run this function to recommend 10 movies for three different users:
print('Recommendations for user 133:')
recommendBooks(model, 1185, 10).toPandas()



