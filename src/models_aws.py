# En hadoop: /opt/spark-3.0.1/bin/pyspark --master spark://hadoop-master:7077
# spark-submit --conf spark.jars.ivy=/tmp/.ivy /intercambio/models.py
# exec(open('/intercambio/models.py', encoding="utf-8").read())

# sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
# sudo chmod +x /usr/local/bin/docker-compose

'''
aws emr create-cluster --applications Name=Spark Name=Zeppelin --ec2-attributes '{"KeyName":"spark","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-731b7d19","EmrManagedSlaveSecurityGroup":"sg-0a2bf65c779ae46d3","EmrManagedMasterSecurityGroup":"sg-0f45869b481f32b74"}' --service-role EMR_DefaultRole --enable-debugging --release-label emr-6.3.0 --log-uri 's3n://bucket-pepitoenpeligro/' --name 'PepeCluster' --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master Instance Group"},{"InstanceCount":2,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"CORE","InstanceType":"m5.xlarge","Name":"Core Instance Group"}]' --configurations '[{"Classification":"spark","Properties":{}}]' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region eu-central-1
'''

import sys
import os.path
from time import *
import pyspark.sql.functions as func

# Librerias Core de spark
from pyspark import SparkContext, SparkConf, sql

from pyspark.sql.functions import udf
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType, DoubleType, IntegerType
from pyspark.sql import SparkSession
from functools import reduce
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml import Pipeline

# Libreria MLKit de Spark
from pyspark.ml.linalg import *
from pyspark.ml.feature import *
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import *
from pyspark.ml.classification import *
from pyspark.ml import *

# Neural Network: https://runawayhorse001.github.io/LearningApacheSpark/fnn.html
# Random Forest: https://runawayhorse001.github.io/LearningApacheSpark/regression.html?highlight=random%20forest#random-forest-regression
# Decision Tree: https://runawayhorse001.github.io/LearningApacheSpark/classification.html#id5
# Gradient Boost Tree: https://runawayhorse001.github.io/LearningApacheSpark/classification.html#gradient-boosted-tree-classification
# Binomial Logistic Regression: https://runawayhorse001.github.io/LearningApacheSpark/classification.html#binomial-logistic-regression

# docker exec -it --user root ubuntu_spark_1 pip install numpy
# docker exec -it  ubuntu_spark_1 /bin/bash
# spark-submit --master spark://spark:7077 --total-executor-cores 4 --executor-memory 4g /intercambio/models.py
# docker exec -it  ubuntu_spark_1 /bin/bash spark-submit --master spark://spark:7077 --total-executor-cores 4 --executor-memory 8g /intercambio/models.py

title = "CC-P4-Modelos"
name_file="s3n://bucket-pepitoenpeligro/pepitoenpeligro"

columns = ['`PredCN_central_2`', '`PredSS_r1_4`', '`PSSM_r1_3_T`', '`AA_freq_central_M`', '`PSSM_r2_-1_L`', '`PSSM_r1_-2_R`']
columns_asIndex= ['PredCN_central_2', 'PredSS_r1_4', 'PSSM_r1_3_T', 'AA_freq_central_M', 'PSSM_r2_-1_L', 'PSSM_r1_-2_R']

portionTrain = 0.8
portionTest = 0.2

def predictions(estimator, paramGrid, dataTrain, dataTest):
    # binary clasification
    # https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#binary-classification
    train_validator = TrainValidationSplit(estimator=estimator, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), trainRatio=portionTrain)
    model = train_validator.fit(dataTrain)
    predictions = model.transform(dataTest)
    predictionAndLabel = predictions.select("prediction","label")

    # convierte labels y prediccones a float
    predictionAndLabel = predictionAndLabel.withColumn("prediction", func.round(predictionAndLabel['prediction']).cast('float'))
    predictionAndLabel = predictionAndLabel.withColumn("label", func.round(predictionAndLabel['label']).cast('float'))
    metrics=MulticlassMetrics(predictionAndLabel.select("prediction","label").rdd.map(tuple))


    evaluator = BinaryClassificationEvaluator()
    auRocRF = evaluator.evaluate(predictions)


    # la matriz de confusion revienta
    cnf_matrix = metrics.confusionMatrix()
    accuracy = round(metrics.accuracy*100, 3)
    f1 = metrics.fMeasure(1.0)
    recall = metrics.recall(1.0)

    print("Results of model %s"  % (estimator.__dict__['uid']))
    print("Accuracy %s" % accuracy)
    print("F1 %s" % f1)
    print("Recall %s" % recall)
    print("AUC %s" % auRocRF)

    return predictions, model

# Ya tengo captura de esta ejecucion
def random_forest_1(trainingData,testData):
    print("[Random Forest] init")
    start_time = time()
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=12345)
    # ParamGridBuilder params:
    # https://spark.apache.org/docs/latest/ml-tuning.html
    paramGridRF = ParamGridBuilder().addGrid(rf.numTrees, [5, 10, 20]).addGrid(rf.maxDepth, [2, 3, 6]).build()
    
    predictionsRF, mRF = predictions(rf,paramGridRF,trainingData,testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[Random Forest] With params %s" % paramGridRF)
    print("[Random Forest] time %s" %(elapsed_time))

# Ya tengo captura de esta ejecucion
def random_forest_2(trainingData,testData):
    print("[Random Forest] init")
    start_time = time()
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=2021)
    # ParamGridBuilder params:
    # https://spark.apache.org/docs/latest/ml-tuning.html
    paramGridRF = ParamGridBuilder().addGrid(rf.numTrees, [10, 30, 60]).addGrid(rf.maxDepth, [3, 6, 12]).build()
    predictionsRF, mRF = predictions(rf,paramGridRF,trainingData,testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[Random Forest] With params %s" % paramGridRF)
    print("[Random Forest] time %s" %(elapsed_time))


def gradient_boosted_tree_1(trainingData, testData):
    print("[Gradient Boosted Tree] init")
    start_time = time()
    gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=2021)
    #paramGridGBT = ParamGridBuilder().addGrid(gbt.maxIter, [10, 15, 20]).addGrid(gbt.maxDepth, [3, 6, 12]).build()
    paramGridGBT = ParamGridBuilder().addGrid(gbt.maxIter, [5, 10, 15]).addGrid(gbt.maxDepth, [2, 3, 9]).build()
    predictionsGBT, mGBT = predictions(gbt,paramGridGBT,trainingData,testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[Gradient Boosted Tree] With params %s" % paramGridGBT)
    print("[Gradient Boosted Tree] time %s" %(elapsed_time))


def gradient_boosted_tree_2(trainingData, testData):
    print("[Gradient Boosted Tree] init")
    start_time = time()
    gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=2021)
    paramGridGBT = ParamGridBuilder().addGrid(gbt.maxIter, [10, 15, 20]).addGrid(gbt.maxDepth, [3, 6, 12]).build()
    
    predictionsGBT, mGBT = predictions(gbt,paramGridGBT,trainingData,testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[Gradient Boosted Tree] With params %s" % paramGridGBT)
    print("[Gradient Boosted Tree] time %s" %(elapsed_time))


def perceptron_1(trainingData, testData):
    print("[Peceptron] init")
    start_time = time()
    mlp = MultilayerPerceptronClassifier(
       featuresCol="features",
       labelCol="label",
       predictionCol="prediction",
       maxIter=100
    )
    mlpGrid = ParamGridBuilder().addGrid(mlp.layers, [[7, 3, 2], [7, 9, 3, 2], [7, 5, 2]]).build()
    predictionsMLP, mMLP = predictions(mlp, mlpGrid, trainingData, testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[Peceptron] With params %s" % predictionsMLP)
    print("[Peceptron] time %s" %(elapsed_time))


# Ya tengo captura de esta ejecucion
def logistic_regresion1(trainingData, testData):
    print("[Logistic Regression] init")
    start_time = time()
    lr = LogisticRegression(featuresCol="features",labelCol="label",maxIter=100,family="multinomial")
    lrGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.001]).addGrid(lr.elasticNetParam, [0.5, 0.6, 0.8]).build()
    predictionsRL, mRL = predictions(lr,lrGrid,trainingData,testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[Logistic Regression] With params %s" % predictionsRL)
    print("[Logistic Regression] time %s" %(elapsed_time))

def logistic_regresion_2(trainingData, testData):
    print("[Logistic Regression] init")
    start_time = time()
    lr = LogisticRegression(featuresCol="features",labelCol="label",maxIter=100,family="multinomial")
    lrGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.001]).addGrid(lr.elasticNetParam, [0.6, 0.7, 0.9]).build()
    predictionsRL, mRL = predictions(lr,lrGrid,trainingData,testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[Logistic Regression] With params %s" % predictionsRL)
    print("[Logistic Regression] time %s" %(elapsed_time))


def naive_bayes_1(trainingData, testData):
    print("[NaiveBayes] init")
    start_time = time()
    nb = NaiveBayes(modelType="multinomial", featuresCol="features", labelCol="label", smoothing=1.0)
    nbGrid = ParamGridBuilder().addGrid(1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).build()
    predictionsNB, mNB = predictions(nb,nbGrid,trainingData,testData)
    end_time = time()
    elapsed_time = end_time - start_time
    print("[NaiveBayes] With params %s" % predictionsNB)
    print("[NaiveBayes] time %s" %(elapsed_time))


if __name__ == "__main__":
    print("Iniciando el contexto de Spark %s", title)
    configurationSpark = SparkConf().setAppName(title)
    sparkContexto = SparkContext.getOrCreate(conf=configurationSpark)
    sqlContext = sql.SQLContext(sparkContexto)

    df_columns = sqlContext.read.csv(name_file, sep=",", header=True, inferSchema=True)
    indexer = StringIndexer(inputCol="PredSS_r1_4", outputCol="PredSS_r1_4_indexado")
    df_columns = indexer.fit(df_columns).transform(df_columns)
    df_columns = df_columns.drop("PredSS_r1_4")
    df_columns = df_columns.withColumnRenamed("PredSS_r1_4_indexado","PredSS_r1_4")
    df_columns.show(20)

    clases_negativas = df_columns.filter(df_columns['class']==0).count()
    clases_positivas = df_columns.filter(df_columns['class']==1).count()
    print("El balanceo negativo/positivio es: %s / %s" % (clases_negativas, clases_positivas))

    #Me quedo con el numero de clases de la menor 
    tam_partition = clases_positivas
    if(clases_positivas > clases_negativas):
        tam_partition = clases_negativas

    # Reduzco ambos al tamaño de la particion anterior: Undersampling
    df_0 = df_columns.filter(df_columns['class'] == 0).limit(tam_partition)
    df_1 = df_columns.filter(df_columns['class'] == 1).limit(tam_partition)

    df_balanced = df_1.union(df_0)
    df_train, df_test = df_balanced.randomSplit([portionTrain, portionTest])
    df_balanced_count = df_balanced.select('class').count()

    df_train_count = df_train.select('class').count()
    df_train_negative_count = df_train.filter(df_columns['class']==0).select('class').count() 
    df_train_positive_count = df_train.filter(df_columns['class']==1).select('class').count() 
    
    df_test_count = df_test.select('class').count()
    df_test_negative_count = df_test.filter(df_columns['class']==0).select('class').count() 
    df_test_positive_count = df_test.filter(df_columns['class']==1).select('class').count()

    print("[Global] total: %s", df_balanced_count)
    print("[Train] positivas: %s, negativas %s, total %s" % (df_train_positive_count, df_train_negative_count, df_train_count ))
    print("[Test] positivas: %s, negativas %s, total %s" % (df_test_positive_count, df_test_negative_count, df_test_count ))

    # Feature Transformer VectorAssembler in PySpark ML Feature 
    # https://medium.com/@nutanbhogendrasharma/feature-transformer-vectorassembler-in-pyspark-ml-feature-part-3-b3c2c3c93ee9
    assembler = VectorAssembler(inputCols=columns_asIndex, outputCol='features')
    trainingData = assembler.transform(df_train).select("features","class").withColumnRenamed("class","label")
    testData = assembler.transform(df_test).select("features","class").withColumnRenamed("class","label")


    # RandomForest - OK
    #random_forest_1(trainingData, testData)
    #random_forest_2(trainingData, testData)
    

    # Gradient Boosted Tree - OK
    #gradient_boosted_tree_1(trainingData,testData)
    #gradient_boosted_tree_2(trainingData,testData)

    # Regresion logistica- OK
    #logistic_regresion1(trainingData, testData)
    logistic_regresion_2(trainingData, testData)


    # Perceptron multicapa
    # perceptron_1(trainingData,testData)

    

    # Naive Bayes
    # naive_bayes_1(trainingData, testData)


    # https://stackoverflow.com/questions/60772315/how-to-evaluate-a-classifier-with-pyspark-2-4-5
    # https://stackoverflow.com/questions/41714698/how-to-get-accuracy-precision-recall-and-roc-from-cross-validation-in-spark-ml
    print("FIN")
    