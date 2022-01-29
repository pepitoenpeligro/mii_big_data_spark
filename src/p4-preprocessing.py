# spark-submit  /p4-preprocessing.py
# docker exec spark_things_spark_1 pip install numpy
# docker exec spark_things_spark_1 spark-submit /p4-preprocessing.py


# ['PredCN_central_2', 'PredSS_r1_4', 'PSSM_r1_3_T', 'AA_freq_central_M', 'PSSM_r2_-1_L', 'PSSM_r1_-2_R']

import sys
import time
from pyspark import SparkContext, SparkConf, sql
from pyspark.ml.classification import LogisticRegression
from functools import reduce

'''
Creamos la configuración del contexto Spark y del SQL Context
'''

configurationSpark = SparkConf().setAppName("CC-P4-Preprocesado")
sparkContexto = SparkContext.getOrCreate(conf=configurationSpark)
sqlContext = sql.SQLContext(sparkContexto)
name_output="pepitoenpeligro"


if __name__ == "__main__":
    start = time.time()
    print("Comenzando el preprocesado")
    ficheroCabeceras = sparkContexto.textFile("s3n://bucket-pepitoenpeligro/raw_data/ECBDL14_IR2.header").collect()
    cabecerasFiltradas = filter(lambda line: "@attribute" in line ,ficheroCabeceras)
    print("Mapeando")
    mapHeaders = list(map(lambda line: line.split()[1], cabecerasFiltradas))
    
    print("Leyendo en un dataframe los datos del dataset pesado")
    df = sqlContext.read.csv("s3://bucket-pepitoenpeligro/raw_data/ECBDL14_IR2.data",header=False,sep=",",inferSchema=True)
    print("Reduciendo")
    dfReducido = reduce(lambda data, idx: data.withColumnRenamed(df.schema.names[idx], mapHeaders[idx]), range(len(df.schema.names)), df)
    
    dfReducido.createOrReplaceTempView("sql_dataset")

    columns= ['`PredCN_central_2`', '`PredSS_r1_4`', '`PSSM_r1_3_T`', '`AA_freq_central_M`', '`PSSM_r2_-1_L`', '`PSSM_r1_-2_R`']
    print("Seleccionando las columnas {%s, %s, %s, %s, %s, %s}" % (columns[0], columns[1], columns[2], columns[3], columns[4], columns[5]))
    sqlDF = sqlContext.sql('SELECT %s, %s, %s, %s, %s, %s, class FROM sql_dataset' % (columns[0], columns[1], columns[2], columns[3], columns[4], columns[5]))

    
    print("Escribiendo en el fichero csv")
    sqlDF.write.format('csv').option('header',True).save('s3n://bucket-pepitoenpeligro/%s' % (name_output))
    
    print("Fin del preprocesado")
    print("[3] Hemos Seleccionando las columnas {%s, %s, %s, %s, %s, %s}" % (columns[0], columns[1], columns[2], columns[3], columns[4], columns[5]))
    end = time.time()
    print("Tiempo consumido en la seleccion de columnas: %s" % (end - start))
    sparkContexto.stop()