from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import happybase

# Step 1: Create a Spark session with Hive support
spark = SparkSession.builder \
    .appName("MLlib Zoo Legs Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

# Step 2: Load data from Hive table 'zoo_data'
# Assuming schema: animal_name, hair, feathers, eggs, milk, airborne, aquatic,
# predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize
zoo_df = spark.sql("""
    SELECT
        hair,
        feathers,
        eggs,
        milk,
        airborne,
        aquatic,
        predator,
        toothed,
        backbone,
        breathes,
        venomous,
        fins,
        tail,
        domestic,
        catsize,
        legs
    FROM zoo_data
""")

# Step 3: Handle null values
zoo_df = zoo_df.na.drop()

# Step 4: Assemble feature vector (all attributes except 'legs')
feature_cols = [
    "hair", "feathers", "eggs", "milk",
    "airborne", "aquatic", "predator", "toothed",
    "backbone", "breathes", "venomous", "fins",
    "tail", "domestic", "catsize"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

assembled_df = assembler.transform(zoo_df).select("features", "legs")

# Step 5: Split into train and test sets
train_data, test_data = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Step 6: Train a Linear Regression model to predict legs
lr = LinearRegression(labelCol="legs", featuresCol="features")
lr_model = lr.fit(train_data)

# Step 7: Evaluate the model
test_results = lr_model.evaluate(test_data)

# Step 8: Print metrics
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"R^2: {test_results.r2}")

# ---- Write metrics to HBase with happybase ----
metrics_data = [
    ('zoo_metrics1', 'cf:rmse', str(test_results.rootMeanSquaredError)),
    ('zoo_metrics1', 'cf:r2',   str(test_results.r2)),
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')   
    connection.open()
    table = connection.table('zoo_data')          
    for row in partition:
        row_key, column, value = row
        table.put(row_key, {column: value.encode('utf-8')})
    connection.close()

rdd = spark.sparkContext.parallelize(metrics_data)
rdd.foreachPartition(write_to_hbase_partition)

# Step 9: Stop Spark session
spark.stop()

