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

# Values to store
rmse_value = str(test_results.rootMeanSquaredError)
r2_value = str(test_results.r2)

# Connect to HBase
# Try 'hbase' first; if that fails in your environment, switch back to 'master'
connection = happybase.Connection('hbase')
connection.open()

# Use your existing HBase table and column family
table = connection.table('zoo_data')

# Row key where we'll store the metrics
row_key = b'zoo_metrics1'

# Put RMSE and R2 into cf:rmse and cf:r2
table.put(
    row_key,
    {
        b'cf:rmse': rmse_value.encode('utf-8'),
        b'cf:r2':   r2_value.encode('utf-8')
    }
)

connection.close()

# Step 9: Stop Spark session
spark.stop()


