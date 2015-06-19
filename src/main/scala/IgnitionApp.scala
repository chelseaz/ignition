import java.nio.charset.Charset
import java.nio.file.{Files, Paths}
import java.sql.Timestamp

import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext, UserDefinedFunction}
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.DateTime
import org.json4s._
import org.json4s.native.JsonMethods._

import scala.collection.JavaConversions._

object InputTransformer {
  // http://stackoverflow.com/questions/29383107/idiomatic-way-to-change-column-types-in-spark-dataframe
  val toTimestamp = udf[Timestamp, String](s => new Timestamp(s.toLong*1000))
  val toLatLongPairs = udf[Seq[Seq[Double]], String](s =>
    parse(s) match {
      case JArray(l) => l.map(_ match {
        case JArray(List(JDouble(long), JDouble(lat))) => Seq(long, lat)
      })
    }
  )
  val extractHour = udf[Int, String](s => new DateTime(s.toLong*1000).getHourOfDay())
  val extractDayOfWeek = udf[Int, String](s => new DateTime(s.toLong*1000).getDayOfWeek())
  val extractLastLat = udf[Option[Double], Seq[Seq[Double]]](_.lastOption.map(_.last))
  val extractLastLong = udf[Option[Double], Seq[Seq[Double]]](_.lastOption.map(_.head))
  def positionAt(minutes: Int): UserDefinedFunction =
    udf[Vector, Seq[Seq[Double]]](polyline => {
      val timeIndex = minutes * 4
      if (polyline.length > timeIndex) {
        Vectors.dense(polyline(timeIndex).toArray)
      } else {
        Vectors.sparse(2, Nil)
      }
    })
  def deltasAfter(minutes: Int): UserDefinedFunction =
    udf[Vector, Seq[Seq[Double]]](polyline => {
      val timeIndex = minutes * 4
      if (polyline.length > timeIndex) {
        val x_0 = polyline(0)
        val x_n = polyline(timeIndex)
        Vectors.dense(Array(x_n(0)-x_0(0), x_n(1)-x_0(1)))
      } else {
        Vectors.sparse(2, Nil)
      }
    })
  val toTripSeconds = udf[Double, Seq[Seq[Double]]](_.length*15)
  val toId = udf[Long, String](_.replace("T", "").toLong)

  def transform(rawDf: DataFrame): DataFrame = {
    val trainingBase = rawDf
      .withColumn("tripIdNum", toId(rawDf("TRIP_ID")))
      .withColumn("hourOfDay", extractHour(rawDf("TIMESTAMP")))
      .withColumn("dayOfWeek", extractDayOfWeek(rawDf("TIMESTAMP")))
      .withColumn("polyline", toLatLongPairs(rawDf("POLYLINE")))
      .filter(rawDf("MISSING_DATA") equalTo "False")
      .filter(rawDf("POLYLINE") notEqual "[]")

    trainingBase
      .withColumn("startPos", positionAt(0)(trainingBase("polyline")))
      .withColumn("1mindelta", deltasAfter(1)(trainingBase("polyline")))
      .withColumn("5mindelta", deltasAfter(5)(trainingBase("polyline")))
      .withColumn("15mindelta", deltasAfter(15)(trainingBase("polyline")))
      .withColumn("lastLat", extractLastLat(trainingBase("polyline")))
      .withColumn("lastLong", extractLastLong(trainingBase("polyline")))
      .withColumn("tripSeconds", toTripSeconds(trainingBase("polyline")))
      .select("TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID",
        "tripIdNum", "hourOfDay", "dayOfWeek",
        "startPos", "1mindelta", "5mindelta", "15mindelta",
        "lastLat", "lastLong", "tripSeconds")
  }
}

trait Model {
  val predictionFile: String
  val submissionFile: String
  val headerLine: String

  def saveSubmission() = {
    val lines = Files.readAllLines(Paths.get(predictionFile), Charset.defaultCharset())
    val newContent = (headerLine :: lines.toList).mkString("\n")
    Files.write(Paths.get(submissionFile), newContent.getBytes)
  }
}

object TripLengthModel extends Model {

  val predictionPath = "../length-preds"
  val predictionFile = s"$predictionPath/part-00000"
  val submissionFile = s"../submission/length-${DateTime.now().toString("yyyyMMddHHmmss")}.csv"
  val headerLine = "TRIP_ID,TRAVEL_TIME"

  def build(training: DataFrame, pipeline: Pipeline): PipelineModel = {
    val tripLengthTraining = training.withColumn("label", training("tripSeconds"))
    val model = pipeline.fit(tripLengthTraining)

    val trainingPredictions = model.transform(tripLengthTraining).select("features", "label", "prediction")
    println("Trip length training RMSE = " + new RegressionEvaluator().evaluate(trainingPredictions))

    model
  }

  def test(test: DataFrame, model: PipelineModel): Unit = {
    val predictions = model.transform(test).sort("tripIdNum")
    //    testPredictions.show()

    // don't predict less than the current travel time in the test set!
    val maxColumn = udf[Double, Double, Double]((pred, curTripSeconds) => pred.max(curTripSeconds))

    predictions
      .withColumn("TRAVEL_TIME", maxColumn(predictions("prediction"), predictions("tripSeconds")))
      .select("TRIP_ID", "TRAVEL_TIME")
      .coalesce(1)
      .write.format("com.databricks.spark.csv").mode("overwrite").save(predictionPath)

    saveSubmission()
  }

}

object TrajectoryModel extends Model {

  val predictionPath = "../trajectory-preds"
  val predictionFile = s"$predictionPath/part-00000"
  val submissionFile = s"../submission/trajectory-${DateTime.now().toString("yyyyMMddHHmmss")}.csv"
  val headerLine = "TRIP_ID,LATITUDE,LONGITUDE"

  def build(training: DataFrame, pipeline: Pipeline): (PipelineModel, PipelineModel) = {
    val lastLatTraining = training.withColumn("label", training("lastLat"))
    val lastLongTraining = training.withColumn("label", training("lastLong"))

    //    lastLatTraining.show()
    //    lastLatTraining.printSchema()

    val lastLatModel = pipeline.fit(lastLatTraining)
    val lastLongModel = pipeline.fit(lastLongTraining)

    val lastLatTrainingPredictions = lastLatModel.transform(lastLatTraining).select("features", "label", "prediction")
    val lastLongTrainingPredictions = lastLongModel.transform(lastLongTraining).select("features", "label", "prediction")

    //    lastLatTest.collect()
    //      .foreach { case Row(features: Vector, label: Double, prediction: Double) =>
    //      println(s"($features, $label) -> prediction = $prediction")
    //    }

    println("Lat training RMSE = " + new RegressionEvaluator().evaluate(lastLatTrainingPredictions))
    println("Long training RMSE = " + new RegressionEvaluator().evaluate(lastLongTrainingPredictions))

    (lastLatModel, lastLongModel)
  }

  def test(test: DataFrame, lastLatModel: PipelineModel, lastLongModel: PipelineModel): Unit = {
    val lastLatPredictions = lastLatModel.transform(test).withColumnRenamed("prediction", "LATITUDE")
    val lastLongPredictions = lastLongModel.transform(test).withColumnRenamed("prediction", "LONGITUDE")
      .withColumnRenamed("tripIdNum", "tripIdNum1")
    lastLatPredictions.join(lastLongPredictions, "TRIP_ID")
      .sort("tripIdNum")
      .select("TRIP_ID", "LATITUDE", "LONGITUDE")
      .coalesce(1)
      .write.format("com.databricks.spark.csv").mode("overwrite").save(predictionPath)

    saveSubmission()
  }
}

object IgnitionApp {

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Ignition")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val trainingRaw = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load("../train-head.csv")
      .repartition(4)

    val training = InputTransformer.transform(trainingRaw).cache()

//    val lr = new LinearRegression().setMaxIter(10)
    val rf = new RandomForestRegressor()
      .setNumTrees(100)
      .setMaxDepth(10)
      .setMaxBins(20)

    val stages = Array(
      new StringIndexer().setInputCol("CALL_TYPE").setOutputCol("callType"),
//      new StringIndexer().setInputCol("ORIGIN_CALL").setOutputCol("originCall"),
//      new StringIndexer().setInputCol("ORIGIN_STAND").setOutputCol("originStand"),
//      new StringIndexer().setInputCol("TAXI_ID").setOutputCol("taxiId"),
      new VectorAssembler()
        .setInputCols(Array("callType", "hourOfDay", "dayOfWeek", "startPos", "1mindelta", "5mindelta", "15mindelta"))
//        .setInputCols(Array("callType", "originCall", "originStand", "taxiId", "timestamp", "firstLat", "firstLong"))
        .setOutputCol("features"),
      rf
    )

    val pipeline = new Pipeline().setStages(stages)

//    val crossval = new CrossValidator()
//      .setEstimator(pipeline)
//      .setEvaluator(new RegressionEvaluator())
//
//    val paramGrid = new ParamGridBuilder()
//      .addGrid(lr.regParam, Array(0, 0.01, 0.1, 1))
//      .addGrid(lr.elasticNetParam, Array(0, 0.01, 0.1, 0.5, 0.9, 0.99, 1))
//      .build()
//    crossval.setEstimatorParamMaps(paramGrid)
//    crossval.setNumFolds(2)


    val tripLengthModel = TripLengthModel.build(training, pipeline)
    val (lastLatModel, lastLongModel) = TrajectoryModel.build(training, pipeline)

    val testRaw = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load("../test.csv")
      .repartition(4)
    val test = InputTransformer.transform(testRaw)

    TripLengthModel.test(test, tripLengthModel)
    TrajectoryModel.test(test, lastLatModel, lastLongModel)

//    println("Model is " + model.stages.last.asInstanceOf[LinearRegressionModel].weights)

    sc.stop()

    // some counts
    //    sanitized.groupBy("CALL_TYPE").agg(count("TRIP_ID")).sort().show()
    //    sanitized.groupBy("DAY_TYPE").agg(count("TRIP_ID")).show()
    //    val temp = sanitized.groupBy("TAXI_ID").agg(count("TRIP_ID").as("COUNT"))
    //    temp.sort(temp("COUNT").desc).coalesce(1).write
    //      .format("com.databricks.spark.csv")
    //      .save("../driverFreqs")


    //    val logData = sc.textFile("../train-head.csv", 2).cache()
    //    val numAs = logData.filter(line => line.contains("A")).count()
    //    val numBs = logData.filter(line => line.contains("B")).count()
    //    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
  }
}