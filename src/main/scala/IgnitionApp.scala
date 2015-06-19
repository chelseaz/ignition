import java.nio.charset.Charset
import java.nio.file.{Paths, Files}
import java.sql.Timestamp

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressor, LinearRegressionModel, LinearRegression}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Vector
import org.joda.time.DateTime
import scala.collection.JavaConversions._

import org.json4s._
import org.json4s.native.JsonMethods._

import scala.io.Source

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
  val extractFirstLat = udf[Option[Double], Seq[Seq[Double]]](_.headOption.map(_.last))
  val extractFirstLong = udf[Option[Double], Seq[Seq[Double]]](_.headOption.map(_.head))
  val extractLastLat = udf[Option[Double], Seq[Seq[Double]]](_.lastOption.map(_.last))
  val extractLastLong = udf[Option[Double], Seq[Seq[Double]]](_.lastOption.map(_.head))
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
      .withColumn("firstLat", extractFirstLat(trainingBase("polyline")))
      .withColumn("firstLong", extractFirstLong(trainingBase("polyline")))
      .withColumn("lastLat", extractLastLat(trainingBase("polyline")))
      .withColumn("lastLong", extractLastLong(trainingBase("polyline")))
      .withColumn("tripSeconds", toTripSeconds(trainingBase("polyline")))
      .select("TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID",
        "tripIdNum", "hourOfDay", "dayOfWeek", "firstLat", "firstLong", "lastLat", "lastLong", "tripSeconds")
  }
}

object IgnitionApp {

  val predictionPath = "../prediction"
  val predictionFile = s"$predictionPath/part-00000"
  val submissionFile = s"../submission/${DateTime.now().toString("yyyyMMdd HH:mm:ss")}"
  val headerLine = "TRIP_ID,TRAVEL_TIME"

  def saveSubmission() = {
    val lines = Files.readAllLines(Paths.get(predictionFile), Charset.defaultCharset())
    val newContent = (headerLine :: lines.toList).mkString("\n")
    Files.write(Paths.get(submissionFile), newContent.getBytes)
  }

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

    val lastLatTraining = training.withColumn("label", training("lastLat"))
    val lastLongTraining = training.withColumn("label", training("lastLong"))
    val tripLengthTraining = training.withColumn("label", training("tripSeconds"))

//    lastLatTraining.show()
//    lastLatTraining.printSchema()

    val lr = new LinearRegression().setMaxIter(10)
    val rf = new RandomForestRegressor()
      .setMaxDepth(10)
      .setMaxBins(20)

    val stages = Array(
      new StringIndexer().setInputCol("CALL_TYPE").setOutputCol("callType"),
//      new StringIndexer().setInputCol("ORIGIN_CALL").setOutputCol("originCall"),
//      new StringIndexer().setInputCol("ORIGIN_STAND").setOutputCol("originStand"),
//      new StringIndexer().setInputCol("TAXI_ID").setOutputCol("taxiId"),
      new VectorAssembler()
        .setInputCols(Array("callType", "hourOfDay", "dayOfWeek", "firstLat", "firstLong"))
//        .setInputCols(Array("callType", "originCall", "originStand", "taxiId", "timestamp", "firstLat", "firstLong"))
        .setOutputCol("features"),
      rf
    )

    val pipeline = new Pipeline().setStages(stages)

    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator())

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01))
      .addGrid(lr.elasticNetParam, Array(0.5))
//      .addGrid(lr.regParam, Array(0, 0.01, 0.1, 1))
//      .addGrid(lr.elasticNetParam, Array(0, 0.01, 0.1, 0.5, 0.9, 0.99, 1))
      .build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(2)

//    val lastLatModel = crossval.fit(lastLatTraining)
//    val lastLongModel = crossval.fit(lastLongTraining)
    val tripLengthModel = pipeline.fit(tripLengthTraining)

//    println("Model is " + model.stages.last.asInstanceOf[LinearRegressionModel].weights)

//    val lastLatTest = lastLatModel.transform(lastLatTraining).select("features", "label", "prediction")
//    val lastLongTest = lastLongModel.transform(lastLongTraining).select("features", "label", "prediction")
    val tripLengthTest = tripLengthModel.transform(tripLengthTraining).select("features", "label", "prediction")

//    lastLatTest.collect()
//      .foreach { case Row(features: Vector, label: Double, prediction: Double) =>
//      println(s"($features, $label) -> prediction = $prediction")
//    }

//    println("Lat training RMSE = " + new RegressionEvaluator().evaluate(lastLatTest))
//    println("Long training RMSE = " + new RegressionEvaluator().evaluate(lastLongTest))
    println("Trip length training RMSE = " + new RegressionEvaluator().evaluate(tripLengthTest))

    val testRaw = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .load("../test.csv")
      .repartition(4)


    val test = InputTransformer.transform(testRaw)
    val testPredictions = tripLengthModel.transform(test).sort("tripIdNum")

    // don't predict less than the current travel time in the test set!
    val maxColumn = udf[Double, Double, Double]((pred, curTripSeconds) => pred.max(curTripSeconds))

    testPredictions
      .withColumn("TRAVEL_TIME", maxColumn(testPredictions("prediction"), testPredictions("tripSeconds")))
      .select("TRIP_ID", "TRAVEL_TIME")
      .coalesce(1)
      .write.format("com.databricks.spark.csv").mode("overwrite").save(predictionPath)

    sc.stop()

    saveSubmission()

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