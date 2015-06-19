name := "Ignition"

version := "1.0"

scalaVersion := "2.11.6"

libraryDependencies ++= {
  val sparkVersion = "1.4.0"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.json4s" %% "json4s-native" % "3.2.11",
    "com.databricks" %% "spark-csv" % "1.0.3",
    "joda-time" % "joda-time" % "2.8.1"
  )
}