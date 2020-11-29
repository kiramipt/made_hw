package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

class StandardScalerTest extends AnyFlatSpec with should.Matchers {

  val delta = 0.001
  lazy val data: DataFrame = StandardScalerTest._data
  lazy val vectors: Seq[Vector] = StandardScalerTest._vectors

  "Model" should "scale input data" in {
    val model = new StandardScalerModel(
      means = Vectors.dense(2.0, -0.5).toDense,
      stds = Vectors.dense(1.5, 0.5).toDense
    ).setInputCol("features")
     .setOutputCol("features")

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5 - 2.0) / 1.5 +- delta)
    vectors(0)(1) should be((12 + 0.5) / 0.5 +- delta)
    vectors(1)(0) should be((-1 - 2.0) / 1.5 +- delta)
    vectors(1)(1) should be((0 + 0.5) / 0.5 +- delta)
  }

  "Estimator" should "calculate means" in {
    val estimator = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    model.means(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    model.means(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
  }

  "Estimator" should "calculate stds" in {
    val estimator = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    model.stds(0) should be(Math.sqrt(vectors.map(v => (v(0) - model.means(0)) * (v(0) - model.means(0))).sum / (vectors.length - 1)) +- delta)
    model.stds(1) should be(Math.sqrt(vectors.map(v => (v(1) - model.means(1)) * (v(1) - model.means(1))).sum / (vectors.length - 1)) +- delta)
  }

  "Estimator" should "produce functional model" in {
    val estimator: StandardScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)

    vectors(0)(0) should be((13.5 - model.means(0)) / model.stds(0) +- delta)
    vectors(0)(1) should be((12 - model.means(1)) / model.stds(1) +- delta)

    vectors(1)(0) should be((-1 - model.means(0)) / model.stds(0) +- delta)
    vectors(1)(1) should be((0 - model.means(1)) / model.stds(1) +- delta)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new StandardScaler()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data).stages(0).asInstanceOf[StandardScalerModel]

    model.means(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    model.means(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
  }

}


object StandardScalerTest extends WithSpark {

  lazy val _vectors = Seq(
    Vectors.dense(13.5, 12),
    Vectors.dense(-1, 0)
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }
}