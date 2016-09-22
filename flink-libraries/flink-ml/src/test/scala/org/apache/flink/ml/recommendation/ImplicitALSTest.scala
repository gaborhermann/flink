/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.recommendation

import org.apache.flink.ml.util.FlinkTestBase
import org.scalatest._

import scala.language.postfixOps

import org.apache.flink.api.scala._

class ImplicitALSTest
  extends FlatSpec
    with Matchers
    with FlinkTestBase {

  override val parallelism = 2

  behavior of "The modification of the alternating least squares (ALS) implementation" +
    "for implicit feedback datasets."

  it should "properly compute Y^T * Y, and factorize matrix" in {
    import ExampleMatrix._

    val rand = scala.util.Random
    val numBlocks = 3
    // randomly split matrix to blocks
    val blocksY = Y
      // add a random block id to every row
      .map { row =>
        (rand.nextInt(numBlocks), row)
      }
      // get the block via grouping
      .groupBy(_._1).values
      // add a block id (-1) to each block
      .map(b => (-1, b.map(_._2)))
      .toSeq

    // use Flink to compute YtY
    val env = ExecutionEnvironment.getExecutionEnvironment

    val distribBlocksY = env.fromCollection(blocksY)

    val YtY = ALS
      .computeXtX(distribBlocksY, factors)
      .collect().head

    // check YtY size
    YtY.length should be (factors * (factors - 1) / 2 + factors)

    // check result is as expected
    expectedUpperTriangleYtY
      .zip(YtY)
      .foreach { case (expected, result) =>
        result should be (expected +- 0.1)
      }

    // factorize matrix with implicit ALS
    val als = ALS()
      .setIterations(iterations)
      .setLambda(lambda)
      .setBlocks(blocks)
      .setNumFactors(factors)
      .setImplicit(true)
      .setAlpha(alpha)
      .setSeed(seed)

    val inputDS = env.fromCollection(implicitRatings)

    als.fit(inputDS)

    // check predictions on some user-item pairs
    val testData = env.fromCollection(expectedResult.map{
      case (userID, itemID, rating) => (userID, itemID)
    })

    val predictions = als.predict(testData).collect()

    predictions.length should equal(expectedResult.length)

    val resultMap = expectedResult map {
      case (uID, iID, value) => (uID, iID) -> value
    } toMap

    predictions.foreach(println)
    predictions foreach {
      case (uID, iID, value) => {
        resultMap.isDefinedAt((uID, iID)) should be(true)

        value should be(resultMap((uID, iID)) +- 0.1)
      }
    }

  }

}

object ExampleMatrix {

  val seed = 500
  val factors = 3
  val blocks = 1
  val alpha = 40.0
  val lambda = 0.1
  val iterations = 10

  val implicitRatings = Seq(
    (0, 3, 1.0),
    (0, 6, 2.0),
    (0, 9, 1.0),
    (1, 0, 1.0),
    (1, 2, 3.0),
    (1, 6, 1.0),
    (1, 7, 5.0),
    (1, 8, 1.0),
    (2, 1, 1.0),
    (2, 4, 3.0),
    (3, 1, 2.0),
    (3, 3, 4.0),
    (3, 5, 5.0),
    (4, 5, 1.0),
    (4, 8, 2.0),
    (4, 10, 2.0),
    (5, 2, 1.0)
  )

  val expectedResult = Seq(
    (1, 1, -0.22642740122582822),
    (3, 2, -0.40638720202261835),
    (4, 3, 0.28037645952568335),
    (2, 3, 0.9176106683061931)
  )

  val Y = Array(
    Array(1.0, 3.0, 1.0),
    Array(-3.0, 4.0, 1.0),
    Array(1.0, 2.0, -1.0),
    Array(4.0, 1.0, 4.0),
    Array(3.0, -2.0, 3.0),
    Array(-1.0, 1.0, 2.0)
  )

  /**
    * Upper triangle representation by columns.
    */
  val expectedUpperTriangleYtY = Array(
    37.0,
    -10.0, 35.0,
    20.0, 5.0, 32.0
  )

}
