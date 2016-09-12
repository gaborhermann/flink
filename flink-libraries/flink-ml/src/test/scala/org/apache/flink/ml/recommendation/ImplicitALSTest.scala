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

  it should "properly compute Y^T * Y" in {
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
  }

}

object ExampleMatrix {

  val factors = 3

  val Y = Array(
    Array(1.0, 3.0, 1.0),
    Array(-3.0, 4.0, 1.0),
    Array(1.0, 2.0, -1.0),
    Array(4.0, 1.0, 4.0),
    Array(3.0, -2.0, 3.0),
    Array(-1.0, 1.0, 2.0)
  )

  val expectedYtY = Array(
    Array(37.0, -10.0, 20.0),
    Array(-10.0, 35.0, 5.0),
    Array(20.0, 5.0, 32.0)
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
