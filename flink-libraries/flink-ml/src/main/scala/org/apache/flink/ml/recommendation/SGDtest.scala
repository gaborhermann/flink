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

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.flink.api.common.operators.base.JoinOperatorBase.JoinHint
import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem
import org.apache.flink.ml.common.ParameterMap

object SGDtest {

  def main(args: Array[String]) {
    // set up the execution environment
    val env = ExecutionEnvironment.getExecutionEnvironment

    val pathToTrainingFile = "/home/dani/data/movielens_train.csv"

    // Read input data set from a csv file
    val inputDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](pathToTrainingFile)

    val userIDs = inputDS.map(_._1).distinct()
    val itemIDs = inputDS.map(_._2).distinct()

    // Setup the ALS learner
    val sgd = SGD()
      .setIterations(100)
      .setNumFactors(10)
      .setBlocks(4)

    // Set the other parameters via a parameter map
    val parameters = ParameterMap()
      .add(SGD.LearningRate, 0.01)
      .add(SGD.Lambda, 0.0)
      .add(SGD.Seed, 42L)

    // Calculate the factorization
    sgd.fit(inputDS, parameters)


    val pathToData = "/home/dani/data/movielens_test.csv"

    val testingDS = env.readCsvFile[(Int, Int)](pathToData)

    testingDS.writeAsCsv("/home/dani/data/tmp/teszt001.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)

    //println(testingDS.getExecutionEnvironment.getExecutionPlan())
    val predictedRatings = sgd.predict(testingDS)

    predictedRatings.writeAsCsv("/home/dani/data/tmp/sgd_001.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)

    println(predictedRatings.getExecutionEnvironment.getExecutionPlan())

    predictedRatings.getExecutionEnvironment.execute()
  }
}
