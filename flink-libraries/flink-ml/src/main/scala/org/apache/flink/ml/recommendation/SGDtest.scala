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

import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem
import org.apache.flink.ml.common.ParameterMap

object SGDtest {
  //    def main(args: Array[String]): Unit = {
  //      println("Ready")
  //    }


  def main(args: Array[String]) {
    // set up the execution environment
    val env = ExecutionEnvironment.getExecutionEnvironment

    val pathToTrainingFile = "/home/dani/data/movielens1M_data.csv"
//    val pathToTrainingFile = "/home/dani/data/selected.csv"

    // Read input data set from a csv file
    val inputDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](pathToTrainingFile)
//    inputDS.writeAsCsv(pathToTrainingFile + "_test").setParallelism(1)

    val userIDs = inputDS.map(_._1).distinct()
    val itemIDs = inputDS.map(_._2).distinct()

    // Setup the ALS learner
    val sgd = SGD()
      .setIterations(20)
      .setNumFactors(3)
    //    .setBlocks(100)
    //    .setTemporaryPath("hdfs://tempPath")

    // Set the other parameters via a parameter map
    val parameters = ParameterMap()
      .add(SGD.Lambda, 0.9)
      .add(SGD.Seed, 42L)

    // Calculate the factorization
    sgd.fit(inputDS, parameters)


    // Read the testing data set from a csv file
//    val pathToData = "/home/dani/data/full.csv"
    //    val testingDS = env.readCsvFile[(Int, Int)](pathToData)

/*
    val pathToData = "/home/dani/data/movielens1M_data_test_1.csv"

    println("-----------------------------------------")
    println(pathToData)
    val testingDS = env.readCsvFile[(Int, Int)](pathToData)

//    println("printing test dataset")
//    testingDS.print()
    // println(s"The size of the testingDS is: $(testingDS.size)")

    // val testingDS = itemIDs cross userIDs
    // testingDS.writeAsCsv("/home/dani/data/teszt2.csv").setParallelism(1)

    // Calculate the ratings according to the matrix factorization
    val testingDS_100 = testingDS.first(100)
    testingDS_100.print()

    val predictedRatings = als.predict(testingDS_100)
    predictedRatings.print()



    predictedRatings.writeAsCsv("/home/dani/data/movielens_flink_1.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
*/

    // println(predictedRatings.map(x => x.toString()))
    println("teszt2")
    // suserIDs.print()
  }
}
