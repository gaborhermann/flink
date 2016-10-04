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
import org.apache.flink.ml.math.BLAS

object SGDtest {
  //    def main(args: Array[String]): Unit = {
  //      println("Ready")
  //    }


  def main(args: Array[String]) {
    // set up the execution environment
    val env = ExecutionEnvironment.getExecutionEnvironment

    val pathToTrainingFile = "/home/dani/data/movielens1M_data.csv"

    // Read input data set from a csv file
    val inputDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](pathToTrainingFile)

    val userIDs = inputDS.map(_._1).distinct()
    val itemIDs = inputDS.map(_._2).distinct()

    // Setup the ALS learner
    val sgd = SGD()
      .setIterations(20)
      .setNumFactors(10)
      .setBlocks(4)

    // Set the other parameters via a parameter map
    val parameters = ParameterMap()
      .add(SGD.LearningRate, 0.001)
      .add(SGD.Lambda, 0.0)
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
*/
    val testingDS = userIDs.first(20) cross itemIDs.first(20)
    testingDS.writeAsCsv("/home/dani/data/tmp/teszt001.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)

    // Calculate the ratings according to the matrix factorization
    val testingDS_100 = testingDS.first(100)
    testingDS_100.print()
    testingDS_100.writeAsCsv("/home/dani/data/tmp/teszt100_001.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)

    //val predictedRatings = sgd.predict(testingDS_100)

    //println("teszt1")
    //predictedRatings.print()

    //sgd.predict(testingDS_100).writeAsCsv("/home/dani/data/tmp/sgd_001.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)


    //DEBUGGING
    val input = testingDS_100
    sgd.factorsOption match {
      case Some((userFactors, itemFactors)) => {
        val abc = input.join(userFactors, JoinHint.REPARTITION_HASH_SECOND).where(0).equalTo(0)
          .join(itemFactors, JoinHint.REPARTITION_HASH_SECOND).where("_1._2").equalTo(i => i.id).map {
          triple => {
            val (((uID, iID), uFactors), iFactors) = triple

            val uFactorsVector = uFactors.factors
            val iFactorsVector = iFactors.factors

            val prediction = blas.ddot(
              uFactorsVector.length,
              uFactorsVector,
              1,
              iFactorsVector,
              1)

            (uID, iID, prediction)
          }

            }
        abc.writeAsCsv("/home/dani/data/tmp/output01.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
          userFactors.writeAsCsv("/home/dani/data/tmp/userF.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)

        println(userFactors.getExecutionEnvironment.getExecutionPlan())
        userFactors.getExecutionEnvironment.execute()
        println("++++++++++++++++++++++++++++++++++++++++++++")
        /*.join(itemFactors, JoinHint.REPARTITION_HASH_SECOND).where("_1._2").equalTo(i => i.id).map {
          triple => {
            val (((uID, iID), uFactors), iFactors) = triple

            val uFactorsVector = uFactors.factors
            val iFactorsVector = iFactors.factors

            val prediction = blas.ddot(
              uFactorsVector.length,
              uFactorsVector,
              1,
              iFactorsVector,
              1)

            (uID, iID, prediction)
          }

    }*/
      }
    }
  }
}
