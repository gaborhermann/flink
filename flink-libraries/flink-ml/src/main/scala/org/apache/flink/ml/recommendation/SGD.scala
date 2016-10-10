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

import java.{lang, util}

import org.apache.flink.api.common.operators.base.JoinOperatorBase.JoinHint
import org.apache.flink.api.scala._
import org.apache.flink.api.common.operators.Order
import org.apache.flink.core.memory.{DataInputView, DataOutputView}
import org.apache.flink.ml.common._
import org.apache.flink.ml.pipeline.{FitOperation, PredictDataSetOperation, Predictor}
import org.apache.flink.types.Value
import org.apache.flink.util.Collector
import org.apache.flink.api.common.functions.{CoGroupFunction, GroupReduceFunction, RichMapFunction, Partitioner => FlinkPartitioner}
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.flink.core.fs.FileSystem
import org.apache.flink.ml.optimization.LearningRateMethod.{Default, LearningRateMethodTrait}
import org.apache.flink.ml.recommendation.SGD.LearningRate
import org.netlib.util.intW

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Alternating least squares algorithm to calculate a matrix factorization.
  *
  * Given a matrix `R`, SGD calculates two matricess `U` and `V` such that `R ~~ U^TV`. The
  * unknown row dimension is given by the number of latent factors. Since matrix factorization
  * is often used in the context of recommendation, we'll call the first matrix the user and the
  * second matrix the item matrix. The `i`th column of the user matrix is `u_i` and the `i`th
  * column of the item matrix is `v_i`. The matrix `R` is called the ratings matrix and
  * `(R)_{i,j} = r_{i,j}`.
  *
  * In order to find the user and item matrix, the following problem is solved:
  *
  * `argmin_{U,V} sum_(i,j\ with\ r_{i,j} != 0) (r_{i,j} - u_{i}^Tv_{j})^2 +
  * lambda (sum_(i) n_{u_i} ||u_i||^2 + sum_(j) n_{v_j} ||v_j||^2)`
  *
  * with `\lambda` being the regularization factor, `n_{u_i}` being the number of items the user `i`
  * has rated and `n_{v_j}` being the number of times the item `j` has been rated. This
  * regularization scheme to avoid overfitting is called weighted-lambda-regularization. Details
  * can be found in the work of [[http://dx.doi.org/10.1007/978-3-540-68880-8_32 Zhou et al.]].
  *
  * By fixing one of the matrices `U` or `V` one obtains a quadratic form which can be solved. The
  * solution of the modified problem is guaranteed to decrease the overall cost function. By
  * applying this step alternately to the matrices `U` and `V`, we can iteratively improve the
  * matrix factorization.
  *
  * The matrix `R` is given in its sparse representation as a tuple of `(i, j, r)` where `i` is the
  * row index, `j` is the column index and `r` is the matrix value at position `(i,j)`.
  *
  * @example
  *          {{{
  *             val inputDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](
  *               pathToTrainingFile)
  *
  *             val SGD = SGD()
  *               .setIterations(10)
  *               .setNumFactors(10)
  *
  *             SGD.fit(inputDS)
  *
  *             val data2Predict: DataSet[(Int, Int)] = env.readCsvFile[(Int, Int)](pathToData)
  *
  *             SGD.predict(data2Predict)
  *          }}}
  *
  * =Parameters=
  *
  *  - [[org.apache.flink.ml.recommendation.SGD.NumFactors]]:
  *  The number of latent factors. It is the dimension of the calculated user and item vectors.
  *  (Default value: '''10''')
  *
  *  - [[org.apache.flink.ml.recommendation.SGD.Lambda]]:
  *  Regularization factor. Tune this value in order to avoid overfitting/generalization.
  *  (Default value: '''1''')
  *
  *  - [[org.apache.flink.ml.regression.MultipleLinearRegression.Iterations]]:
  *  The number of iterations to perform. (Default value: '''10''')
  *
  *  - [[org.apache.flink.ml.recommendation.SGD.Blocks]]:
  *  The number of blocks into which the user and item matrix a grouped. The fewer
  *  blocks one uses, the less data is sent redundantly. However, bigger blocks entail bigger
  *  update messages which have to be stored on the Heap. If the algorithm fails because of
  *  an OutOfMemoryException, then try to increase the number of blocks. (Default value: '''None''')
  *
  *  - [[org.apache.flink.ml.recommendation.SGD.Seed]]:
  *  Random seed used to generate the initial item matrix for the algorithm.
  *  (Default value: '''0''')
  *
  *  - [[org.apache.flink.ml.recommendation.SGD.TemporaryPath]]:
  *  Path to a temporary directory into which intermediate results are stored. If
  *  this value is set, then the algorithm is split into two preprocessing steps, the SGD iteration
  *  and a post-processing step which calculates a last SGD half-step.
  *  The result of the individual steps are stored in the specified directory. By splitting the
  *  algorithm into multiple smaller steps, Flink does not have to split the available memory
  *  amongst too many operators. This allows the system to process bigger individual messasges and
  *  improves the overall performance. (Default value: '''None''')
  *
  * The SGD implementation is based on Spark's MLLib implementation of SGD which you can find
  * [[https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/
  * recommendation/SGD.scala here]].
  */
class SGD extends Predictor[SGD] {

  import SGD._

  // Stores the matrix factorization after the fitting phase
  var factorsOption: Option[(DataSet[Factors], DataSet[Factors])] = None

  /** Sets the number of latent factors/row dimension of the latent model
    *
    * @param numFactors
    * @return
    */
  def setNumFactors(numFactors: Int): SGD = {
    parameters.add(NumFactors, numFactors)
    this
  }

  /** Sets the regularization coefficient lambda
    *
    * @param lambda
    * @return
    */
  def setLambda(lambda: Double): SGD = {
    parameters.add(Lambda, lambda)
    this
  }

  /** Sets the number of iterations of the SGD algorithm
    *
    * @param iterations
    * @return
    */
  def setIterations(iterations: Int): SGD = {
    parameters.add(Iterations, iterations)
    this
  }

  /** Sets the number of blocks into which the user and item matrix shall be partitioned
    *
    * @param blocks
    * @return
    */
  def setBlocks(blocks: Int): SGD = {
    parameters.add(Blocks, blocks)
    this
  }

  /** Sets the random seed for the initial item matrix initialization
    *
    * @param seed
    * @return
    */
  def setSeed(seed: Long): SGD = {
    parameters.add(Seed, seed)
    this
  }

  /** Sets the temporary path into which intermediate results are written in order to increase
    * performance.
    *
    * @param temporaryPath
    * @return
    */
  def setTemporaryPath(temporaryPath: String): SGD = {
    parameters.add(TemporaryPath, temporaryPath)
    this
  }

  /** Sets the learning rate for the algorithm
    *
    * @param learningRate
    * @return
    */
  def setLearningRate(learningRate: Double): SGD = {
    parameters.add(LearningRate, learningRate)
    this
  }

  /** Empirical risk of the trained model (matrix factorization).
    *
    * @param labeledData Reference data
    * @param riskParameters Additional parameters for the empirical risk calculation
    * @return
    */
  def empiricalRisk(
                     labeledData: DataSet[(Int, Int, Double)],
                     riskParameters: ParameterMap = ParameterMap.Empty)
  : DataSet[Double] = {
    val resultingParameters = parameters ++ riskParameters

    val lambda = resultingParameters(Lambda)

    val data = labeledData map {
      x => (x._1, x._2)
    }

    factorsOption match {
      case Some((userFactors, itemFactors)) => {
        val predictions = data.join(userFactors, JoinHint.REPARTITION_HASH_SECOND).where(0)
          .equalTo(0).join(itemFactors, JoinHint.REPARTITION_HASH_SECOND).where("_1._2")
          .equalTo(0).map {
          triple => {
            val (((uID, iID), uFactors), iFactors) = triple

            val uFactorsVector = uFactors.factors
            val iFactorsVector = iFactors.factors

            val squaredUNorm2 = blas.ddot(
              uFactorsVector.length,
              uFactorsVector,
              1,
              uFactorsVector,
              1)
            val squaredINorm2 = blas.ddot(
              iFactorsVector.length,
              iFactorsVector,
              1,
              iFactorsVector,
              1)

            val prediction = blas.ddot(uFactorsVector.length, uFactorsVector, 1, iFactorsVector, 1)

            (uID, iID, prediction, squaredUNorm2, squaredINorm2)
          }
        }

        labeledData.join(predictions).where(0,1).equalTo(0,1) {
          (left, right) => {
            val (_, _, expected) = left
            val (_, _, predicted, squaredUNorm2, squaredINorm2) = right

            val residual = expected - predicted

            residual * residual + lambda * (squaredUNorm2 + squaredINorm2)
          }
        } reduce {
          _ + _
        }
      }

      case None => throw new RuntimeException("The SGD model has not been fitted to data. " +
        "Prior to predicting values, it has to be trained on data.")
    }
  }
}

object SGD {
  val USER_FACTORS_FILE = "userFactorsFile"
  val ITEM_FACTORS_FILE = "itemFactorsFile"

  // ========================================= Parameters ==========================================

  case object NumFactors extends Parameter[Int] {
    val defaultValue: Option[Int] = Some(10)
  }

  case object Lambda extends Parameter[Double] {
    val defaultValue: Option[Double] = Some(1.0)
  }

  case object Iterations extends Parameter[Int] {
    val defaultValue: Option[Int] = Some(10)
  }

  case object Blocks extends Parameter[Int] {
    val defaultValue: Option[Int] = None
  }

  case object Seed extends Parameter[Long] {
    val defaultValue: Option[Long] = Some(0L)
  }

  case object TemporaryPath extends Parameter[String] {
    val defaultValue: Option[String] = None
  }

  case object LearningRate extends Parameter[Double] {
    val defaultValue: Option[Double] = Some(1.0)
  }

  case object LearningRateMethod extends Parameter[LearningRateMethodTrait] {
    val defaultValue: Option[LearningRateMethodTrait] = Some(Default)
  }

  // ==================================== SGD type definitions =====================================

  /** Representation of a user-item rating
    *
    * @param user User ID of the rating user
    * @param item Item iD of the rated item
    * @param rating Rating value
    */
  case class Rating(user: Int, item: Int, rating: Double)

  /** Latent factor model vector
    *
    * @param id
    * @param factors
    * @param omega
    */
  case class Factors(id: Int, isUser: Boolean, factors: Array[Double], omega: Int) extends Serializable{
    override def toString = s"(id:$id,isUser:$isUser,omega:$omega,array:${factors.map(d => f"$d%1.2f").mkString(",")})"
  }

  case class Factorization(userFactors: DataSet[Factors], itemFactors: DataSet[Factors])

  // ================================= Factory methods =============================================

  def apply(): SGD = {
    new SGD()
  }

  // ===================================== Operations ==============================================

  /** Predict operation which calculates the matrix entry for the given indices  */
  implicit val predictRating = new PredictDataSetOperation[SGD, (Int, Int), (Int ,Int, Double)] {
    override def predictDataSet(
                                 instance: SGD,
                                 predictParameters: ParameterMap,
                                 input: DataSet[(Int, Int)])
    : DataSet[(Int, Int, Double)] = {

      instance.factorsOption match {
        case Some((userFactors, itemFactors)) => {
          input.join(userFactors, JoinHint.REPARTITION_HASH_SECOND).where(0).equalTo(i => i.id)
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
        }

        case None => throw new RuntimeException("The SGD model has not been fitted to data. " +
          "Prior to predicting values, it has to be trained on data.")
      }
    }
  }

  /** Calculates the matrix factorization for the given ratings. A rating is defined as
    * a tuple of user ID, item ID and the corresponding rating.
    *
    * @return Factorization containing the user and item matrix
    */
  implicit val fitSGD =  new FitOperation[SGD, (Int, Int, Double)] {
    override def fit(
                      instance: SGD,
                      fitParameters: ParameterMap,
                      input: DataSet[(Int, Int, Double)])
    : Unit = {
      val resultParameters = instance.parameters ++ fitParameters

      val numBlocks = resultParameters.get(Blocks).getOrElse(1)
      val persistencePath = resultParameters.get(TemporaryPath)
      val seed = resultParameters(Seed)
      val factors = resultParameters(NumFactors)
      val iterations = resultParameters(Iterations)
      val lambda = resultParameters(Lambda)
      val learningRate = resultParameters(LearningRate)
      val learningRateMethod = resultParameters(LearningRateMethod)

      val ratings = input.map {
        entry => {
          val (userID, itemID, rating) = entry
          Rating(userID, itemID, rating)
        }
      }

      val userIDs = ratings.map(_.user).distinct()
      val itemIDs = ratings.map(_.item).distinct()

      val userGroups = userIDs.map(id => (id, Random.nextInt(numBlocks)))

      val itemGroups = itemIDs.map(id => (id, Random.nextInt(numBlocks)))

      val userCount = ratings.map {rating => (rating.user, 1)}.groupBy(0).sum(1)
      val itemCount = ratings.map {rating => (rating.item, 1)}.groupBy(0).sum(1)

      val ratingsGrouped = ratings.join(userGroups).where(_.user).equalTo(0).join(itemGroups).where(_._1.item).equalTo(0)
        .map(i => (i._1._1, i._1._2._2 * numBlocks + i._2._2))
        .groupBy(1).reduceGroup {
          ratings => {
            val seq = ratings.toSeq
            val rating = seq.map(elem => elem._1)
            val group = seq(0)._2

            (group, rating)
          }
        }

      val initialUsers = userGroups
        .join(userCount).where(0).equalTo(0)
        .map(row => (row._1._2, new Factors(row._1._1, true,  Array.fill(factors)(Random.nextDouble()), row._2._2)))
        .groupBy(0).reduceGroup {
        users => {
          val seq = users.toSeq
          val factors = seq.map(elem => elem._2)
          val group = seq(0)._1 * (numBlocks + 1)

          (group, factors)
        }
      }

      val initialItems = itemGroups
        .join(itemCount).where(0).equalTo(0)
        .map(row => (row._1._2, new Factors(row._1._1, false, Array.fill(factors)(Random.nextDouble()), row._2._2)))
        .groupBy(0).reduceGroup {
        items => {
          val seq = items.toSeq
          val factors = seq.map(elem => elem._2)
          val group = seq(0)._1  * (numBlocks + 1)

          (group, factors)
        }
      }

      val initUserItem = initialUsers.union(initialItems)
//      initUserItem.writeAsCsv("/home/dani/data/tmp/initUser.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)

      var iterCount = 0

      val userItem = initUserItem.iterate(iterations * numBlocks) {
        ui => updateFactors(ui, ratingsGrouped, learningRate, learningRateMethod, lambda, numBlocks)
      }

      userItem.writeAsCsv("/home/dani/data/tmp/useritem_final.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
      userItem.getExecutionEnvironment.execute()

      val users = userItem.filter(i => i._2(0).isUser).flatMap((group, col: Collector[SGD.Factors]) => {
        group._2.foreach(col.collect(_))
      })
      val items = userItem.filter(i => !i._2(0).isUser).flatMap((group, col: Collector[SGD.Factors]) => {
        group._2.foreach(col.collect(_))
      })

      users.writeAsCsv("/home/dani/data/tmp/users_final.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
      items.writeAsCsv("/home/dani/data/tmp/items_final.csv", writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
      instance.factorsOption = Some((users, items))
    }
  }

  /** Calculates a single sweep for the SGD optimization. The result is the new value for
    * the user and item matrix.
    *
    * @param userItem Fixed matrix value for the half step
    * @return New value for the optimized matrix (either user or item)
    */
  def updateFactors(userItem: DataSet[(Int, Seq[SGD.Factors])],
                    groupedRatings: DataSet[(Int, Seq[SGD.Rating])],
                    learningRate: Double,
                    learningRateMethod: LearningRateMethodTrait,
                    lambda: Double,
                    numBlocks: Int): DataSet[(Int, Seq[SGD.Factors])] = {

    val users = userItem.filter(i => i._2(0).isUser)
    val items = userItem.filter(i => !i._2(0).isUser)

    val grouped = users.join(items, JoinHint.REPARTITION_SORT_MERGE).where(0).equalTo(0) {
      (user, item, out: Collector[(Int, Seq[SGD.Factors], Seq[SGD.Factors])]) =>
        out.collect(user._1, user._2, item._2)
    }.join(groupedRatings, JoinHint.REPARTITION_SORT_MERGE).where(0).equalTo(0) {
      (userItem, rating, out: Collector[(Int, Seq[SGD.Factors], Seq[SGD.Factors], Seq[SGD.Rating])]) =>
        out.collect(userItem._1, userItem._2, userItem._3, rating._2)
    }

    val pr = grouped.map(new RichMapFunction[(Int, Seq[SGD.Factors], Seq[SGD.Factors], Seq[SGD.Rating]), ((Int, Seq[SGD.Factors]), (Int, Seq[SGD.Factors]))] {
        override def map(row: (Int, Seq[SGD.Factors], Seq[SGD.Factors], Seq[SGD.Rating])):
        ((Int, Seq[SGD.Factors]), (Int, Seq[SGD.Factors])) = {
          val iteration = getIterationRuntimeContext.getSuperstepNumber / numBlocks

          val effectiveLearningRate = learningRateMethod.calculateLearningRate(
            learningRate,
            iteration + 1,
            lambda)

          println("++++++++EFFECTIVE LR:" + effectiveLearningRate.toString)
          val group = row._1
          val users = row._2
          val items = row._3
          val ratings = row._4

          val userMap = collection.mutable.Map(users.map(factor => (factor.id, (factor.factors, factor.omega))).toSeq: _*)
          val itemMap = collection.mutable.Map(items.map(factor => (factor.id, (factor.factors, factor.omega))).toSeq: _*)

          Random.shuffle(ratings) foreach {
            rating => {
              val (pi, omegai) = userMap(rating.user)
              val (qj, omegaj) = itemMap(rating.item)

              val rij = rating.rating

              val piqj = rij - blas.ddot(pi.length, pi, 1, qj, 1)

              val newPi = pi.zip(qj).map { case (p, q) => p - effectiveLearningRate * (lambda / omegai * p - piqj * q) }
              val newQj = pi.zip(qj).map { case (p, q) => q - effectiveLearningRate * (lambda / omegaj * q - piqj * p) }

              userMap.update(rating.user, (newPi, omegai))
              itemMap.update(rating.item, (newQj, omegaj))
            }
          }

          val userResult = userMap.map{case (id, (fact, omega))  => new Factors(id, true, fact, omega)}.toSeq
          val itemResult = itemMap.map{case (id, (fact, omega))  => new Factors(id, false, fact, omega)}.toSeq

          // Calculating the new group ids
          val pRow = group / numBlocks
          val qRow = group % numBlocks

          val newP = pRow * numBlocks + (group + 1) % numBlocks
          val newQ = ((pRow + numBlocks - 1) % numBlocks) * numBlocks + qRow
          ((newP, userResult), (newQ, itemResult))
        }
      }
    ).setParallelism(numBlocks)

    pr.flatMap{(a, col: Collector[(Int, Seq[SGD.Factors])]) => {
      col.collect(a._1)
      col.collect(a._2)
    }}
  }

  // ================================ Math helper functions ========================================
  def randomFactors(factors: Int, random: Random): Array[Double] = {
    Array.fill(factors)(random.nextDouble())
  }
}
