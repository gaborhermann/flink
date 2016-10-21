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
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.operators.base.JoinOperatorBase.JoinHint
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.core.fs.FileSystem
import org.apache.flink.ml.common._
import org.apache.flink.ml.optimization.LearningRateMethod.{Default, LearningRateMethodTrait}
import org.apache.flink.ml.pipeline.{FitOperation, PredictDataSetOperation, Predictor}
import org.apache.flink.util.Collector

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Alternating least squares algorithm to calculate a matrix factorization.
  *
  * Given a matrix `R`, SGD calculates two matrices `U` and `V` such that `R ~~ U^TV`. The
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
  */
class SGD extends Predictor[SGD] {

  import SGD._

  // Stores the matrix factorization after the fitting phase
  var factorsOption: Option[(DataSet[Factor], DataSet[Factor])] = None

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

        labeledData.join(predictions).where(0, 1).equalTo(0, 1) {
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

  type IndexInFactorBlock = Int

  /** Representation of a user-item rating
    *
    * @param user User ID of the rating user
    * @param item Item iD of the rated item
    * @param rating Rating value
    */
  case class RatingInfo(rating: Double,
                        userIdx: IndexInFactorBlock,
                        itemIdx: IndexInFactorBlock,
                       uiId: (Int, Int))

  /** Latent factor model vector
    *
    * @param id
    * @param factors
    * @param omega
    */
  case class Factor(id: Int, factors: Array[Double], omega: Int)
    extends Serializable

  type RatingBlockId = Int
  case class RatingBlock(id: RatingBlockId, block: Array[RatingInfo])

  type FactorBlockId = Int
  // todo sealed trait + user,item class
  case class FactorBlock(factorBlockId: FactorBlockId,
                         currentRatingBlock: RatingBlockId,
                         isUser: Boolean,
                         factors: Array[Factor]) {
    override def toString: String = {
      s"${if (isUser) "user" else "item" } #$factorBlockId -> ${factors.toSeq}"
    }
  }

  def toRatingBlockId(userBlockId: FactorBlockId,
                      itemBlockId: FactorBlockId,
                      numOfBlocks: Int): RatingBlockId = {
    userBlockId * numOfBlocks + itemBlockId
  }

  case class Factorization(userFactors: DataSet[Factor], itemFactors: DataSet[Factor])

  // ================================= Factory methods =============================================

  def apply(): SGD = {
    new SGD()
  }

  // ===================================== Operations ==============================================

  /** Predict operation which calculates the matrix entry for the given indices  */
  implicit val predictRating = new PredictDataSetOperation[SGD, (Int, Int), (Int, Int, Double)] {
    override def predictDataSet(
                                 instance: SGD,
                                 predictParameters: ParameterMap,
                                 input: DataSet[(Int, Int)])
    : DataSet[(Int, Int, Double)] = {

      instance.factorsOption match {
        case Some((userFactors, itemFactors)) => {
          input.join(userFactors, JoinHint.REPARTITION_HASH_SECOND).where(0).equalTo(i => i.id)
            .join(itemFactors, JoinHint.REPARTITION_HASH_SECOND).where("_1._2").equalTo(i => i.id)
            .map {
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
  implicit val fitSGD = new FitOperation[SGD, (Int, Int, Double)] {
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

      val ratings = input

      val userIDs = ratings.map(_._1).distinct()
      val itemIDs = ratings.map(_._2).distinct()

      // TODO: maybe use better random init
      val userBlockIds: DataSet[(Int, FactorBlockId)] = userIDs.map(id =>
        (id, new Random(id ^ seed).nextInt(numBlocks))
      )
      val itemBlockIds: DataSet[(Int, FactorBlockId)] = itemIDs.map(id =>
        (id, new Random(id ^ seed).nextInt(numBlocks))
      )

      def initialFactorBlocks(factorBlockIds: DataSet[(Int, FactorBlockId)],
                              factorCounts: DataSet[(Int, Int)],
                              isUser: Boolean) = {

        factorBlockIds
          .join(factorCounts).where(0).equalTo(0)
          .map(_ match {
            case ((id, factorBlockId), (_, count)) =>
              val random = new Random(id ^ seed)
              (factorBlockId, Factor(id, ALS.randomFactors(factors, random), count))
          })
          .groupBy(0).reduceGroup {
          users => {
            val arr = users.toArray
            val factors = arr.map(elem => elem._2)
            val factorBlockId = arr(0)._1
            val initialRatingBlock = factorBlockId * (numBlocks + 1)

            FactorBlock(factorBlockId, initialRatingBlock, isUser, factors)
          }
        }
      }

      val userCounts = ratings.map { rating => (rating._1, 1) }.groupBy(0).sum(1)
      val itemCounts = ratings.map { rating => (rating._2, 1) }.groupBy(0).sum(1)

      val initialUserBlocks = initialFactorBlocks(userBlockIds, userCounts, isUser = true)
      val initialItemBlocks = initialFactorBlocks(itemBlockIds, itemCounts, isUser = false)

      def factorIdxInBlock(blocks: DataSet[FactorBlock]) = {
        blocks.flatMap { _ match {
            case FactorBlock(factorBlockId, _, _, fs) =>
              fs.zipWithIndex.map {
                case (Factor(id, _, _), idx) =>
                  (id, idx, factorBlockId)
              }
          }
        }
      }

      val userIdxInBlock = factorIdxInBlock(initialUserBlocks)
      val itemIdxInBlock = factorIdxInBlock(initialItemBlocks)

      val ratingBlocks = ratings
        .join(userIdxInBlock).where(_._1).equalTo(0)
        .join(itemIdxInBlock).where(_._1._2).equalTo(0)
        .map(_ match {
          case (((user, item, rating), (_, userIdx, userBlockId)), (_, itemIdx, itemBlockId)) =>
            (toRatingBlockId(userBlockId, itemBlockId, numBlocks),
              RatingInfo(rating, userIdx, itemIdx, (user, item)),
              // todo eliminate this last item, only needed for deterministic result
              (user, item))
        })
        .groupBy(0)
        .reduceGroup {
          ratings =>
            // todo eliminate sorting, only needed for deterministic result
            val arr = ratings.toArray.sortBy(_._3)
            val ratingBlockId = arr(0)._1
            val ratingInfos = arr.map(_._2)
            RatingBlock(ratingBlockId, ratingInfos)
        }

      ratingBlocks.collect.foreach(println)
      // todo maybe optimize 3-way join
//      val ratingsBlocks = ratings
//        .join(userBlockIds).where(_._1).equalTo(0)
//        .join(itemBlockIds).where(_._1._2).equalTo(0)
//        .map(_ match {
//          case ((rating, (_, userBlock)), (_, itemBlock)) =>
//            (rating, toRatingBlockId(userBlock, itemBlock, numBlocks))
//        })
//        .groupBy(1).reduceGroup {
//        ratings => {
//          val seq = ratings.toSeq
//          val rating = seq.map(elem => elem._1)
//          val ratingBlockId = seq.head._2
//
//          (ratingBlockId, rating)
//        }
//      }

      val initUserItem = initialUserBlocks.union(initialItemBlocks)

      initUserItem.collect.foreach(println)
      println("_-----------------_")

      val userItem = initUserItem.iterate(iterations * numBlocks) {
        ui => updateFactors(ui, ratingBlocks, learningRate, learningRateMethod,
          lambda, numBlocks, seed)
      }

      // TODO: REMOVE FROM FINAL VERSION
      userItem.writeAsCsv("/home/ghermann/tmp/useritem_final.csv",
        writeMode =FileSystem.WriteMode.OVERWRITE).setParallelism(1)
      userItem.getExecutionEnvironment.execute()
      // END OF TODO

      userItem.collect.foreach(println)

      val users = userItem.filter(i => i.isUser)
        .flatMap((group, col: Collector[SGD.Factor]) => {
          group.factors.foreach(col.collect)
        })
      val items = userItem.filter(i => !i.isUser)
        .flatMap((group, col: Collector[SGD.Factor]) => {
          group.factors.foreach(col.collect)
        })

      // TODO: REMOVE FROM FINAL VERSION
      users.writeAsCsv("/home/ghermann/tmp/users_final.csv",
        writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
      items.writeAsCsv("/home/ghermann/tmp/items_final.csv",
        writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
      // END OF TODO

      instance.factorsOption = Some((users, items))
    }
  }

  /** Calculates a single sweep for the SGD optimization. The result is the new value for
    * the user and item matrix.
    *
    * @param userItem Fixed matrix value for the half step
    * @return New value for the optimized matrix (either user or item)
    */
  def updateFactors(userItem: DataSet[FactorBlock],
                    ratingBlocks: DataSet[RatingBlock],
                    learningRate: Double,
                    learningRateMethod: LearningRateMethodTrait,
                    lambda: Double,
                    numBlocks: Int,
                    seed: Long): DataSet[FactorBlock] = {

//    val users = userItem.filter(i => i._2(0).isUser)
//    val items = userItem.filter(i => !i._2(0).isUser)

    def extractUserItemBlock(factorBlocks: Iterator[FactorBlock]): (FactorBlock, FactorBlock) = {
      val b1 = factorBlocks.next()
      val b2 = factorBlocks.next()

      if (b1.isUser) {
        (b1, b2)
      } else {
        (b2, b1)
      }
    }

    val grouped =
      // todo coGroup does an outer join, we need an inner join, eliminate coGroup
      userItem.coGroup(ratingBlocks)
        .where(factorBlock => factorBlock.currentRatingBlock)
        .equalTo(ratingBlock => ratingBlock.id).apply {
        (factorBlocks: Iterator[FactorBlock], ratingBlock: Iterator[RatingBlock],
         out: Collector[(RatingBlock, FactorBlock, FactorBlock)]) =>
          if (factorBlocks.hasNext && ratingBlock.hasNext) {
            // There are factors matched to the current rating block,
            // so we are updating those factors by the current rating block.
            // We could eliminate this check with an inner join.

            // there are two factor blocks, one user and one item
            val (userBlock, itemBlock) = extractUserItemBlock(factorBlocks)
            // there is one rating block
            out.collect((ratingBlock.next(), userBlock, itemBlock))
          } else {
            // TODO cleanup
//            println(s"No factor blocks (${!factorBlocks.hasNext})" +
//              s"or no rating block (${!ratingBlock.hasNext})")
          }
      }

    val pr = grouped.map(
      new RichMapFunction[(SGD.RatingBlock, FactorBlock, FactorBlock),
        (FactorBlock, FactorBlock)] {

      @transient
      var random: Random = _

      override def open(parameters: Configuration): Unit = {
        // todo use optional seed
        random = new Random(seed)
      }

      override def map(row: (RatingBlock, FactorBlock, FactorBlock)):
      (FactorBlock, FactorBlock) = {
        val iteration = getIterationRuntimeContext.getSuperstepNumber / numBlocks

        val effectiveLearningRate = learningRateMethod.calculateLearningRate(
          learningRate,
          iteration + 1,
          lambda)

        val ratingBlock = row._1

        val group = ratingBlock.id
        val users = row._2.factors
        val items = row._3.factors
        val ratings = ratingBlock.block

        val userBlockId = row._2.factorBlockId
        val itemBlockId = row._3.factorBlockId
//
//        val userMap = collection.mutable.Map(users.map(factor =>
//          (factor.id, (factor.factors, factor.omega))).toSeq: _*)
//        val itemMap = collection.mutable.Map(items.map(factor =>
//          (factor.id, (factor.factors, factor.omega))).toSeq: _*)

        // todo shuffle ratings deterministically
        //          random.shuffle(ratings) foreach {
//        val ratings2 = ratings.sortBy(r => (r.user, r.item))
        ratings foreach {
          rating => {
            val Factor(pId, pi, omegai) = users(rating.userIdx)
            val Factor(qId, qj, omegaj) = items(rating.itemIdx)

            val rij = rating.rating

            val piqj = rij - blas.ddot(pi.length, pi, 1, qj, 1)

            val newPi = pi.zip(qj).map { case (p, q) =>
              p - effectiveLearningRate * (lambda / omegai * p - piqj * q) }
            val newQj = pi.zip(qj).map { case (p, q) =>
              q - effectiveLearningRate * (lambda / omegaj * q - piqj * p) }

            users(rating.userIdx) = Factor(pId, newPi, omegai)
            items(rating.itemIdx) = Factor(qId, newQj, omegaj)
//            userMap.update(rating.user, (newPi, omegai))
//            itemMap.update(rating.item, (newQj, omegaj))
          }
        }

//        val userResult = userMap.map { case (id, (fact, omega)) =>
//          Factor(id, fact, omega) }.toSeq
//        val itemResult = itemMap.map { case (id, (fact, omega)) =>
//          Factor(id, fact, omega) }.toSeq

        val (newP, newQ) = nextGroup(group, numBlocks)
        (FactorBlock(userBlockId, newP, isUser = true, users),
          FactorBlock(itemBlockId, newQ, isUser = false, items))
      }
    }
    ).setParallelism(numBlocks)

    pr.flatMap { (a, col: Collector[FactorBlock]) => {
      col.collect(a._1)
      col.collect(a._2)
    }
    }
  }

  // ================================ Math helper functions ========================================

  /**
    * Logic that creates the rating block id for the next iteration step,
    * returning the next rating block ids for the current user factor block and item factor block.
    *
    * @param currentRatingBlock
    * @param numFactorBlocks
    * @return
    */
  def nextGroup(currentRatingBlock: RatingBlockId,
                numFactorBlocks: Int): (RatingBlockId, RatingBlockId) = {
    val pRow = currentRatingBlock / numFactorBlocks
    val qRow = currentRatingBlock % numFactorBlocks

    val newP = pRow * numFactorBlocks + (currentRatingBlock + 1) % numFactorBlocks
    val newQ = ((pRow + numFactorBlocks - 1) % numFactorBlocks) * numFactorBlocks + qRow
    (newP, newQ)
  }
}
