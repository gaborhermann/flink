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

import java.lang.Iterable

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.flink.api.common.functions.{RichCoGroupFunction, RichMapFunction}
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
import scala.collection.JavaConverters._

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
                        itemIdx: IndexInFactorBlock)

  /** Latent factor model vector
    *
    * @param id
    * @param factors
    */
  case class Factor(id: Int, factors: Array[Double])
    extends Serializable

  type RatingBlockId = Int
  case class RatingBlock(id: RatingBlockId, block: Array[RatingInfo])

  type FactorBlockId = Int
  // todo sealed trait + user,item class
  case class FactorBlock(factorBlockId: FactorBlockId,
                         currentRatingBlock: RatingBlockId,
                         isUser: Boolean,
                         factors: Array[Array[Double]],
                         omegas: Array[Int]) {
    override def toString: String = {
      s"${if (isUser) "user" else "item" } #$factorBlockId -> ${factors.toSeq}"
    }
  }

  case class UnblockInformation(factorBlockId: FactorBlockId, factorIds: Array[Int])

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

  def initFactorBlockAndIndices(factorIdsForRatings: DataSet[Int],
                                isUser: Boolean,
                                numBlocks: Int,
                                seed: Long,
                                factors: Int):
  (DataSet[FactorBlock], DataSet[(Int, Int, FactorBlockId)], DataSet[UnblockInformation]) = {

    val factorIDs = factorIdsForRatings.distinct()

    val factorBlockIds: DataSet[(Int, FactorBlockId)] = factorIDs.map(id =>
      (id, new Random(id ^ seed).nextInt(numBlocks))
    )

    val factorCounts = factorIdsForRatings.map((_, 1)).groupBy(0).sum(1)

    val initialFactorBlocks =
      factorBlockIds
        .join(factorCounts).where(0).equalTo(0)
        .map(_ match {
          case ((id, factorBlockId), (_, count)) =>
            val random = new Random(id ^ seed)
            (factorBlockId, Factor(id, ALS.randomFactors(factors, random)), count)
        })
        .groupBy(0).reduceGroup {
        users => {
          val arr = users.toArray
          val factors = arr.map(_._2)
          val omegas = arr.map(_._3)
          val factorBlockId = arr(0)._1
          val initialRatingBlock = factorBlockId * (numBlocks + 1)

          val factorIds = factors.map(_.id)

          (FactorBlock(factorBlockId, initialRatingBlock, isUser, factors.map(_.factors), omegas),
            factorIds)
        }
      }

    val factorIdxInBlock = initialFactorBlocks
      .map(x => (x._1.factorBlockId, x._2))
      .flatMap {
        _ match {
          case (factorBlockId, ids) =>
            ids.zipWithIndex.map {
              case (id, idx) =>
                (id, idx, factorBlockId)
            }
        }
      }

    val unblockInfo = initialFactorBlocks
      .map(x => UnblockInformation(x._1.factorBlockId, x._2))

    (initialFactorBlocks.map(_._1), factorIdxInBlock, unblockInfo)
  }

  def unblock(factorBlocks: DataSet[FactorBlock],
              unblockInfo: DataSet[UnblockInformation],
              isUser: Boolean): DataSet[Factor] = {
    factorBlocks
      .filter(i => i.isUser == isUser)
      .join(unblockInfo).where(_.factorBlockId).equalTo(_.factorBlockId)
      .flatMap(x => x match {
        case (FactorBlock(_, _, _, factorsInBlock, _), UnblockInformation(_, ids)) =>
          ids.zip(factorsInBlock).map(x => Factor(x._1, x._2))
      })
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

      val (initialUserBlocks, userIdxInBlock, userUnblockInfo) =
        initFactorBlockAndIndices(ratings.map(_._1), isUser = true, numBlocks, seed, factors)
      val (initialItemBlocks, itemIdxInBlock, itemUnblockInfo) =
        initFactorBlockAndIndices(ratings.map(_._2), isUser = false, numBlocks, seed, factors)

      // todo maybe optimize 3-way join
      val ratingBlocks = ratings
        .join(userIdxInBlock).where(_._1).equalTo(0)
        .join(itemIdxInBlock).where(_._1._2).equalTo(0)
        .map(_ match {
          case (((user, item, rating), (_, userIdx, userBlockId)), (_, itemIdx, itemBlockId)) =>
            (toRatingBlockId(userBlockId, itemBlockId, numBlocks),
              RatingInfo(rating, userIdx, itemIdx),
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

      val initUserItem = initialUserBlocks.union(initialItemBlocks)

      val userItem = initUserItem.iterate(iterations * numBlocks) {
        ui => updateFactors(ui, ratingBlocks, learningRate, learningRateMethod,
          lambda, numBlocks, seed)
      }

      // fixme: if commented it runs significantly slower: 2.5 sec vs 3.7 sec for 1000 iter on test
      // TODO: REMOVE FROM FINAL VERSION
//      userItem.writeAsCsv("/home/ghermann/tmp/useritem_final.csv",
//        writeMode =FileSystem.WriteMode.OVERWRITE).setParallelism(1)
//      userItem.getExecutionEnvironment.execute()
      // END OF TODO

      val users = unblock(userItem, userUnblockInfo, isUser = true)
      val items = unblock(userItem, itemUnblockInfo, isUser = false)

      // TODO: REMOVE FROM FINAL VERSION
//      users.writeAsCsv("/home/ghermann/tmp/users_final.csv",
//        writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
//      items.writeAsCsv("/home/ghermann/tmp/items_final.csv",
//        writeMode = FileSystem.WriteMode.OVERWRITE).setParallelism(1)
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

    def updateLocalFactors(ratingBlock: RatingBlock,
                           userBlock: FactorBlock,
                           itemBlock: FactorBlock,
                           iteration: Int): (FactorBlock, FactorBlock) = {

      val effectiveLearningRate = learningRateMethod.calculateLearningRate(
        learningRate,
        iteration + 1,
        lambda)

      val RatingBlock(ratingBlockId, ratings) = ratingBlock
      val FactorBlock(userBlockId, _, _, users, userOmegas) = userBlock
      val FactorBlock(itemBlockId, _, _, items, itemOmegas) = itemBlock

      val random = new Random(iteration ^ ratingBlockId ^ seed)
      val shuffleIdxs = random.shuffle[Int, IndexedSeq](ratings.indices)

      shuffleIdxs.map(ratingIdx => ratings(ratingIdx)) foreach {
        rating => {
          val pi = users(rating.userIdx)
          val omegai = userOmegas(rating.userIdx)

          val qj = items(rating.itemIdx)
          val omegaj = itemOmegas(rating.itemIdx)

          val rij = rating.rating

          val piqj = rij - blas.ddot(pi.length, pi, 1, qj, 1)

          val newPi = pi.zip(qj).map { case (p, q) =>
            p - effectiveLearningRate * (lambda / omegai * p - piqj * q) }
          val newQj = pi.zip(qj).map { case (p, q) =>
            q - effectiveLearningRate * (lambda / omegaj * q - piqj * p) }

          users(rating.userIdx) = newPi
          items(rating.itemIdx) = newQj
        }
      }

      (userBlock.copy(factors = users), itemBlock.copy(factors = items))
    }

    def extractUserItemBlock(factorBlocks: Iterator[FactorBlock]):
      (Option[FactorBlock], Option[FactorBlock], RatingBlockId) = {
      val b1 = factorBlocks.next()
      val b2 = factorBlocks.toIterable.headOption

      if (b1.isUser) {
        (Some(b1), b2, b1.currentRatingBlock)
      } else {
        (b2, Some(b1), b1.currentRatingBlock)
      }
    }

    // todo Consider left outer join.
    //   - pros
    //      . it would eliminate rating block check (no need to "invoke" unused rating block)
    //      . flexible underlying implementation (shipping/local strategy)
    //   - cons
    //      . has to aggregate the two factor blocks in the join function
    userItem.coGroup(ratingBlocks)
      .where(factorBlock => factorBlock.currentRatingBlock)
      .equalTo(ratingBlock => ratingBlock.id).apply(
      new RichCoGroupFunction[FactorBlock, RatingBlock, FactorBlock] {

        override def coGroup(factorBlocksJava: Iterable[FactorBlock],
                             ratingBlocksJava: Iterable[RatingBlock],
                             out: Collector[FactorBlock]): Unit = {

          val factorBlocks = factorBlocksJava.asScala.iterator
          val ratingBlocks = ratingBlocksJava.asScala.iterator

          // We only need to update factors by the current rating block
          // if there are factors matched to that rating block.
          if (factorBlocks.hasNext) {
            // There are factors matched to the current rating block,
            // so we are updating those factors by the current rating block.

            // There are two factor blocks, one user and one item.
            // One of them might be missing when there is no rating block,
            // so we use options.
            val (userBlock, itemBlock, currentRatingBlock) = extractUserItemBlock(factorBlocks)

            val (updatedUserBlock, updatedItemBlock) =
              if (ratingBlocks.hasNext) {
                // there is one rating block
                val ratingBlock = ratingBlocks.next()

                val currentIteration = getIterationRuntimeContext.getSuperstepNumber / numBlocks

                val (userB, itemB) =
                  updateLocalFactors(ratingBlock, userBlock.get, itemBlock.get, currentIteration)

                (Some(userB), Some(itemB))
              } else {
                // There are no ratings in the current block, so we do not update the factors,
                // just pass them forward.

                (userBlock, itemBlock)
              }

            // calculating the next rating block for the factor blocks
            val (newP, newQ) = nextRatingBlock(currentRatingBlock, numBlocks)

            updatedUserBlock.foreach(x => out.collect(x.copy(currentRatingBlock = newP)))
            updatedItemBlock.foreach(x => out.collect(x.copy(currentRatingBlock = newQ)))
          }
        }
      })
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
  def nextRatingBlock(currentRatingBlock: RatingBlockId,
                      numFactorBlocks: Int): (RatingBlockId, RatingBlockId) = {
    val pRow = currentRatingBlock / numFactorBlocks
    val qRow = currentRatingBlock % numFactorBlocks

    val newP = pRow * numFactorBlocks + (currentRatingBlock + 1) % numFactorBlocks
    val newQ = ((pRow + numFactorBlocks - 1) % numFactorBlocks) * numFactorBlocks + qRow
    (newP, newQ)
  }
}
