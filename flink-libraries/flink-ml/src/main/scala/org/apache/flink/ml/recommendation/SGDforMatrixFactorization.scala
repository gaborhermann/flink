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
import org.apache.flink.api.common.functions.RichCoGroupFunction
import org.apache.flink.api.scala._
import org.apache.flink.ml.common._
import org.apache.flink.ml.optimization.LearningRateMethod.{Default, LearningRateMethodTrait}
import org.apache.flink.ml.pipeline.FitOperation
import org.apache.flink.util.Collector

import scala.util.Random
import scala.collection.JavaConverters._

/** Alternating least squares algorithm to calculate a matrix factorization.
  *
  *  todo describe algorithm properly
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
  * todo refine example
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
  *  - [[org.apache.flink.ml.recommendation.MatrixFactorization.NumFactors]]:
  *  The number of latent factors. It is the dimension of the calculated user and item vectors.
  *  (Default value: '''10''')
  *
  *  - [[org.apache.flink.ml.recommendation.MatrixFactorization.Lambda]]:
  *  Regularization factor. Tune this value in order to avoid overfitting/generalization.
  *  (Default value: '''1''')
  *
  *  - [[org.apache.flink.ml.regression.MultipleLinearRegression.Iterations]]:
  *  The number of iterations to perform. (Default value: '''10''')
  *
  *  - [[org.apache.flink.ml.recommendation.MatrixFactorization.Blocks]]:
  *  The number of blocks into which the user and item matrix a grouped. The fewer
  *  blocks one uses, the less data is sent redundantly. However, bigger blocks entail bigger
  *  update messages which have to be stored on the Heap. If the algorithm fails because of
  *  an OutOfMemoryException, then try to increase the number of blocks. (Default value: '''None''')
  *
  *  - [[org.apache.flink.ml.recommendation.MatrixFactorization.Seed]]:
  *  Random seed used to generate the initial item matrix for the algorithm.
  *  (Default value: '''0''')
  *
  *  todo other parameters
  */
class SGDforMatrixFactorization extends MatrixFactorization[SGDforMatrixFactorization] {

  import SGDforMatrixFactorization._

  /** Sets the learning rate for the algorithm.
    *
    * @param learningRate
    * @return
    */
  def setLearningRate(learningRate: Double): SGDforMatrixFactorization = {
    parameters.add(LearningRate, learningRate)
    this
  }

  // todo scala docs
  def setLearningRateMethod(learningRateMethod: LearningRateMethodTrait):
  SGDforMatrixFactorization = {
    parameters.add(LearningRateMethod, learningRateMethod)
    this
  }
}

object SGDforMatrixFactorization {

  import MatrixFactorization._

  // ========================================= Parameters ==========================================

  case object LearningRate extends Parameter[Double] {
    val defaultValue: Option[Double] = Some(1.0)
  }

  case object LearningRateMethod extends Parameter[LearningRateMethodTrait] {
    val defaultValue: Option[LearningRateMethodTrait] = Some(Default)
  }

  // ==================================== SGD type definitions =====================================

  /**
    * Index of a factor in a factor block.
    */
  type IndexInFactorBlock = Int

  /**
    * Representation of a rating in a rating block.
    *
    * @param rating Value of rating.
    * @param userIdx Index of user vector in the corresponding user block.
    * @param itemIdx Index of item vector in the corresponding item block.
    */
  case class RatingInfo(rating: Double,
                        userIdx: IndexInFactorBlock,
                        itemIdx: IndexInFactorBlock)

  /**
    * Rating block identifier.
    */
  type RatingBlockId = Int

  /**
    * Factor block identifier.
    */
  type FactorBlockId = Int

  /**
    * Representation of a rating block.
    *
    * @param id Identifier of the block.
    * @param block Array containing the ratings corresponding to the block.
    */
  case class RatingBlock(id: RatingBlockId, block: Array[RatingInfo])

  /**
    * Representation of a factor block.
    *
    * @param factorBlockId Identifier of the block.
    * @param currentRatingBlock Id of the rating block with which we are currently computing the
    *                           gradients
    * @param isUser Boolean marking whether it's a user or item block.
    * @param factors Array containing the vectors corresponding to the block.
    * @param omegas Array containing the omegas for every factor,
    *               i.e. the number of occurrences of that factor in the ratings.
    */
  case class FactorBlock(factorBlockId: FactorBlockId,
                         currentRatingBlock: RatingBlockId,
                         isUser: Boolean,
                         factors: Array[Array[Double]],
                         omegas: Array[Int])

  /**
    * Information for unblocking the factors at the end of the algorithm.
    *
    * @param factorBlockId Id of the factor block.
    * @param factorIds Ids of the factors in the corresponding factor block.
    */
  case class UnblockInformation(factorBlockId: FactorBlockId, factorIds: Array[Int])

  // ================================= Factory methods =============================================

  def apply(): SGDforMatrixFactorization = {
    new SGDforMatrixFactorization()
  }

  // ===================================== Operations ==============================================

  /**
    * Unblocks the factors (either user or item).
    *
    * @return [[DataSet]] containing the factors.
    */
  def unblock(factorBlocks: DataSet[FactorBlock],
              unblockInfo: DataSet[UnblockInformation],
              isUser: Boolean): DataSet[Factors] = {
    factorBlocks
      .filter(i => i.isUser == isUser)
      .join(unblockInfo).where(_.factorBlockId).equalTo(_.factorBlockId)
      .flatMap(x => x match {
        case (FactorBlock(_, _, _, factorsInBlock, _), UnblockInformation(_, ids)) =>
          ids.zip(factorsInBlock).map(x => Factors(x._1, x._2))
      })
  }

  /** Calculates the matrix factorization for the given ratings. A rating is defined as
    * a tuple of user ID, item ID and the corresponding rating.
    *
    * @return Factorization containing the user and item matrix
    */
  implicit val fitSGD = new FitOperation[SGDforMatrixFactorization, (Int, Int, Double)] {
    override def fit(
                      instance: SGDforMatrixFactorization,
                      fitParameters: ParameterMap,
                      input: DataSet[(Int, Int, Double)])
    : Unit = {
      val resultParameters = instance.parameters ++ fitParameters

      val numBlocks = resultParameters.get(Blocks).getOrElse(1)
      val seed = resultParameters(Seed)
      val factors = resultParameters(NumFactors)
      val iterations = resultParameters(Iterations)
      val lambda = resultParameters(Lambda)
      val learningRate = resultParameters(LearningRate)
      val learningRateMethod = resultParameters(LearningRateMethod)

      val ratings = input

      // Initializing user and item blocks.
      // We keep information about indices in the blocks so we can construct rating blocks
      // without having to refer to the user or item ids directly. This way we don't have to use
      // these ids during the iteration. Thus, for doing the unblocking at the end,
      // we need to store information about the user and item ids.
      val (initialUserBlocks, userIdxInBlock, userUnblockInfo) =
        initFactorBlockAndIndices(ratings.map(_._1), isUser = true, numBlocks, seed, factors)
      val (initialItemBlocks, itemIdxInBlock, itemUnblockInfo) =
        initFactorBlockAndIndices(ratings.map(_._2), isUser = false, numBlocks, seed, factors)

      // Constructing the rating blocks. There are numBlocks * numBlocks rating blocks.
      // todo maybe optimize 3-way join
      val ratingBlocks = ratings
        .join(userIdxInBlock).where(_._1).equalTo(0)
        .join(itemIdxInBlock).where(_._1._2).equalTo(0)
        // matching the indices in the factor blocks (user and item) with the ratings
        // and creating rating block ids
        .map(_ match {
          case (((user, item, rating), (_, userIdx, userBlockId)), (_, itemIdx, itemBlockId)) =>
            (toRatingBlockId(userBlockId, itemBlockId, numBlocks),
              RatingInfo(rating, userIdx, itemIdx),
              // todo eliminate this last item, only needed for deterministic result
              (user, item))
        })
        // grouping by the rating block ids and constructing rating blocks
        .groupBy(0)
        .reduceGroup {
          ratings =>
            // todo eliminate sorting, only needed for deterministic result
            val arr = ratings.toArray.sortBy(_._3)
            val ratingBlockId = arr(0)._1
            val ratingInfos = arr.map(_._2)
            RatingBlock(ratingBlockId, ratingInfos)
        }

      // We union the user and item blocks so that we can use them as one [[DataSet]]
      // in the iteration.
      val initUserItem = initialUserBlocks.union(initialItemBlocks)

      // Iteratively updating the factors. We sweep through numBlocks rating blocks
      // in one iteration, thus we sweep through the whole rating matrix in numBlocks iterations.
      val userItem = initUserItem.iterate(iterations * numBlocks) {
        ui => updateFactors(ui, ratingBlocks, learningRate, learningRateMethod,
          lambda, numBlocks, seed)
      }

      // unblocking the user and item matrices
      val users = unblock(userItem, userUnblockInfo, isUser = true)
      val items = unblock(userItem, itemUnblockInfo, isUser = false)

      instance.factorsOption = Some((users, items))
    }
  }

  /** Calculates a single sweep for the SGD optimization. The result is the new value for
    * the user and item matrix.
    *
    * @return New values for the optimized matrices.
    */
  def updateFactors(userItem: DataSet[FactorBlock],
                    ratingBlocks: DataSet[RatingBlock],
                    learningRate: Double,
                    learningRateMethod: LearningRateMethodTrait,
                    lambda: Double,
                    numBlocks: Int,
                    seed: Long): DataSet[FactorBlock] = {

    /**
      * Updates one user and item block based on one corresponding rating block.
      * This is the local logic of the SGD.
      *
      * @return The updated user and item block.
      */
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

    /**
      * Helper method for extracting the user and item block for a rating block.
      *
      * Because one of the two blocks might be missing, it returns [[Option]]s,
      * and also the rating block id, so we even if we cannot update the factors in the block,
      * we can update current rating block id in the factor blocks, preparing the next step.
      *
      * @return Optional user and item blocks and the current rating block id.
      */
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
    // Matching the user and item blocks to the current rating block.
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

  // ============================== Blocking helper functions ======================================

  /**
    * Initializes blocks for one factor matrix (either user or item).
    *
    * @param factorIdsForRatings [[DataSet]] containing the ids of the factors.
    * @param isUser Indicates whether it's a user or item matrix.
    * @param numBlocks Number of matrix blocks.
    * @param seed Random seed
    * @param factors Number of factors.
    * @return Three [[DataSet]]s: the factor blocks,
    *         the factor ids matched to their index in the factor block,
    *         and the information for unblocking.
    */
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
            (factorBlockId, Factors(id, randomFactors(factors, random)), count)
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

  // ================ Helper functions for matching rating and factor blocks =======================

  /**
    * Logic that calculates the rating block id based on the user and item block ids.
    *
    * @return Rating block id.
    */
  def toRatingBlockId(userBlockId: FactorBlockId,
                      itemBlockId: FactorBlockId,
                      numOfBlocks: Int): RatingBlockId = {
    userBlockId * numOfBlocks + itemBlockId
  }

  /**
    * Logic that creates the rating block id for the next iteration step,
    * returning the next rating block ids for the current user factor block and item factor block.
    *
    * @param currentRatingBlock Current rating block id.
    * @param numFactorBlocks Number of factor blocks.
    * @return The next rating block id for the user and item blocks.
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
