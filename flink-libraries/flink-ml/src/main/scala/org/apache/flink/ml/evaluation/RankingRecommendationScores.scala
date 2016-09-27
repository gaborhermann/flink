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
package org.apache.flink.ml.evaluation

import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.scala.DataSet
import org.apache.flink.ml.pipeline.Predictor
import org.apache.flink.ml.recommendation.ALS
import org.apache.flink.util.Collector
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._

import scala.collection.mutable.SortedSet
import org.apache.flink.ml._
import java.io.Serializable
import java.lang.Iterable

import org.apache.flink.api.common.functions.{CombineFunction, GroupCombineFunction, RichGroupReduceFunction}

import scala.collection.mutable
import collection.JavaConverters.mapAsScalaMapConverter

//import org.apache.flink.ml.RichNumericDataSet

import org.apache.flink.api.common.operators.Order
import org.apache.flink.ml.pipeline.Predictor
import org.apache.flink.ml.recommendation.ALS
import org.apache.flink.util.Collector

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.math.log

/**
  * Created by kd on 9/27/16.
  */

class RankingRecommendationScores(val topK : Int) {
  def predictScores(als: Predictor[ALS], userItemPairs: DataSet[(Int,Int)]): DataSet[(Int,Int,Double)] = {
    als.predict(userItemPairs)
  }

  def getUserItemPairs(users : DataSet[Int], items : DataSet[Int], exclude : DataSet[(Int,Int)]) : DataSet[(Int,Int)] = {
    users.cross(items)
      .leftOuterJoin(exclude).where(0,1).equalTo(0,1)
      .apply((l,r,o:Collector[(Int,Int)])=>{
        Option(r) match {
          case Some(_) => ()
          case None => o.collect(l)
        }
      })
  }

  def predictions(als:Predictor[ALS], users : DataSet[Int], items : DataSet[Int], exclude : DataSet[(Int,Int)]): DataSet[(Int, Int, Int)] = {
    val userItemPairs = getUserItemPairs(users,items,exclude)
    val topKcopy = topK
    predictScores(als, userItemPairs)
      .groupBy(0)
      .combineGroup((elements, collector : Collector[(Int,Array[(Int,Double)])])=>{
        val bufferedElements = elements.buffered
        val head = bufferedElements.head
        var topKitems = mutable.SortedSet[(Int,Int,Double)]()(Ordering[(Double, Int)].reverse.on(x => (x._3, x._2)))
        for(e <- bufferedElements){
          topKitems += e
          if(topKitems.size > topKcopy){
            topKitems = topKitems.dropRight(1)
          }
        }
        collector.collect((head._1,topKitems.toArray.map(x=>(x._2, x._3))))
      })
      .withForwardedFields("0")
      .groupBy(0)
      .reduce((l,r)=>{
        var topKitems = mutable.SortedSet[(Int,Double)]()(Ordering[(Double, Int)].reverse.on(x => (x._2, x._1)))
        topKitems ++= l._2 ++= r._2
        (l._1, topKitems.take(topKcopy).toArray)
      })
      .flatMap(x=>for(e<-x._2.zip(1 to topKcopy)) yield (x._1, e._1._1, e._2))
  }

  def idcgs(test: DataSet[(Int,Int,Double)]): DataSet[(Int, Double)] = {
    val topKcopy = topK
    test
      .groupBy(0)
      .sortGroup(2, Order.DESCENDING)
      .reduceGroup(elements => {
        val bufferedElements = elements.buffered
        val user = bufferedElements.head._1
        val idcg = bufferedElements
          .map(_._3)
          .zip((1 to topKcopy).toIterator)
          .map { case (rel, rank) => rel / (scala.math.log(rank + 1) / scala.math.log(2)) }
          .sum

        (user, idcg)
      })
  }

  def dcgs(predictions: DataSet[(Int,Int,Int)], test: DataSet[(Int,Int,Double)]) : DataSet[(Int,Double)] = predictions
    .leftOuterJoin(test)
    .where(0, 1)
    .equalTo(0, 1)
    .apply((l,r) => (l,Option(r)))
    .groupBy(_._1._1)
    .reduceGroup( elements => {
      val bufferedElements = elements.buffered
      val user = bufferedElements.head._1._1
      val dcg : Double = bufferedElements
        .map {
          case ((u1, i1, rank), Some((u2, i2, rel))) =>
            rel / (scala.math.log(rank + 1) / scala.math.log(2))
          case (l, None) =>
            0.0
        }
        .sum
      (user, dcg)
    })

  def ndcgs(dcgs:DataSet[(Int,Double)], idcgs:DataSet[(Int,Double)]) : DataSet[(Int,Double)] = idcgs
    .join(dcgs)
    .where(0)
    .equalTo(0)
    .apply((idcg,dcg)=>(idcg._1, dcg._2/idcg._2))

  def averageNdcg(rankingPredictions : DataSet[(Int,Int,Int)], test : DataSet[(Int,Int,Double)]) : Double = {
    val idcgsV = idcgs(test)
    val dcgsV = dcgs(rankingPredictions, test)
    val ndcgsV = ndcgs(dcgsV, idcgsV)

//    ndcgsV.map(_._2).mean().collect().head
    ndcgsV.sum(1).map(_._2).collect().head/ndcgsV.count()
  }

  def joinWithTest(rankingPredictions : DataSet[(Int,Int,Int)], test : DataSet[(Int,Int,Double)]) : DataSet[(Int,Int,Option[Int],String)] = {
    rankingPredictions.fullOuterJoin(test).where(0,1).equalTo(0,1).apply((l,r,o:Collector[(Int,Int,Option[Int],String)]) => {
      (Option(l), Option(r)) match {
        case (None, Some(rr)) => o.collect((rr._1, rr._2, None, "false_negative"))
        case (Some(ll), None) => o.collect((ll._1, ll._2, Some(ll._3), "false_positive"))
        case (Some(ll), Some(rr)) => o.collect((ll._1, ll._2, Some(ll._3), "true_positive"))
        case default => ()
      }
    })
  }

  def countTypes(joined : DataSet[(Int,Int,Option[Int],String)]) : DataSet[(Int,Int,Int,Int)] = {
    joined
      .groupBy(0)
      .reduceGroup(elements => {
        val bufferedElements = elements.buffered
        val user=bufferedElements.head._1
        var truePositive, falsePositive, falseNegative=0
        for((_,_,_,t)<-bufferedElements){
          t match {
            case "true_positive" => truePositive+=1
            case "false_positive" => falsePositive+=1
            case "false_negative" => falseNegative+=1
          }
        }
        (user,truePositive,falsePositive,falseNegative)
      })
  }


  def countTypesUpToK(joined : DataSet[(Int,Int,Option[Int],String)]) : DataSet[(Int,Int,Int,Int,Int)] = joined
    .map(x=>x match {
      case (user,item,Some(rank),t)=>(user,item,rank,t)
      case (user,item,None,t)=>(user,item,0,t)
    })
    .groupBy(0)
    .sortGroup(2, Order.ASCENDING)
    .reduceGroup((elements,collector:Collector[(Int,Int,Int,Int,Int)]) => {
      val bufferedElements = elements.toList
      val user=bufferedElements.head._1
      var truePositive, falsePositive, falseNegative=0
      val allTruePositive = bufferedElements.count(_._4=="true_positive")
      for((_,_,rank,t)<-bufferedElements) {
        t match {
          case "true_positive" => truePositive += 1
          case "false_positive" => falsePositive += 1
          case "false_negative" => falseNegative += 1
        }
        if(rank != 0){
          collector.collect((user,rank,truePositive,falsePositive,falseNegative+(allTruePositive-truePositive)))
        }
      }
    })


  def precisions(rankingPredictions : DataSet[(Int,Int,Int)], test : DataSet[(Int,Int,Double)]) : DataSet[(Int,Double)] = {
    val countedPerUser = countTypes(joinWithTest(rankingPredictions,test))

    val calculatePrecision = (user:Int, truePositive:Int, falsePositive:Int, falseNegative:Int) =>
      truePositive.toDouble/(truePositive.toDouble+falsePositive.toDouble)

    countedPerUser.map(x=>(x._1, calculatePrecision.tupled(x)))
  }

  def recalls(rankingPredictions : DataSet[(Int,Int,Int)], test : DataSet[(Int,Int,Double)]) : DataSet[(Int,Double)] = {
    val countedPerUser = countTypes(joinWithTest(rankingPredictions,test))

    val calculateRecall = (user:Int, truePositive:Int, falsePositive:Int, falseNegative:Int) =>
      truePositive.toDouble/(truePositive.toDouble+falseNegative.toDouble)

    countedPerUser.map(x=>(x._1, calculateRecall.tupled(x)))
  }

  def meanAveragePrecisionAndRecall(rankingPredictions : DataSet[(Int,Int,Int)], test : DataSet[(Int,Int,Double)]) : DataSet[(Int,Double,Double)] = {
    val counted = countTypesUpToK(joinWithTest(rankingPredictions,test))

    val calculatePrecision = (user:Int, truePositive:Int, falsePositive:Int, falseNegative:Int) =>
      truePositive.toDouble/(truePositive.toDouble+falsePositive.toDouble)

    val calculateRecall = (user:Int, truePositive:Int, falsePositive:Int, falseNegative:Int) =>
      truePositive.toDouble/(truePositive.toDouble+falseNegative.toDouble)

    counted
      .map(x=>(x._1, x._3, x._4, x._5))
      .map(x=>(x._1, calculatePrecision.tupled(x), calculateRecall.tupled(x)))
      .groupBy(0)
      .reduceGroup( elements => {
        val bufferedElements = elements.toList
        ( bufferedElements.head._1,
          bufferedElements.map(_._2).sum/bufferedElements.length,
          bufferedElements.map(_._3).sum/bufferedElements.length )
      })
  }
}
