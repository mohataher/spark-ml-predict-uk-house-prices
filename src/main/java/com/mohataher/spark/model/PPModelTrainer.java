package com.mohataher.spark.model;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by bsmtaa on 17/11/2016.
 */
public class PPModelTrainer implements Serializable{

    public Double testMSE(JavaRDD<LabeledPoint> data, DecisionTreeModel model, JavaPairRDD<Double, Double> predictionAndLabel) {
        Double testMSE =
                predictionAndLabel.map(new Function<Tuple2<Double, Double>, Double>() {
                    @Override
                    public Double call(Tuple2<Double, Double> pl) {
                        Double diff = pl._1() - pl._2();
                        return diff * diff;
                    }
                }).reduce(new Function2<Double, Double, Double>() {
                    @Override
                    public Double call(Double a, Double b) {
                        return a + b;
                    }
                }) / data.count();
        return testMSE;
    }

    /**
     * Tests the model
     * @param testData
     * @param model the model used to predict the test the data
     * @return {@link JavaPairRDD} with the results of the test
     */
    public JavaPairRDD<Double, Double> testModel(JavaRDD<LabeledPoint> testData, final DecisionTreeModel model) {
        // Evaluate model on test instances and compute test error
        return testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            public Tuple2<Double, Double> call(LabeledPoint p) {
                return new Tuple2<>(model.predict(p.features()), p.label());
            }
        });
    }

    /**
     *Train the model
     * @param trainingData training data
     * @return The decision tree created by the training routine.
     */
    public DecisionTreeModel trainModel(JavaRDD<LabeledPoint> trainingData) {
        // Set parameters.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "variance";
        Integer maxDepth = 5;
        Integer maxBins = 32;

        // Train a DecisionTree model.
        return DecisionTree.trainRegressor(trainingData,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);
    }
}
