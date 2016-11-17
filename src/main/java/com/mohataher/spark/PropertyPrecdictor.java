package com.mohataher.spark;

/**
 * Created by bsmtaa on 17/11/2016.
 */

import com.mohataher.spark.dataloader.PPDataLoader;

import com.mohataher.spark.model.PPModelTrainer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

public class PropertyPrecdictor {

    public static void main(String[] args) {


        SparkConf sparkConf = new SparkConf()
                .setAppName("JavaDecisionTreeRegressionExample")
                .setMaster("local[10]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        // Load and parse the data file.
        String datapath = "data/pp-complete.csv";

        JavaRDD<LabeledPoint> data = PPDataLoader.makeLabeledPointJavaRDD(jsc, datapath);

        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        final DecisionTreeModel model = PPModelTrainer.trainModel(trainingData);

        JavaPairRDD<Double, Double> predictionAndLabel = PPModelTrainer.testModel(testData, model);

        PPModelTrainer.testMSE(data, model, predictionAndLabel);

    }

}
