package com.mohataher.spark;

/**
 * Created by bsmtaa on 17/11/2016.
 */

import com.mohataher.spark.dataloader.PPDataLoader;

import com.mohataher.spark.model.PPModelTrainer;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

public class PropertyPredictor {

    final transient static Logger logger = Logger.getLogger(PropertyPredictor.class);

    public static void main(String[] args) {

        String dataPathFull = "data/pp-sample.csv";
        String dataPathSample = "data/pp-sample.csv";

        String dataUrl = "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1" +
                ".amazonaws.com/pp-complete.csv";

        SparkConf sparkConf = new SparkConf()
                .setAppName("JavaDecisionTreeRegressionExample")
                .setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        // Load and parse the data file.
        JavaRDD<LabeledPoint> data = new PPDataLoader(
                dataPathFull,
                dataPathSample,
                dataUrl
        ).makeLabeledPointJavaRDD(jsc);

        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        PPModelTrainer pModelTrainer=new PPModelTrainer();

        final DecisionTreeModel model = pModelTrainer.trainModel(trainingData);

        JavaPairRDD<Double, Double> predictionAndLabel = pModelTrainer.testModel(testData, model);


        Double testMSE = pModelTrainer.testMSE(data, model, predictionAndLabel);

        logger.info("Test Mean Squared Error: " + testMSE);
        logger.info("Learned regression tree model:\n" + model.toDebugString());

        model.save(jsc.sc(), "myDecisionTreeRegressionModel");

    }

}
