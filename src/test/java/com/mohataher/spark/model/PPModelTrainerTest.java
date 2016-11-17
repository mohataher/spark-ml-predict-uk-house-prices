package com.mohataher.spark.model;

import com.mohataher.spark.dataloader.PPDataLoader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by bsmtaa on 17/11/2016.
 */
public class PPModelTrainerTest {

    private JavaSparkContext jsc;
    private JavaRDD<LabeledPoint> data;
    private JavaRDD<LabeledPoint> trainingData;
    private JavaRDD<LabeledPoint> testData;

    @Before
    public void setUp() throws Exception {
        String dataPathFull = "data/pp-sample.csv";
        String dataPathSample = "data/pp-sample.csv";
        String dataUrl = null;

        SparkConf sparkConf = new SparkConf()
                .setAppName("JavaDecisionTreeRegressionExampleTest")
                .setMaster("local[*]");
        jsc = new JavaSparkContext(sparkConf);

        // Load and parse the data file.
        data = new PPDataLoader(
                dataPathFull,
                dataPathSample,
                dataUrl
        ).makeLabeledPointJavaRDD(jsc);

        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        trainingData = splits[0];
        testData = splits[1];

    }

    @After
    public void tearDown() throws Exception {
        jsc.close();
    }

    @Test
    public void testTrainTestMseModel() throws Exception {
        PPModelTrainer pModelTrainer=new PPModelTrainer();

        final DecisionTreeModel model = pModelTrainer.trainModel(trainingData);
        assertNotNull(model);
        assertTrue(model.numNodes()>0);

        JavaPairRDD<Double, Double> predictionAndLabel = pModelTrainer.testModel(testData, model);
        assertNotNull(predictionAndLabel);
        assertTrue(predictionAndLabel.count()>0);

        Double testMSE = pModelTrainer.testMSE(data, model, predictionAndLabel);
        assertNotNull(testMSE);
        assertTrue(testMSE>=0);

    }
}