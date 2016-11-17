package com.mohataher.spark.dataloader;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by bsmtaa on 17/11/2016.
 */
public class PPDataLoaderTest {

    String dataPathFull = "data/pp-complete.csv";
    String dataPathSample = "data/pp-sample.csv";

    String dataUrl = "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1" +
            ".amazonaws.com/pp-complete.csv";
    private JavaSparkContext jsc;

    @Before
    public void setUp() throws Exception {

        SparkConf sparkConf = new SparkConf()
                .setAppName("JavaDecisionTreeRegressionExampleTest")
                .setMaster("local[*]");
        jsc = new JavaSparkContext(sparkConf);


    }

    @After
    public void tearDown() throws Exception {
        jsc.close();
    }

    @Test
    public void testMakeLabeledPointJavaRDD_loadsSampleFile() throws Exception {
        JavaRDD<LabeledPoint> data = new PPDataLoader(
                dataPathSample,
                dataPathSample,
                dataUrl
        ).makeLabeledPointJavaRDD(jsc);
        assertNotNull(data);
        assertEquals(data.count(), 1000);
    }

    @Test
    public void testMakeLabeledPointJavaRDD_cannotDownloadFile() throws Exception {
        JavaRDD<LabeledPoint> data = new PPDataLoader(
                dataPathFull+"RANDOM",
                dataPathSample,
                dataUrl+"RANDON"
        ).makeLabeledPointJavaRDD(jsc);
        assertNotNull(data);
        assertEquals(data.count(), 1000);
    }

    @Test
    @Ignore
    public void testMakeLabeledPointJavaRDD_downloadFile() throws Exception {
        JavaRDD<LabeledPoint> data = new PPDataLoader(
                dataPathFull,
                dataPathSample,
                dataUrl
        ).makeLabeledPointJavaRDD(jsc);
        assertNotNull(data);
        assertEquals(data.count(), 1000);
    }

}