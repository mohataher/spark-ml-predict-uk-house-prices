package com.mohataher.spark.dataloader;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.net.URL;

/**
 * Loads data from a CSV text file to {@link LabeledPoint}.
 * @author Mohamed Taher Alrefaie
 */
public class PPDataLoader implements Serializable{

    final transient static Logger logger = Logger.getLogger(PPDataLoader.class);


    private String dataPathFull = "data/pp-complete.csv";
    private String dataPathSample = "data/pp-sample.csv";

    private String dataUrl = "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1" +
            ".amazonaws.com/pp-complete.csv";
    private static final int connectionTimeout=10* 1000;
    private static final int readTimeout=10* 1000;

    public PPDataLoader(String dataPathFull, String dataPathSample, String dataUrl) {
        this.dataPathFull = dataPathFull;
        this.dataPathSample = dataPathSample;
        this.dataUrl = dataUrl;
    }

    /**
     * Loads data from a CSV text file to an RDD of {@link LabeledPoint}.
     * @param jsc
     * @return A Spark RDD of {@link LabeledPoint}
     */
    public JavaRDD<LabeledPoint> makeLabeledPointJavaRDD(JavaSparkContext jsc) {
        JavaRDD<LabeledPoint> data;
        String dataPath= dataPathFull;
        File file = new File(dataPathFull);
        if (!file.exists()){
            try {
                logger.info("Downloading data set from "+ dataUrl);
                FileUtils.copyURLToFile(new URL(dataUrl), file, connectionTimeout, readTimeout);
            } catch (IOException e) {
                logger.warn("FAILED to download the file on URL "+ dataUrl, e);
                dataPath= dataPathSample;
                logger.warn("USING sample file located at "+ dataPath );
            }
        }
        data=jsc.textFile(dataPath).map(line-> {
            String[] split = line.split("\\s*,\\s*");
            return new LabeledPoint(convertPriceToDouble(split[1]),
                    Vectors.dense(
                            convertTownToBitWise(split[11]),
                            convertPropertyTypeToInt(split[4]),
                            convertLeaseDurationToInt(split[6])
                    )
            );
        });
        return data;
    }

    private Double convertPriceToDouble(String priceStr) {
        return new Double(priceStr.replaceAll("\"", ""));
    }

    private Double convertLeaseDurationToInt(String leaseDuration) {
        //F = Freehold, L= Leasehold
        if (leaseDuration.equalsIgnoreCase("\"F\""))
            return 0D;
        if (leaseDuration.equalsIgnoreCase("\"L\""))
            return 1D;
        if (leaseDuration.equalsIgnoreCase("\"U\""))
            return 2D;

        throw new IllegalStateException("Shouldn't reach here. Strange item, "+leaseDuration);
    }

    private Integer convertPropertyTypeToInt(String propertyType) {
        //D = Detached, S = Semi-Detached, T = Terraced, F = Flats/Maisonettes, O = Other
        propertyType=propertyType.replaceAll("\"", "");
        if (propertyType.equals("D"))
            return 0;
        if (propertyType.equals("S"))
            return 1;

        if (propertyType.equals("T"))
            return 2;
        if (propertyType.equals("F"))
            return 3;

        if (propertyType.equals("O"))
            return 4;
        throw new IllegalStateException("Shouldn't reach here. Strange item, "+propertyType);
    }

    private Integer convertTownToBitWise(String townStr) {
        return ("\"London\"".equalsIgnoreCase(townStr))?1:0;
    }
}
