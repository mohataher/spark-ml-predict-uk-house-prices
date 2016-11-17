package com.mohataher.spark.dataloader;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * Created by bsmtaa on 17/11/2016.
 */
public class PPDataLoader {
    public static JavaRDD<LabeledPoint> makeLabeledPointJavaRDD(JavaSparkContext jsc, String datapath) {
        //JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        JavaRDD<LabeledPoint> data;
        data=jsc.textFile(datapath).map(line-> {
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

    private static Double convertPriceToDouble(String priceStr) {
        return new Double(priceStr.replaceAll("\"", ""));
    }

    private static Double convertLeaseDurationToInt(String leaseDuration) {
        //F = Freehold, L= Leasehold
        if (leaseDuration.equalsIgnoreCase("\"F\""))
            return 0D;
        if (leaseDuration.equalsIgnoreCase("\"L\""))
            return 1D;
        if (leaseDuration.equalsIgnoreCase("\"U\""))
            return 2D;

        throw new IllegalStateException("Shouldn't reach here. Strange item, "+leaseDuration);
    }

    private static Integer convertPropertyTypeToInt(String propertyType) {
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

    private static Integer convertTownToBitWise(String townStr) {
        return ("\"London\"".equalsIgnoreCase(townStr))?1:0;
    }
}
