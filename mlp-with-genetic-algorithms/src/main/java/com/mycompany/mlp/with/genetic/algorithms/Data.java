package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */

import java.util.ArrayList;

public class Data {
    private static ArrayList<ArrayList<Double>> trainPatterns;
    private static ArrayList<ArrayList<Double>> testPatterns;
    private static int dimension;

    /**
     * @return the trainPatterns
     */
    public static ArrayList<ArrayList<Double>> getTrainPatterns() {
        return trainPatterns;
    }

    /**
     * @param trainPatterns the trainPatterns to set
     */
    public static void setTrainPatterns(ArrayList<ArrayList<Double>> trainPatterns) {
        Data.trainPatterns = trainPatterns;
    }

    /**
     * @return the dimension
     */
    public static int getDimension() {
        return dimension;
    }

    /**
     * @param dimension the dimension to set
     */
    public static void setDimension(int dimension) {
        Data.dimension = dimension;
    }

    /**
     * @return the testPatterns
     */
    public static ArrayList<ArrayList<Double>> getTestPatterns() {
        return testPatterns;
    }

    /**
     * @param aTestPatterns the testPatterns to set
     */
    public static void setTestPatterns(ArrayList<ArrayList<Double>> aTestPatterns) {
        testPatterns = aTestPatterns;
    }
}
