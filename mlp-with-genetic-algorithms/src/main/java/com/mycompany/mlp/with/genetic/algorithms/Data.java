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
    private static int nodes = 10;
    private static double learning_rate = 0.01;
    private static int max_epoches = 50;

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

    /**
     * @return the nodes
     */
    public static int getNodes() {
        return nodes;
    }

    /**
     * @param aNodes the nodes to set
     */
    public static void setNodes(int aNodes) {
        nodes = aNodes;
    }

    /**
     * @return the learning_rate
     */
    public static double getLearning_rate() {
        return learning_rate;
    }

    /**
     * @param aLearning_rate the learning_rate to set
     */
    public static void setLearning_rate(double aLearning_rate) {
        learning_rate = aLearning_rate;
    }

    /**
     * @return the max_epoches
     */
    public static int getMax_epoches() {
        return max_epoches;
    }

    /**
     * @param aMax_epoches the max_epoches to set
     */
    public static void setMax_epoches(int aMax_epoches) {
        max_epoches = aMax_epoches;
    }
}
