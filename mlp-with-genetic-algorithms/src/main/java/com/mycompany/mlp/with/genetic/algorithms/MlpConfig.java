package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */

public class MlpConfig {
    private static int nodes = 10;
    private static Double learning_rate = 0.01;
    private static int max_epoches = 100;
    private static Double min_value = 0.0;
    private static Double max_value = 1.0;

    /**
     * @return the nodes
     */
    public static int getNodes() {
        return nodes;
    }

    /**
     * @return the learning_rate
     */
    public static Double getLearning_rate() {
        return learning_rate;
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

    /**
     * @return the min_value
     */
    public static Double getMin_value() {
        return min_value;
    }

    /**
     * @return the max_value
     */
    public static Double getMax_value() {
        return max_value;
    }
}
