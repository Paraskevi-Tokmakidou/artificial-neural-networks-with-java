package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */

public class GaConfig {
    private static int count_of_population = 300;
    private static int max_epoches = 100;
    private static Double crossoverRatio = 0.94; // 94%
    private static Double elitismRatio = 0.04; // 4%
    private static Double mutatioRatio = 0.02; // 2%
    private static Double min_value = 0.0;
    private static Double max_value = 1.0;

    /**
     * @return the count_of_population
     */
    public static int getCount_of_population() {
        return count_of_population;
    }

    /**
     * @return the max_epoches
     */
    public static int getMax_epoches() {
        return max_epoches;
    }

    /**
     * @return the crossoverRatio
     */
    public static Double getCrossoverRatio() {
        return crossoverRatio;
    }

    /**
     * @return the elitismRatio
     */
    public static Double getElitismRatio() {
        return elitismRatio;
    }

    /**
     * @param aElitismRatio the elitismRatio to set
     */
    public static void setElitismRatio(Double aElitismRatio) {
        elitismRatio = aElitismRatio;
    }

    /**
     * @return the mutatioRatio
     */
    public static Double getMutatioRatio() {
        return mutatioRatio;
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
