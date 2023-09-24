package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */
public class MathematicalFunctions {
    public static double sig(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigder(double x) {
        return sig(x) * (1.0 - sig(x));
    }
}
