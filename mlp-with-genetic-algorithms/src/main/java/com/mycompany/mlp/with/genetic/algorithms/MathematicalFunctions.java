package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */
public class MathematicalFunctions {
    public static Double sig(Double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static Double sigder(Double x) {
        return sig(x) * (1.0 - sig(x));
    }
}
