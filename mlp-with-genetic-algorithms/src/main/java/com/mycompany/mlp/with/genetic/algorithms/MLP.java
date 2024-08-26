package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */

import java.util.ArrayList;
import java.util.Random;

public class MLP {
    private final ArrayList<ArrayList<Double>> _patterns;
    private ArrayList<Double> _uniqueOutputClasses;
    private ArrayList<Double> _weights;
    private final int _count;
    private final int _nodes;
    private final int _dimension;
    private final double _learningRate;
    private final int _maxEpoches;
    private final boolean _wantToDisplayTrainErrorInEachEpoch;

    MLP() {
        this._patterns = Data.getTrainPatterns();
        this._count = this._patterns.size();
        this._nodes = Data.getNodes();
        this._dimension = Data.getDimension();
        this._weights = new ArrayList<>();
        this._learningRate = Data.getLearning_rate();
        this._maxEpoches = Data.getMax_epoches();
        this._wantToDisplayTrainErrorInEachEpoch = false;

        System.out.println("Count of patterns: " + this._count);
        System.out.println("Dimension: " + this._dimension);
        System.out.println("Nodes: " + this._nodes);
    }

    private void initializeRandomWeights(Integer countOfWeightsDimension) {
        for (int i = 0; i < countOfWeightsDimension; i++) {
            Random r = new Random();
            double min = -1;
            double max = 1;

            Double randomValue = min + (max - min) * r.nextDouble();
            this._weights.add(randomValue);
        }
    }

    public void initializeWeights(Boolean geneticOption) {
        int countOfWeightsDimension = (this._dimension + 2) * this._nodes;
        System.out.println("Count of weights: " + countOfWeightsDimension);

        if (geneticOption) {
            GeneticAlgorithm ga = new GeneticAlgorithm(countOfWeightsDimension);
            this._weights = ga.getBestChromosome();
        } else {
            this.initializeRandomWeights(countOfWeightsDimension);
        }
    }

    private void findUniqueClasses() {
        this._uniqueOutputClasses = new ArrayList<>();

        for (int i = 0; i < this._patterns.size(); i++) {
            if (this._uniqueOutputClasses.isEmpty())
                this._uniqueOutputClasses.add(this._patterns.get(i).get(this._dimension));
            else {
                if (this._uniqueOutputClasses.indexOf(this._patterns.get(i).get(this._dimension)) == -1) {
                    this._uniqueOutputClasses.add(this._patterns.get(i).get(this._dimension));
                }
            }
        }

        System.out.println("Unique classes: " + _uniqueOutputClasses);
    }

    public static double getOutput(ArrayList<Double> pattern, int nodes, int dimension, ArrayList<Double> weights) {
        double sum = 0.0;

        for (int i = 0; i < nodes; i++) {
            double arg = 0.0;

            for (int j = 0; j < dimension; j++) {
                int posj = (dimension + 2) * (i + 1) - (dimension + 1) + (j + 1);
                arg += weights.get(posj - 1) * pattern.get((j + 1) - 1);
            }

            int posbias = (dimension + 2) * (i + 1);
            arg += weights.get(posbias - 1);
            int pos = (dimension + 2) * (i + 1) - (dimension + 1);
            sum += weights.get(pos - 1) * MathematicalFunctions.sig(arg);
        }

        return sum;
    }

    public static double getTrainError(ArrayList<ArrayList<Double>> patterns, int dimension, int nodes,
            ArrayList<Double> weights) {
        double sum = 0.0;

        for (int i = 0; i < patterns.size(); i++) {
            double yx = patterns.get(i).get(dimension); // The desired output
            double ox = getOutput(patterns.get(i), nodes, dimension, weights);
            sum += (ox - yx) * (ox - yx);
        }

        return sum;
    }

    private ArrayList<Double> getPatternDeriv(ArrayList<Double> pattern) {
        ArrayList<Double> patternDeriv = new ArrayList<>();
        this._weights.forEach(_item -> {
            patternDeriv.add(0.0);
        });

        for (int i = 0; i < this._nodes; i++) {
            double arg = 0.0;

            for (int j = 0; j < this._dimension; j++) {
                int pos = (this._dimension + 2) * (i + 1) - (this._dimension + 1) + (j + 1);
                arg += pattern.get(j) * this._weights.get(pos - 1);
            }
            arg += this._weights.get((this._dimension + 2) * (i + 1) - 1);
            double s = MathematicalFunctions.sig(arg);
            double s1 = MathematicalFunctions.sigder(arg);
            patternDeriv.set((this._dimension + 2) * (i + 1) - (this._dimension + 1) - 1, s);
            patternDeriv.set((this._dimension + 2) * (i + 1) - 1,
                    this._weights.get((this._dimension + 2) * (i + 1) - (this._dimension + 1) - 1) * s1);
            for (int j = 0; j < this._dimension; j++) {
                int pos = (this._dimension + 2) * (i + 1) - (this._dimension + 1) + j;
                patternDeriv.set(pos - 1, this._weights.get((this._dimension + 2) * (i + 1) - (this._dimension + 1) - 1)
                        * pattern.get((j + 1) - 1) * s1);
            }
        }

        return patternDeriv;
    }

    private ArrayList<Double> getDeriv() {
        ArrayList<Double> deriv = new ArrayList<>();
        this._weights.forEach(_item -> {
            deriv.add(0.0);
        });

        for (int i = 0; i < this._patterns.size(); i++) {
            ArrayList<Double> patternDeriv = getPatternDeriv(this._patterns.get(i));
            double yx = this._patterns.get(i).get(this._dimension);
            double ox = getOutput(this._patterns.get(i), this._nodes, this._dimension, this._weights);

            for (int j = 0; j < deriv.size(); j++) {
                deriv.set(j, 2.0 * (ox - yx) * patternDeriv.get(j));
            }
        }
        return deriv;
    }

    public double train(Boolean geneticOption) {
        this.initializeWeights(geneticOption);
        this.findUniqueClasses();

        double trainError = -1;
        for (int i = 0; i < this._maxEpoches; i++) {
            trainError = getTrainError(this._patterns, this._dimension, this._nodes, this._weights);

            if (trainError < 1e-5)
                break;
            else {
                ArrayList<Double> deriv = getDeriv();
                for (int j = 0; j < this._weights.size(); j++) {
                    this._weights.set(j, this._weights.get(j) - this._learningRate * deriv.get(j));
                }
            }

            if (this._wantToDisplayTrainErrorInEachEpoch) {
                this.displayTrainError(i, trainError);
            }
        }

        if (trainError > -1) {
            System.out.println("******************");
        } else {
            System.out.println("Something went wrong!");
        }

        return trainError;
    }

    public double getTestError() {
        ArrayList<ArrayList<Double>> testPatterns = Data.getTestPatterns();
        if (testPatterns != null) {
            return MLP.getTrainError(testPatterns, this._dimension, this._nodes, this._weights);
        } else {
            return -1.0;
        }

    }

    private void displayTrainError(int i, double trainError) {
        System.out.println("MLP Train Error: i[" + i + "] = " + trainError);
    }
}
