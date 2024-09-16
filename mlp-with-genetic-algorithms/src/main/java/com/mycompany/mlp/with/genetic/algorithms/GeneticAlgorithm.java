package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class GeneticAlgorithm {
    private ArrayList<ArrayList<Double>> _population;
    private ArrayList<ArrayList<Double>> _newPopulation;
    private final ArrayList<ArrayList<Double>> _patterns;
    private final int _dimensionPatterns;
    private final int _nodes;
    private final int _countGenesOfChromosome;
    private final int _countOfPopulation;
    private final int _maxEpochs;
    private final double _crossoverRatio;
    private final double _elitismRatio;
    private final double _mutationRatio;
    private final int _elitismCountOfChromosomesThatPassToNextEpoch;
    private final boolean _wantToDisplayTrainErrorInEachEpoch;
    private final GENETIC_CROSSOVER_OPTIONS _crossoverOption;
    private Random random;
    private MLP mlp;

    GeneticAlgorithm(int dimension, GENETIC_CROSSOVER_OPTIONS geneticCrossoverOption) {
        this.random = new Random();
        this._countOfPopulation = 100;
        this._maxEpochs = 50;
        this._crossoverRatio = 0.95; // 95%
        this._elitismRatio = 0.04; // 4%
        this._mutationRatio = 0.02; // 2%
        this._elitismCountOfChromosomesThatPassToNextEpoch = (int) Math
                .round(this._countOfPopulation * this._elitismRatio);
        this._countGenesOfChromosome = dimension;
        this._patterns = Data.getTrainPatterns();
        this._dimensionPatterns = Data.getDimension();
        this._nodes = Data.getNodes();
        this._population = new ArrayList<>();
        this._wantToDisplayTrainErrorInEachEpoch = true;
        this._crossoverOption = geneticCrossoverOption;
        mlp = new MLP();
    }

    private void randomInitializationGenes() {
        this._population = new ArrayList<>();

        double min = 0;
        double max = 1;

        for (int i = 0; i < this._countOfPopulation; i++) {
            ArrayList<Double> tempChromosome = new ArrayList<>();
            for (int j = 0; j < this._countGenesOfChromosome; j++) {
                Double randomValue = min + (max - min) * this.random.nextDouble();
                tempChromosome.add(randomValue);
            }

            tempChromosome.addAll(Collections.nCopies(3, -999.999)); // Default values for future usage

            this._population.add(tempChromosome);
        }
    }

    private void calculateFitness() {
        for (int i = 0; i < this._countOfPopulation; i++) {
            ArrayList<Double> tempChromosome = new ArrayList<>(this._population.get(i));
            Double fitnessValueOfChromosome = mlp.calculateError(this._population.get(i));
            tempChromosome.set(this._countGenesOfChromosome, fitnessValueOfChromosome);
            this._population.set(i, tempChromosome);
        }
    }

    private void fitnessScaling() {
        boolean hasNegativeFitnessValue = false;
        for (ArrayList<Double> chromosome : this._population) {
            if (chromosome.get(this._countGenesOfChromosome) < 0)
                hasNegativeFitnessValue = true;
        }

        if (hasNegativeFitnessValue) {
            double minFitnessValue = this._population.get(0).get(this._countGenesOfChromosome);
            for (int i = 0; i < this._population.size(); i++) {
                if (this._population.get(i).get(this._countGenesOfChromosome) < minFitnessValue)
                    minFitnessValue = this._population.get(i).get(this._countGenesOfChromosome);
            }

            minFitnessValue = Math.abs(minFitnessValue);
            for (int i = 0; i < this._population.size(); i++) {
                double scaledFitnessValue = 1.0 * this._population.get(i).get(this._countGenesOfChromosome)
                        + minFitnessValue + 1; // +1 to avoid devide by zero after
                this._population.get(i).set(this._countGenesOfChromosome, scaledFitnessValue);
            }
        }
    }

    // Devide the fitness value of each chromosome
    // by the total values of all fitnesses
    private void normalizeFitnessValues() {
        double sumFitnessValue = 0.0;
        for (int k = 0; k < this._population.size(); k++) {
            sumFitnessValue += this._population.get(k).get(this._countGenesOfChromosome);
        }

        for (int i = 0; i < this._population.size(); i++) {
            ArrayList<Double> inner = new ArrayList<>(this._population.get(i));
            Double normalizationValue = (inner.get(this._countGenesOfChromosome) / sumFitnessValue);
            inner.set(this._countGenesOfChromosome + 1, normalizationValue);
            this._population.set(i, inner);
        }
    }

    private void sortPopulationByFitnessValue() {
        Collections.sort(this._population,
                (a, b) -> Double.compare(a.get(this._countGenesOfChromosome), b.get(this._countGenesOfChromosome)));
    }

    private void calculateCumulativeSumOfNormalizedFitnessValues() {
        for (int i = 0; i < this._population.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>(this._population.get(i));
            double sum = temp.get(this._countGenesOfChromosome + 1);
            for (int j = (i + 1); j < this._population.size(); j++) {
                sum += this._population.get(j).get(this._countGenesOfChromosome + 1);
            }
            temp.set(this._countGenesOfChromosome + 2, sum);
            this._population.set(i, temp);
        }
    }

    private void elitism() {
        for (int i = 0; i < this._elitismCountOfChromosomesThatPassToNextEpoch; i++) {
            this._newPopulation.add(this._population.get(i));
        }
    }

    private void crossover(ArrayList<Double> parent1, ArrayList<Double> parent2) {
        // 0 - count genes Of chromosome
        int randomGenePosition = this.random.nextInt(this._countGenesOfChromosome - 1) + 1;

        int secondRandomGenePosition;
        do {
            // 0 - count genes Of chromosome
            secondRandomGenePosition = this.random.nextInt(this._countGenesOfChromosome - 1) + 1;
        } while (secondRandomGenePosition == randomGenePosition);

        if (secondRandomGenePosition < randomGenePosition) {
            int temp = randomGenePosition;
            randomGenePosition = secondRandomGenePosition;
            secondRandomGenePosition = temp;
        }

        if (this._crossoverOption == GENETIC_CROSSOVER_OPTIONS.SINGLE) {
            this.singlePointCrossover(parent1, parent2, randomGenePosition);
        } else {
            this.doublePointCrossover(parent1, parent2, randomGenePosition, secondRandomGenePosition);
        }
    }

    private void singlePointCrossover(ArrayList<Double> parent1, ArrayList<Double> parent2, int randomGenePosition) {
        ArrayList<Double> child = new ArrayList<>();
        // 0 - randomGenePosition
        child.addAll(parent1.subList(0, randomGenePosition));
        // randomGenePosition - end
        child.addAll(parent2.subList(randomGenePosition, this._countGenesOfChromosome));
        child.addAll(Collections.nCopies(3, -1.0));
        this._newPopulation.add(child);

        child = new ArrayList<>();
        // 0 - randomGenePosition
        child.addAll(parent2.subList(0, randomGenePosition));
        // randomGenePosition - end
        child.addAll(parent1.subList(randomGenePosition, this._countGenesOfChromosome));
        child.addAll(Collections.nCopies(3, -1.0));
        this._newPopulation.add(child);
    }

    private void doublePointCrossover(ArrayList<Double> parent1, ArrayList<Double> parent2, int randomGenePosition,
            int secondRandomGenePosition) {
        ArrayList<Double> child = new ArrayList<>();
        // 0 - randomGenePosition
        child.addAll(parent1.subList(0, randomGenePosition));
        // randomGenePosition - secondRandomGenePosition
        child.addAll(parent2.subList(randomGenePosition, secondRandomGenePosition));
        // secondRandomGenePosition - end
        child.addAll(parent1.subList(secondRandomGenePosition, this._countGenesOfChromosome));
        child.addAll(Collections.nCopies(3, -1.0));
        this._newPopulation.add(child);

        child = new ArrayList<>();
        // 0 - randomGenePosition
        child.addAll(parent2.subList(0, randomGenePosition));
        // randomGenePosition - secondRandomGenePosition
        child.addAll(parent1.subList(randomGenePosition, secondRandomGenePosition));
        // secondRandomGenePosition - end
        child.addAll(parent2.subList(secondRandomGenePosition, this._countGenesOfChromosome));
        child.addAll(Collections.nCopies(3, -1.0));
        this._newPopulation.add(child);
    }

    private void rouletteWheel() {
        while (this._newPopulation.size() < this._countOfPopulation) {
            ArrayList<ArrayList<Double>> tempChromosomes = new ArrayList<>();

            for (int i = 0; i < 2; i++) {
                double randomCumulativeValue = Math.random();
                boolean findParent = false;
                for (int k = 1; k < this._population.size(); k++) {
                    if (randomCumulativeValue > this._population.get(k).get(this._countGenesOfChromosome + 2)) {
                        tempChromosomes.add(this._population.get(k - 1));
                        findParent = true;
                        break;
                    }
                }

                if (!findParent) {
                    tempChromosomes.add(this._population.get(this._population.size() - 1));
                }
            }

            double crossoverProbability = Math.random();
            if (crossoverProbability <= this._crossoverRatio) {
                // From 2 parants, produce 2 child
                this.crossover(tempChromosomes.get(0), tempChromosomes.get(1));
            } else {
                this._newPopulation.add(tempChromosomes.get(0));
                this._newPopulation.add(tempChromosomes.get(1));
            }
        }
    }

    private void mutation() {
        for (int i = this._elitismCountOfChromosomesThatPassToNextEpoch; i < this._newPopulation.size(); i++) {
            ArrayList<Double> mutatedChromosome = new ArrayList<>();
            ArrayList<Double> currentChromosome = this._newPopulation.get(i);

            for (int j = 0; j < currentChromosome.size(); j++) {
                double gene = currentChromosome.get(j);

                if (j < currentChromosome.size() - 3) {
                    double mutationProbability = Math.random();

                    if (mutationProbability <= this._mutationRatio) {
                        gene += gene * mutationProbability;
                    }
                }

                mutatedChromosome.add(gene);
            }

            this._newPopulation.set(i, mutatedChromosome);
        }
    }

    private void selection() {
        this.rouletteWheel();
    }

    private void fitnessFunctions() {
        this.calculateFitness();
        this.fitnessScaling();
        this.normalizeFitnessValues();
        this.sortPopulationByFitnessValue();
        this.calculateCumulativeSumOfNormalizedFitnessValues();
    }

    public ArrayList<Double> getBestChromosome() {
        for (int i = 0; i < this._maxEpochs; i++) {
            if (i == 0) {
                this.randomInitializationGenes();
            } else {
                this._newPopulation = new ArrayList<>();
                this.elitism();
                this.selection();
                this.mutation();

                this._population = new ArrayList<>(this._newPopulation);
                this._newPopulation = new ArrayList<>();
            }

            this.fitnessFunctions();

            if (this._wantToDisplayTrainErrorInEachEpoch) {
                this.displayTrainError(i);
            }
        }

        return this._population.get(0);
    }

    private void displayTrainError(int i) {
        System.out.println("Genetic Train Error: i[" + i + "] = "
                + this._population.get(0).get(this._countGenesOfChromosome));
    }
}
