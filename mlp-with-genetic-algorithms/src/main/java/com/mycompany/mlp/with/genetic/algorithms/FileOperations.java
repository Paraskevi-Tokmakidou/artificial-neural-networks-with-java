package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;
import javax.swing.JFileChooser;

public class FileOperations {
    public boolean loadDataset(String fileName) {
        if (fileName.isEmpty())
            return false;

        ArrayList<ArrayList<Double>> patterns = new ArrayList<>();
        boolean isValid = true;
        int dimension = -1;

        try {
            FileReader file = new FileReader(fileName);
            try (Scanner in = new Scanner(file)) {
                int i = 0;
                // Read the file line-by-line
                while (in.hasNextLine()) {
                    if (isValid) {
                        String line = in.nextLine();

                        if (line.startsWith("//")) // For comments
                            continue;

                        ArrayList<Double> newPattern = new ArrayList<>();
                        String[] characteristics = line.split(",");
                        if (i == 0) {
                            dimension = characteristics.length;
                            for (String characteristic : characteristics) {
                                newPattern.add(Double.valueOf(characteristic));
                            }
                            patterns.add(newPattern);
                        } else if (characteristics.length == dimension) {
                            for (String characteristic : characteristics) {
                                newPattern.add(Double.valueOf(characteristic));
                            }
                            patterns.add(newPattern);
                        } else {
                            isValid = false;
                        }
                    }
                }

                if (isValid && fileName.endsWith(".train")) {
                    Data.setTrainPatterns(patterns);
                    Data.setDimension(dimension - 1); // The last column is the output class of pattern
                } else if (isValid && fileName.endsWith(".test")) {
                    if ((dimension - 1) == Data.getDimension()) {
                        Data.setTestPatterns(patterns);
                    } else {
                        isValid = false;
                    }

                }

                return isValid;
            }
        } catch (FileNotFoundException | NumberFormatException ex) {
            System.out.println("Cannot read " + fileName + " file");
            return false;
        }
    }

    public boolean chooseDatasetToLoad() {
        // Permisions for MAC devices to see files in Download folder
        System.setProperty("apple.awt.fileDialogForDirectories", "true");

        // Prompt the user to choose a .txt file from his system
        JFileChooser chooser = new JFileChooser();
        int returnVal = chooser.showOpenDialog(null);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            String filename = chooser.getSelectedFile().getAbsolutePath();
            return this.loadDataset(filename);
        } else {
            return false;
        }
    }
}
