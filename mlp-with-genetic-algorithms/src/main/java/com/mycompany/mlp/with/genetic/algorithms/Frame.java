package com.mycompany.mlp.with.genetic.algorithms;

/**
 *
 * @author Paraskevi Tokmakidou
 */

import java.awt.Color;
import java.awt.Event;
import java.awt.FlowLayout;
import java.awt.Menu;
import java.awt.MenuBar;
import java.awt.MenuItem;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Frame extends JFrame {
    private final String HOME_PAGE = "<html><h2>Home page</h2></html>";
    private final String ABOUT_PAGE = "<html><h2>About page</h2></html>";
    private final String SOMETHING_WRONG = "<html><h2>Something went wrong..</h2></html>";
    private final String LOAD_TRAIN_DATASET = "<html><h2>Loading dataset..</h2></html>";
    private final String LOAD_CUSTOM_TRAIN_DATASET = "<html><h2>Loading train dataset..</h2></html>";
    private final String LOAD_CUSTOM_TEST_DATASET = "<html><h2>Loading test dataset..</h2></html>";
    private final String SUCCESS_LOAD_IONOSPHERE_DATASETS = "<html><br/><h2>Succeed action!</h2>" +
            "<br/> <h3>Loaded ionosphere dataset.</h3></html>";
    private final String SUCCESS_LOAD_WINE_DATASETS = "<html><br/><h2>Succeed action!</h2>" +
            "<br/> <h3>Loaded wine dataset.</h3></html>";
    private final String SUCCESS_LOAD_IRIS_DATASETS = "<html><br/><h2>Succeed action!</h2>" +
            "<br/> <h3>Loaded iris dataset.</h3></html>";
    private final String SUCCESS_LOAD_CUSTOM_DATASETS = "<html><br/><h2>Succeed action!</h2></html>";
    private final String NEED_DATASETS = "<html><h2>Need to load a dataset first</h2></html>";
    private final String NEED_TRAIN_DATASETS = "<html><h2>Need to load a train dataset first</h2></html>";
    private final MenuBar menuBar;
    private final Menu menuMenu, datasetMenu, algorithmsMenu;
    private final MenuItem[] menuItems, datasetItems, algorithmsItems;
    private static JLabel label;
    private boolean loadedDatasets;

    Frame(String title) {
        // Configuration of display window
        super(title);
        setResizable(false);
        this.getContentPane().setBackground(Color.lightGray);
        // A flow layout arranges components in a left-to-right flow,
        // much like lines of text in a paragraph
        setLayout(new FlowLayout());

        // Menu configuration
        menuBar = new MenuBar();

        menuMenu = new Menu("MENU");
        menuItems = new MenuItem[2];
        menuItems[0] = new MenuItem("Home");
        menuItems[1] = new MenuItem("About");
        for (short i = 0; i < menuItems.length; i++) {
            menuMenu.add(menuItems[i]);
        }
        menuBar.add(menuMenu);

        datasetMenu = new Menu("LOAD DATASETS");
        datasetItems = new MenuItem[5];
        datasetItems[0] = new MenuItem("ionosphere");
        datasetItems[1] = new MenuItem("wine");
        datasetItems[2] = new MenuItem("iris");
        datasetItems[3] = new MenuItem("Load train dataset");
        datasetItems[4] = new MenuItem("Load test dataset");
        for (short i = 0; i < datasetItems.length; i++)
            datasetMenu.add(datasetItems[i]);
        menuBar.add(datasetMenu);

        algorithmsMenu = new Menu("NETWORKS-ALGORITHMS");
        algorithmsItems = new MenuItem[3];
        algorithmsItems[0] = new MenuItem("MLP with Back Propagation");
        algorithmsItems[1] = new MenuItem(
                "MLP with Back Propagation (Using Genetic Algorithm) - Single point crossover");
        algorithmsItems[2] = new MenuItem(
                "MLP with Back Propagation (Using Genetic Algorithm) - Double point crossover");
        for (short i = 0; i < algorithmsItems.length; i++)
            algorithmsMenu.add(algorithmsItems[i]);
        menuBar.add(algorithmsMenu);

        setMenuBar(menuBar);

        label = new JLabel();
        this.setTextLabel("<html><h2>Welcome!</h2></html>");
        this.add(label);
    }

    private void setTextLabel(String text) {
        label.setText(text);
    }

    // Action depending on the user's choice from the menu
    @SuppressWarnings("deprecation")
    @Override
    public boolean action(Event event, Object obj) {
        if (event.target instanceof MenuItem) {
            String choice = (String) obj;
            switch (choice) {
                case "Home" -> {
                    this.setTextLabel(this.HOME_PAGE);
                    break;
                }
                case "About" -> {
                    this.setTextLabel(this.ABOUT_PAGE);
                    break;
                }
                case "ionosphere" -> {
                    this.setTextLabel(this.LOAD_TRAIN_DATASET);

                    FileOperations fileOperations = new FileOperations();
                    if (!fileOperations.loadDataset("ionosphere.train")) {
                        this.loadedDatasets = false;
                        this.setTextLabel(this.SOMETHING_WRONG);
                    } else {
                        if (!fileOperations.loadDataset("ionosphere.test")) {
                            this.loadedDatasets = false;
                            this.setTextLabel(this.SOMETHING_WRONG);
                        } else {
                            this.loadedDatasets = true;
                            this.setTextLabel(this.SUCCESS_LOAD_IONOSPHERE_DATASETS);
                        }
                    }

                    break;
                }
                case "wine" -> {
                    this.setTextLabel(this.LOAD_TRAIN_DATASET);

                    FileOperations fileOperations = new FileOperations();
                    if (!fileOperations.loadDataset("wine.train")) {
                        this.loadedDatasets = false;
                        this.setTextLabel(this.SOMETHING_WRONG);
                    } else {
                        if (!fileOperations.loadDataset("wine.test")) {
                            this.loadedDatasets = false;
                            this.setTextLabel(this.SOMETHING_WRONG);
                        } else {
                            this.loadedDatasets = true;
                            this.setTextLabel(this.SUCCESS_LOAD_WINE_DATASETS);
                        }
                    }

                    break;
                }
                case "iris" -> {
                    this.setTextLabel(this.LOAD_TRAIN_DATASET);

                    FileOperations fileOperations = new FileOperations();
                    if (!fileOperations.loadDataset("iris.train")) {
                        this.loadedDatasets = false;
                        this.setTextLabel(this.SOMETHING_WRONG);
                    } else {
                        if (!fileOperations.loadDataset("iris.test")) {
                            this.loadedDatasets = false;
                            this.setTextLabel(this.SOMETHING_WRONG);
                        } else {
                            this.loadedDatasets = true;
                            this.setTextLabel(this.SUCCESS_LOAD_IRIS_DATASETS);
                        }
                    }

                    break;
                }
                case "MLP with Back Propagation" -> {
                    if (!loadedDatasets)
                        this.setTextLabel(this.NEED_DATASETS);
                    else {
                        MLP mlp = new MLP();
                        Double trainError = mlp.train(false, null);
                        Double testError = mlp.getTestError();
                        String testErrorDisplay = (testError == -1.0) ? "No test dataset" : testError.toString();
                        this.setTextLabel(
                                "<html>"
                                        + "<h2>"
                                        + "<br/><br/><br/><br/><br/>"
                                        + "<table style='font: 16px Arial, Helvetica, sans-serif;'>"
                                        + "<tr>"
                                        + "<td>"
                                        + " Train Error: "
                                        + "</td>"
                                        + "<td>"
                                        + trainError
                                        + "<td/>"
                                        + "<tr/>"
                                        + "<tr>"
                                        + "<td>"
                                        + " Test Error: "
                                        + "</td>"
                                        + "<td>"
                                        + testErrorDisplay
                                        + "<td/>"
                                        + "<tr/>"
                                        + "</table>"
                                        + "</h2>"
                                        + "</html>");
                    }
                    break;
                }
                case "MLP with Back Propagation (Using Genetic Algorithm) - Single point crossover" -> {
                    if (!loadedDatasets)
                        this.setTextLabel(this.NEED_DATASETS);
                    else {
                        MLP mlp = new MLP();
                        Double trainError = mlp.train(true, GENETIC_CROSSOVER_OPTIONS.SINGLE);
                        Double testError = mlp.getTestError();
                        String testErrorDisplay = (testError == -1.0) ? "No test dataset" : testError.toString();
                        this.setTextLabel(
                                "<html>"
                                        + "<h2>"
                                        + "<br/><br/><br/><br/><br/>"
                                        + "<table style='font: 16px Arial, Helvetica, sans-serif;;'>"
                                        + "<tr>"
                                        + "<td>"
                                        + " Train Error: "
                                        + "</td>"
                                        + "<td>"
                                        + trainError
                                        + "<td/>"
                                        + "<tr/>"
                                        + "<tr>"
                                        + "<td>"
                                        + " Test Error: "
                                        + "</td>"
                                        + "<td>"
                                        + testErrorDisplay
                                        + "<td/>"
                                        + "<tr/>"
                                        + "</table>"
                                        + "</h2>"
                                        + "</html>");
                    }
                    break;
                }
                case "MLP with Back Propagation (Using Genetic Algorithm) - Double point crossover" -> {
                    if (!loadedDatasets)
                        this.setTextLabel(this.NEED_DATASETS);
                    else {
                        MLP mlp = new MLP();
                        Double trainError = mlp.train(true, GENETIC_CROSSOVER_OPTIONS.DOUBLE);
                        Double testError = mlp.getTestError();
                        String testErrorDisplay = (testError == -1.0) ? "No test dataset" : testError.toString();
                        this.setTextLabel(
                                "<html>"
                                        + "<h2>"
                                        + "<br/><br/><br/><br/><br/>"
                                        + "<table style='font: 16px Arial, Helvetica, sans-serif;;'>"
                                        + "<tr>"
                                        + "<td>"
                                        + " Train Error: "
                                        + "</td>"
                                        + "<td>"
                                        + trainError
                                        + "<td/>"
                                        + "<tr/>"
                                        + "<tr>"
                                        + "<td>"
                                        + " Test Error: "
                                        + "</td>"
                                        + "<td>"
                                        + testErrorDisplay
                                        + "<td/>"
                                        + "<tr/>"
                                        + "</table>"
                                        + "</h2>"
                                        + "</html>");
                    }
                    break;
                }
                case "Load train dataset" -> {
                    this.setTextLabel(this.LOAD_CUSTOM_TRAIN_DATASET);
                    FileOperations fileOperations = new FileOperations();
                    if (fileOperations.chooseDatasetToLoad()) {
                        this.loadedDatasets = true;
                        this.setTextLabel(this.SUCCESS_LOAD_CUSTOM_DATASETS);
                    } else {
                        this.loadedDatasets = false;
                        this.setTextLabel(this.SOMETHING_WRONG);
                    }

                    break;
                }
                case "Load test dataset" -> {
                    if (this.loadedDatasets) {
                        this.setTextLabel(this.LOAD_CUSTOM_TEST_DATASET);
                        FileOperations fileOperations = new FileOperations();
                        if (fileOperations.chooseDatasetToLoad()) {
                            this.setTextLabel(this.SUCCESS_LOAD_CUSTOM_DATASETS);
                        } else {
                            this.setTextLabel(this.SOMETHING_WRONG);
                        }
                    } else {
                        this.setTextLabel(this.NEED_TRAIN_DATASETS);
                    }
                    break;
                }
            }
        } else
            super.action(event, obj);

        return true;
    }
}
