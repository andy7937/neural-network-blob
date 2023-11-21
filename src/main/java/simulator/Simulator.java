package simulator;

import java.util.Iterator;
import java.awt.Color;
import java.awt.Graphics;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;
import organisms.Blob;
import organisms.Food;
import neuralNetwork.BlobNeuralNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Simulator extends JPanel {

    public List<Food> foods = new ArrayList<>();
    public List<Blob> blobs = new ArrayList<>();
    public int simulationStep = 0;
    public int currentGeneration = 0;
    public int numOfStartingBlobs = 0;
    public int numOfDeadBlobs = 0;
    public int numOfAliveBlobs = 0;
    public int numbOfFoodLeft = 0;
    public int numOfInputSensors = 15;
    public int numOfHiddenNeurons = 20;
    public int numOfOutputNeurons = 6;

 
    // SETTINGS FOR THE SIMULATION

    // size of the map
    int mapSize = 800;

    // amount of simulations each generation
    public int maxSimulationSteps = 200;

    // amouunt of total generations
    public int maxGenerations = 1000;

    // max amount of blobs at every generation
    public int maxNumOfBlobs = 30;

    // amount of food for each generation
    public int foodAmount = 200;

    // initial blob amount
    public int blobAmount = 30;

    // reproduction clone amount
    public int spawnAmount = 2;

    // 10 percent chance that each connection will be changed to a different weight
    public double mutationRate = 0.1;

    // how much the mutation will change the weight by (will be negative or positive)
    public int mutationChangeAmount = 10;

    // how random the mutation will be. Min range will be how low it can be multipled by decreasing the mutation amount
    // Max range will be how high it can be multipled by increasing the mutation amount. For example, if minRange is 0 and
    // maxRange is 2, the mutation amount will be between 0 and double the mutation amount. Keep both at 1 if you don't 
    // want to change the mutation amount
    public double minRange = 0.99;
    public double maxRange = 1;

    private static Simulator instance;
    private BlobNeuralNetwork blobNetwork;
    Random random = new Random(System.currentTimeMillis());


    private Simulator() {
        initializeSimulation();
    }

    public static Simulator getInstance() {
        if (instance == null) {
            instance = new Simulator();
        }
        return instance;
    }

    private void initializeSimulation() {
        createNewFood();
        clearStatistics();
        numOfStartingBlobs = blobAmount;
        numbOfFoodLeft = foodAmount;

        Blob.mapSize = mapSize;
        Blob.numOfInputSensors = numOfInputSensors;
        Blob.numOfHiddenNeurons = numOfHiddenNeurons;
        Blob.numOfOutputNeurons = numOfOutputNeurons;
        
    
        // Create blobs with associated neural networks
        for (int i = 0; i < blobAmount; i++) {
            BlobNeuralNetwork randomNetwork = new BlobNeuralNetwork(numOfInputSensors, numOfHiddenNeurons, numOfOutputNeurons);
            Blob blob = new Blob(new Point(random.nextInt(mapSize), random.nextInt(mapSize)), randomNetwork);
    
            blobs.add(blob);
    
            // Set the first neural network as the main blobNetwork
            if (i == 0) {
                blobNetwork = randomNetwork;
            }
        }
    
        runSimulation();
    }

    // updating simulation for each step
    private void updateSimulation() {
        updateBlobs();

        simulationStep++;
        System.out.println("Simulation step: " + simulationStep);

        if (simulationStep >= maxSimulationSteps) {
            // At the end of each generation
            numbOfFoodLeft = foods.size();
            createNewFood();
            if (currentGeneration < maxGenerations - 1) {
                // Save the trained model for the current generation
                blobNetwork.saveModel("trained_blob_model_generation_" + currentGeneration + ".zip");
                System.out.println("Generation step: " + currentGeneration);
                
                // Create a new generation of blobs
                createNewGeneration();
                
                // Reset simulation step
                simulationStep = 0;
                saveStatistics();
                currentGeneration++;
            } else {
                // Save the final model and terminate the simulation
                saveStatistics();
                blobNetwork.saveModel("final_trained_blob_model.zip");
                System.exit(0);
            }
        }
    }

    private void updateBlobs() {
        // Collect indices to remove
        List<Integer> indicesToRemove = new ArrayList<>();
    
        // Identify blobs to remove
        for (int i = maxNumOfBlobs; i < blobs.size(); i++) {
            indicesToRemove.add(random.nextInt(blobs.size()));
        }
    
        // Remove marked blobs
        List<Blob> blobsToRemove = new ArrayList<>();
        for (int indexToRemove : indicesToRemove) {
            blobsToRemove.add(blobs.get(indexToRemove));
        }
        blobs.removeAll(blobsToRemove);
    
        // Update remaining blobs
        for (Blob blob : blobs) {
            blob.update(foods, blobs);
        }
    }

    private void createNewGeneration() {
        // Create a new list for the next generation
        List<Blob> newBlobs = new ArrayList<>();

        // create new points
        Point point = new Point(random.nextInt(), random.nextInt());
    
        // Create new blobs for the next generation
        for (Blob blob : blobs) {
            if (blob.hasEaten()) {
                numOfAliveBlobs++;
                // creating the original blob and putting it back into the next generation
                blob.hasEaten = false;
                blob.position = point;
                newBlobs.add(blob);

                // if the blob survives spawn 2 clones with mutations
                for (int i = 0; i < spawnAmount; i++){
                Blob newBlob = cloneBlobWithMutation(blob);
                newBlobs.add(newBlob);
                }
            }
            else{
                numOfDeadBlobs++;
            }
        }
        numOfStartingBlobs = numOfAliveBlobs;
        // Clear existing blobs and add the new ones
        blobs.clear();
        blobs.addAll(newBlobs);
    }

    private Blob cloneBlobWithMutation(Blob originalBlob) {
        // Create a new blob as a clone of the original
        Point Point = new Point(random.nextInt(mapSize), random.nextInt(mapSize));
        Blob clonedBlob = new Blob(Point, null);
        
        // Clone the neural network
        BlobNeuralNetwork originalNN = originalBlob.neuralNetwork;
        BlobNeuralNetwork clonedNN = originalNN.clone(); // Use the clone method of BlobNeuralNetwork
        
        // Introduce mutation to the weights (adjust this based on your requirements)
        applyMutation(clonedNN.getOutputWeights());
        
        // Set the cloned neural network to the new blob
        clonedBlob.neuralNetwork = clonedNN;
        clonedBlob.hasEaten = false;
        
        return clonedBlob;
    }

    private void applyMutation(INDArray weights) {
        // Add small random values to the weights

        mutationChangeAmount *= random.nextDouble(minRange, maxRange);
        weights.addi(Nd4j.rand(weights.shape()).subi(mutationChangeAmount).muli(mutationRate));
    }

    private void createNewFood(){

        foods.clear();
    
        // adding foods
        for (int i = 0; i < foodAmount; i++) {
            foods.add(new Food(new Point(random.nextInt(mapSize), random.nextInt(mapSize))));
        }
    }

    // save the statistics of the simulation
    private void saveStatistics() {
        String filename = "generation_statistics.txt";

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename, true))) {
            writer.write("Generation: " + currentGeneration + "\n");
            writer.write("Num of Starting Blobs: " + numOfStartingBlobs + "\n");
            writer.write("Num of Dead Blobs: " + numOfDeadBlobs + "\n");
            writer.write("Num of Alive Blobs: " + numOfAliveBlobs + "\n");
            writer.write("Num of Food Left: " + numbOfFoodLeft + "\n");
            writer.write("\n");

            // Reset statistics for the next generation
            numOfDeadBlobs = 0;
            numOfAliveBlobs = 0;

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // clear the statistics of the simulation
    private void clearStatistics() {
        String filename = "generation_statistics.txt";
    
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // The file is opened without the 'true' parameter, so it will be cleared
    
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    // drawing the simulation
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        drawFoods(g);
        drawBlobs(g);
    }

    private void drawBlobs(Graphics g) {
        // drawing blobs
        for (Blob blob : blobs) {

            if (blob.hasEaten() == true){
                g.setColor(Color.BLUE);
            }
            else{
                g.setColor(Color.GREEN);
            }
            g.fillRect(blob.position.x, blob.position.y, 10, 10);
        }
    }

    private void drawFoods(Graphics g) {
        // drawing foods
        for (Food food : foods) {
            g.setColor(Color.RED);
            g.fillRect(food.position.x, food.position.y, 5, 5);
        }
    }

    public void runSimulation() {
        JFrame frame = new JFrame("Blob Simulator");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(mapSize, mapSize);
        frame.getContentPane().add(this);
        frame.setVisible(true);

        Timer timer = new Timer(10, e -> {
            updateSimulation();
            repaint();
        });

        timer.start();
    }
}
