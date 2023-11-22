package simulator;

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
    public double survivalRate = 0;
    public int numOfDeadBlobs = 0;
    public int numOfAliveBlobs = 0;
    public int numbOfFoodLeft = 0;
    public int numOfInputSensors = 16;

    // good start for number of inner neurons is number of input neurons + number of output neurons / 2
    public int numOfHiddenNeurons = 13;
    public int numOfOutputNeurons = 10;

 
    // SETTINGS FOR THE SIMULATION

    // size of the blobs
    int blobSize = 10;

    // size of the map
    int mapSize = 800;

    // amount of simulations each generationcvf
    public int maxSimulationSteps = 70;

    // amouunt of total generations
    public int maxGenerations = 100000;

    // max amount of blobs at every generation
    public int maxNumOfBlobs = 40;

    // amount of food for each generation
    public int foodAmount = 200;

    // initial blob amount
    public int blobAmount = 40;

    // reproduction of clone amount of original blob
    public int cloneSpawnAmount = 2;

    // 10 percent chance that each connection will be changed to a different weight
    public double mutationRate = 0.1;

    // how much the mutation will change the weight by (will be negative or positive)
    public int mutationChangeAmount = 30;

    // sensing range of the blobs
    public int sensingRange = 1000;


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
        Blob.sensingRange = sensingRange;
        Blob.blobSize = blobSize;
        BlobNeuralNetwork.numInputs = numOfInputSensors;
        BlobNeuralNetwork.numHiddenNeurons = numOfHiddenNeurons;
        BlobNeuralNetwork.numOutputs = numOfOutputNeurons;

        
    
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
    }

    // updating simulation for each step
    private void updateSimulation() {
        updateBlobs();

        // testing adds a blob if there are no blobs left in the simulation allowing for continual training even if all blobs die
        if (blobs.size() <= 0){
            BlobNeuralNetwork randomNetwork = new BlobNeuralNetwork(numOfInputSensors, numOfHiddenNeurons, numOfOutputNeurons);
            Blob blob = new Blob(new Point(random.nextInt(mapSize), random.nextInt(mapSize)), randomNetwork);
            blobs.add(blob);

            blobNetwork = randomNetwork;
        }
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

    // updating the blobs while also removing blobs so that the max amount of blobs is not exceeded
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
        Point point = new Point(random.nextInt(mapSize), random.nextInt(mapSize));
    
        // Create new blobs for the next generation
        for (Blob blob : blobs) {
            if (blob.getEatenAmount() > 0) {
                numOfAliveBlobs++;
                // creating the original blob and putting it back into the next generation
                blob.eatenAmount = 0;
                blob.position = point;

                // if the blob has eaten, spawn clones of the original blob with the amount of food it has eaten
                for (int i = 0; i < blob.eatenAmount + 1; i++){
                    point = new Point(random.nextInt(mapSize), random.nextInt(mapSize));
                    blob.position = point;
                    newBlobs.add(blob);
                }
                // if the blob survives spawn clones with mutations
                for (int i = 0; i < cloneSpawnAmount; i++){
                    Blob newBlob = cloneBlobWithMutation(blob);
                    newBlobs.add(newBlob);
                }
            }
        }
        // Clear existing blobs and add the new ones
        numOfDeadBlobs = numOfStartingBlobs - numOfAliveBlobs;
        survivalRate = (double)numOfAliveBlobs / (double)numOfStartingBlobs;
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
        clonedBlob.eatenAmount = 0;
        
        return clonedBlob;
    }

    private void applyMutation(INDArray weights) {
        // Add small random values to the weights

        mutationChangeAmount *= random.nextDouble(minRange, maxRange);
        weights.addi(Nd4j.rand(weights.shape()).subi(mutationChangeAmount).muli(mutationRate));
    }

    public BlobNeuralNetwork crossOver(BlobNeuralNetwork male, BlobNeuralNetwork female){
        return null;
    }

    private void createNewFood(){

        foods.clear();
    
        // adding foods (can also change this to be the condition for blob to survive)
        for (int i = 0; i < foodAmount; i++) {
            foods.add(new Food(new Point(random.nextInt(mapSize), random.nextInt(mapSize - 50, mapSize))));
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
            writer.write("Survival Rate: " + survivalRate + "\n");
            writer.write("\n");

            // Reset statistics for the next generation
            numOfStartingBlobs = numOfAliveBlobs;
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
            if (blob.getEatenAmount() > 0) {
                g.setColor(Color.BLUE);
            } else {
                g.setColor(Color.GREEN);
            }
    
            // Draw the filled rectangle with the blob color
            g.fillRect(blob.position.x, blob.position.y, blobSize, blobSize);

            // Draw the outline of the blob
            g.setColor(Color.BLACK);
            g.drawRect(blob.position.x, blob.position.y, blobSize, blobSize);
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
