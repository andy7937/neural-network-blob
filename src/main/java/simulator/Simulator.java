package simulator;

import java.awt.Color;
import java.awt.Graphics;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;
import organisms.Blob;
import organisms.Food;
import neuralNetwork.BlobNeuralNetwork;
import simulator.Point;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Simulator extends JPanel {
    int mapSize = 1080;
    public List<Food> foods = new ArrayList<>();
    public List<Blob> blobs = new ArrayList<>();
    public int simulationStep = 0;
    public int maxSimulationSteps = 100;
    public int currentGeneration = 0;
    public int maxGenerations = 100;
    private static Simulator instance;
    private BlobNeuralNetwork blobNetwork;

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
        Random random = new Random();

        // adding foods
        for (int i = 0; i < 100; i++) {
            foods.add(new Food(new Point(random.nextInt(mapSize), random.nextInt(mapSize))));
        }

        // Create blobs with associated neural networks
        for (int i = 0; i < 20; i++) {
            BlobNeuralNetwork randomNetwork = new BlobNeuralNetwork(15, 4, 5);
            Blob blob = new Blob(new Point(random.nextInt(mapSize), random.nextInt(mapSize)), randomNetwork);
    
            blobs.add(blob);
        }

        runSimulation();
    }

    // updating simulation for each step
    private void updateSimulation() {
        updateBlobs();

        simulationStep++;
        System.out.println("Simulation step: " + simulationStep);
        for (Blob blob: blobs) {
            System.out.println("Initial Weights: " + blob.neuralNetwork.getOutputWeights());        }

        if (simulationStep >= maxSimulationSteps) {
            // At the end of each generation
            if (currentGeneration < maxGenerations - 1) {
                // Save the trained model for the current generation
                blobNetwork.saveModel("trained_blob_model_generation_" + currentGeneration + ".zip");
                
                // Create a new generation of blobs
                createNewGeneration();
                
                // Reset simulation step
                simulationStep = 0;
                currentGeneration++;
            } else {
                // Save the final model and terminate the simulation
                blobNetwork.saveModel("final_trained_blob_model.zip");
                System.exit(0);
            }
        }
    }

    private void updateBlobs() {
        // Update blob positions, sensor data, etc.
        for (Blob blob : blobs) {
            blob.update(foods, blobs);
        }
    }

    private void createNewGeneration() {
        // Create a new list for the next generation
        List<Blob> newBlobs = new ArrayList<>();
        
        // Create new blobs for the next generation
        for (Blob blob : blobs) {
            if (blob.hasEaten()) {

                Blob newBlob = cloneBlobWithMutation(blob);
                newBlobs.add(newBlob);

            } else {
                blobs.remove(blob);
            }
        }
        
        // Clear existing blobs and add the new ones
        blobs.clear();
        blobs.addAll(newBlobs);
    }

    private Blob cloneBlobWithMutation(Blob originalBlob) {
        // Create a new blob as a clone of the original
        Random random = new Random();
        Point Point = new Point(random.nextInt(mapSize), random.nextInt(mapSize));
        Blob clonedBlob = new Blob(Point, null);
        
        // Clone the neural network
        BlobNeuralNetwork originalNN = originalBlob.neuralNetwork;
        BlobNeuralNetwork clonedNN = originalNN.clone(); // Use the clone method of BlobNeuralNetwork
        
        // Introduce mutation to the weights (adjust this based on your requirements)
        applyMutation(clonedNN.getOutputWeights());
        
        // Set the cloned neural network to the new blob
        clonedBlob.neuralNetwork = clonedNN;
        
        return clonedBlob;
    }

    private void applyMutation(INDArray weights) {
        // Adjust this mutation rate based on your requirements
        double mutationRate = 0.1;
    
        // Add small random values to the weights
        weights.addi(Nd4j.rand(weights.shape()).subi(0.5).muli(mutationRate));
    }

    
    

    // drawing the simulation
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        drawBlobs(g);
        drawFoods(g);
    }

    private void drawBlobs(Graphics g) {
        // drawing blobs
        for (Blob blob : blobs) {
            g.setColor(Color.GREEN);
            g.fillRect(blob.position.x, blob.position.y, 20, 20);
        }
    }

    private void drawFoods(Graphics g) {
        // drawing foods
        for (Food food : foods) {
            g.setColor(Color.RED);
            g.fillRect(food.position.x, food.position.y, 10, 10);
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
