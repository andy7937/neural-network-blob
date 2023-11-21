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

public class Simulator extends JPanel {
    int mapSize = 1080;
    public List<Food> foods = new ArrayList<>();
    public List<Blob> blobs = new ArrayList<>();
    public int simulationStep = 0;
    public int maxSimulationSteps = 1000;
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
        for (int i = 0; i < 500; i++) {
            foods.add(new Food(new Point(random.nextInt(mapSize), random.nextInt(mapSize))));
        }

        // Create a neural network for each blob
        blobNetwork = new BlobNeuralNetwork(4, 10, 4);

        // Create blobs with associated neural networks
        for (int i = 0; i < 10; i++) {
            Blob blob = new Blob(new Point(random.nextInt(mapSize), random.nextInt(mapSize)), blobNetwork.copy());
            blobs.add(blob);
        }

        runSimulation();
    }

    // updating simulation for each step
    private void updateSimulation() {
        updateBlobs();
        simulationStep++;

        if (simulationStep >= maxSimulationSteps) {
            // Save the trained model at the end of the simulation
            blobNetwork.saveModel("trained_blob_model.zip");
            System.exit(0); // Terminate the simulation when done
        }
    }

    private void updateBlobs() {
        // Update blob positions, sensor data, etc.

        for (Blob blob : blobs) {
            // Get input data based on the blob's state
            INDArray input = 5;/* generate input data based on blob's state */;
            // Generate target data based on the desired behavior (not implemented in this example)
            INDArray target = 5;/* generate target data based on desired behavior */
            // Train the neural network with the current simulation step
            blob.neuralNetwork.trainStep(input, target);
        }

        // Continue simulation...
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
