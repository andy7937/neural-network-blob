package neuralNetwork;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BlobNeuralNetwork {

    private MultiLayerNetwork model;

    public BlobNeuralNetwork(int numInputs, int numHiddenNeurons, int numOutputs) {

        Random random = new Random(System.currentTimeMillis());
        int i = random.nextInt(100000000);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123 + i)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.1))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNeurons)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNeurons)
                        .nOut(numOutputs)
                        .build())
                .build();
    
        this.model = new MultiLayerNetwork(conf);
        this.model.init();
    }
    // Train the neural network with a single step of simulation data
    public INDArray trainStep(INDArray input, INDArray target) {
        model.fit(input, target);
        // Assuming the output of the neural network is needed after training
        return model.output(input);
    }

    // Predict based on the current state of the neural network
    public INDArray predict(INDArray input) {
        return model.output(input);
    }

    // Get the weights of the output layer
    public INDArray getOutputWeights() {
        return model.getLayer(1).getParam("W");
    }

    public BlobNeuralNetwork clone() {
        try {
            // Clone the model to create an independent copy
            MultiLayerNetwork clonedModel = model.clone();

            // Create a new BlobNeuralNetwork and set the cloned model
            BlobNeuralNetwork clonedNetwork = new BlobNeuralNetwork(15, 4, 6); // Adjust the parameters as needed
            clonedNetwork.setModel(clonedModel);

            return clonedNetwork;
        } catch (Exception e) {
            e.printStackTrace();
            return null; // Handle the exception appropriately in your application
        }
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

    // Save and load methods remain the same

    public void saveModel(String path) {
        try {
            model.save(new File(path), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadModel(String path) {
        try {
            model = MultiLayerNetwork.load(new File(path), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
