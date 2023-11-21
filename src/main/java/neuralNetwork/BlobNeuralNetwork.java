package neuralNetwork;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class BlobNeuralNetwork {

    private MultiLayerNetwork model;

    public BlobNeuralNetwork(int numInputs, int numHiddenNeurons, int numOutputs) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.1))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNeurons)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(numHiddenNeurons)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        this.model = new MultiLayerNetwork(conf);
        this.model.init();
    }

    // Train the neural network with a single step of simulation data
    public void trainStep(INDArray input, INDArray target) {
        model.fit(input, target);
    }

    // Predict based on the current state of the neural network
    public INDArray predict(INDArray input) {
        return model.output(input);
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
