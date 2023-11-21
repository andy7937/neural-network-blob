package simulator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import java.io.File;
import java.io.IOException;

public class ModelLoader {

    public static MultiLayerNetwork loadModel(String path) {
        String modelsFolderPath = "models/";
        String fullPath = modelsFolderPath + path;

        try {
            // Load the model from the "models" folder
            return MultiLayerNetwork.load(new File(fullPath), true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null; // Return null if there's an error loading the model
    }

}
