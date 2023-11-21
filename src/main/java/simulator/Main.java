package simulator;

import javax.swing.SwingUtilities;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;


public class Main {
    public static void main(String[] args) {


        SwingUtilities.invokeLater(() -> {
            Simulator simulator = Simulator.getInstance();
            simulator.runSimulation();
        });


        // UNCOMMENT THIS TO LOAD MODEL CURRENTLY IN THE FOLDER MODEL.
        // IF YOU WANT TO SAVE THE MODEL, YOU NEED TO CHANGE THE FOLDER NAME TO SOMETHING ELSE.


        // for (int i = 0; i < 30; i++) {
        //     // Construct the model path based on your naming convention
        //     String modelPath = "trained_blob_model_generation_" + i + ".zip";
            
        //     // Load the model using the ModelLoader class
        //     MultiLayerNetwork model = ModelLoader.loadModel(modelPath);

        //     // Use the loaded model as needed
        //     if (model != null) {
        //         // Do something with the loaded model
        //         System.out.println("Loaded model for generation " + i);
        //         System.out.println(model.summary());

        //     } else {
        //         System.out.println("Failed to load model for generation " + i);
        //     }
        // }

        
    }
}