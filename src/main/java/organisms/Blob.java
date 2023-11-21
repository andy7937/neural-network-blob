package organisms;
import neuralNetwork.BlobNeuralNetwork;
import simulator.Point;

public class Blob {

    public Point position;
    public BlobNeuralNetwork neuralNetwork;

    public Blob(Point position) {
        this.position = position;
    }
    
}
