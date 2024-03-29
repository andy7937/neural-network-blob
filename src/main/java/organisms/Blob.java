package organisms;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import neuralNetwork.BlobNeuralNetwork;
import simulator.Point;
import simulator.Simulator;

public class Blob {

    public Point position;
    public BlobNeuralNetwork neuralNetwork;
    public static int mapSize;
    public int eatenAmount = 0;
    public static int numOfInputSensors;
    public static int numOfHiddenNeurons;
    public static int numOfOutputNeurons;
    public static int sensingRange;
    public static int blobSize;
    public int lastActionTaken = 10;
    public Random random = new Random(System.currentTimeMillis());


    public Blob(Point position, BlobNeuralNetwork neuralNetwork) {
        this.position = position;
        this.neuralNetwork = neuralNetwork;
    }

    public void update(List<Food> foods, List<Blob> blobs) {
        for (Blob blob : blobs) {
            // Get input data based on the blob's state
            INDArray input = blob.generateInputVector(foods, blobs);

            // Generate target data based on the desired behavior (not implemented in this example)
            INDArray target = blob.generateOutputVector(input);

            blob.updateBlobPosition(target);
        }



        checkForFoodEating(foods);
    }

    private void updateBlobPosition(INDArray output) {
        // Assuming the neural network has 6 outputs:
        // 0. Move left
        // 1. Move right
        // 2. Move up
        // 3. Move down
        // 4. Move top left
        // 5. Move top right
        // 6. Move bottom left
        // 7. Move bottom right
        // 8. Eat everything adjacent
        // 9. Random Movement    
        // 10. Do nothing
        int numOutputs = output.columns();
    
        // Find the index with the maximum value in the output vector
        int maxIndex = 0;
        double maxValue = output.getDouble(0);
        for (int i = 1; i < numOutputs; i++) {
            double value = output.getDouble(i);
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            }
        }
        lastActionTaken = maxIndex;
        // Update the blob's position based on the identified action or direction
        if (maxIndex >= 0 && maxIndex < 8){
            moveInAdjacentDirection(maxIndex);
        }

    switch (maxIndex) {
        case 8:
            // Eat everything adjacent, meaning remove all foods adjacent to the blob
            checkForFoodEating(Simulator.getInstance().foods);
            break;
        case 9:
            // Random Movement
            int randomDirection = random.nextInt(8);
            moveInAdjacentDirection(randomDirection);
            break;
        }

        // make sure the blob stays within the map which is 1080x1080
        if (position.x < 0) {
            position.x = 0;
        } else if (position.x > mapSize) {
            position.x = mapSize;
        }

        if (position.y < 0) {
            position.y = 0;
        } else if (position.y > mapSize) {
            position.y = mapSize;
        }

    }

    public INDArray generateOutputVector(INDArray input) {
        // Get the raw output from the neural network
        INDArray rawOutput = neuralNetwork.predict(input);


        // Find the index with the maximum value in the softmax output vector
        int maxIndex = Nd4j.argMax(rawOutput, 1).getInt(0);
    
        // Create a one-hot encoded vector for the selected action
        INDArray outputVector = Nd4j.zeros(1, numOfOutputNeurons);
        outputVector.putScalar(maxIndex, 1.0);
    
        return outputVector;
    }

    public INDArray generateInputVector(List<Food> foods, List<Blob> blobs) {
        // Assuming the neural network has 14 inputs:
        // 0. Food on the left in a radius of 50
        // 1. Food on the right in a radius of 50
        // 2. Food on the Top in a radius of 50
        // 3. Food on the Bottom in a radius of 50
        // 4. Blob on the left in a radius of 50
        // 5. Blob on the right in a radius of 50
        // 6. Blob on the Top in a radius of 50
        // 7. Blob on the Bottom in a radius of 50
        // 8. Distance from North Border
        // 9. Distance from South Border
        // 10. Distance from East Border
        // 11. Distance from West Border
        // 12. Distance from nearest food
        // 13. Distance from nearest blob
        // 14. Food adjacent to the blob
        // 15. Blob adjacent to the blob
        // 16. Density of food in top right
        // 17. Density of food in top left
        // 18. Density of food in bottom right
        // 19. Density of food in bottom left
        // 20. Density of blob in top right
        // 21. Density of blob in top left
        // 22. Density of blob in bottom right
        // 23. Density of blob in bottom left
        // 24. Last action taken


    
        // Initialize the input vector with zeros
        INDArray inputVector = Nd4j.zeros(1, numOfInputSensors);

        // Initialize the food density variables
        int trFoodDensity = 0;
        int tlFoodDensity = 0;
        int brFoodDensity = 0;
        int blFoodDensity = 0;

        // Initialize the blob density variables
        int trBlobDensity = 0;
        int tlBlobDensity = 0;
        int brBlobDensity = 0;
        int blBlobDensity = 0;

        // Calculate the normalization factor for the density
        double normalizationFactor = 1.0 / sensingRange;
    
        // Iterate over all foods and blobs to populate the input vector
        for (Food food : foods) {
            double distance = distance(position, food.position);
            if (distance <= sensingRange) {
                  double weight = calculateDynamicWeight(distance);
                // Food on the left
                if (food.position.x < position.x && (position.x - food.position.x) <= sensingRange) {
                    inputVector.putScalar(0, weight);
                }
                // Food on the right
                else if (food.position.x > position.x && (food.position.x - position.x) <= sensingRange) {
                    inputVector.putScalar(1, weight);
                }
                // Food on the top
                else if (food.position.y < position.y && (position.y - food.position.y) <= sensingRange) {
                    inputVector.putScalar(2, weight);
                }
                // Food on the bottom
                else if (food.position.y > position.y && (food.position.y - position.y) <= sensingRange) {
                    inputVector.putScalar(3, weight);
                }
    
                // Distance from nearest food
                double currentDistance = distance(position, food.position);
                double currentNearestDistance = inputVector.getDouble(12);
                if (currentDistance < currentNearestDistance || currentNearestDistance == 0) {
                    // Calculate dynamic weight based on the nearest food distance
                    double dynamicWeight = calculateDynamicWeight(currentDistance);

                    // Apply dynamic weight to the input vector at index 12
                    inputVector.putScalar(12, currentDistance * dynamicWeight);
                }

                // Calculating density of food in top right, top left, bottom right, bottom left
                // Top right
                if (food.position.x > position.x && food.position.y < position.y && food.position.x - position.x <= sensingRange && position.y - food.position.y <= sensingRange) {
                    trFoodDensity++;
                }
                // Top left
                if (food.position.x < position.x && food.position.y < position.y && position.x - food.position.x <= sensingRange && position.y - food.position.y <= sensingRange) {
                    tlFoodDensity++;
                }
                // Bottom right
                if (food.position.x > position.x && food.position.y > position.y && food.position.x - position.x <= sensingRange && food.position.y - position.y <= sensingRange) {
                    brFoodDensity++;
                }
                // Bottom left
                if (food.position.x < position.x && food.position.y > position.y && position.x - food.position.x <= sensingRange && food.position.y - position.y <= sensingRange) {
                    blFoodDensity++;
                }

                // Apply density to the input vector at index 16, 17, 18, 19
                inputVector.putScalar(16, trFoodDensity * normalizationFactor);
                inputVector.putScalar(17, tlFoodDensity * normalizationFactor);
                inputVector.putScalar(18, brFoodDensity * normalizationFactor);
                inputVector.putScalar(19, blFoodDensity * normalizationFactor);

            }

            // Food adjacent to the blob make sure it is radius of blob
            if (distance(position, food.position) <= blobSize) {
                inputVector.putScalar(14, 1.0);
            }

        }
    
        for (Blob blob : blobs) {
            double distance = distance(position, blob.position);
            if (!blob.equals(this) && distance <= sensingRange) {
                double weight = calculateDynamicWeight(distance);
                // Blob on the left
                if (blob.position.x < position.x && (position.x - blob.position.x) <= sensingRange) {
                    inputVector.putScalar(4, weight);
                }
                // Blob on the right
                else if (blob.position.x > position.x && (blob.position.x - position.x) <= sensingRange) {
                    inputVector.putScalar(5, weight);
                }
                // Blob on the top
                else if (blob.position.y < position.y && (position.y - blob.position.y) <= sensingRange) {
                    inputVector.putScalar(6, weight);
                }
                // Blob on the bottom
                else if (blob.position.y > position.y && (blob.position.y - position.y) <= sensingRange) {
                    inputVector.putScalar(7, weight);
                }
    
                // Distance from nearest blob
                double currentDistance = distance(position, blob.position);
                double currentNearestDistance = inputVector.getDouble(13);
                if (currentDistance < currentNearestDistance || currentNearestDistance == 0) {
                    currentDistance = calculateDynamicWeight(currentDistance);
                    inputVector.putScalar(13, currentDistance);
                }
                // Blob adjacent to the blob make sure it is radius of blob
                if (distance(position, blob.position) <= blobSize * 2) {
                    inputVector.putScalar(15, 1.0);
                }

                // Calculating density of blob in top right, top left, bottom right, bottom left
                // Top right
                if (blob.position.x > position.x && blob.position.y < position.y && blob.position.x - position.x <= sensingRange && position.y - blob.position.y <= sensingRange) {
                    trBlobDensity++;
                }
                // Top left
                if (blob.position.x < position.x && blob.position.y < position.y && position.x - blob.position.x <= sensingRange && position.y - blob.position.y <= sensingRange) {
                    tlBlobDensity++;
                }
                // Bottom right
                if (blob.position.x > position.x && blob.position.y > position.y && blob.position.x - position.x <= sensingRange && blob.position.y - position.y <= sensingRange) {
                    brBlobDensity++;
                }
                // Bottom left
                if (blob.position.x < position.x && blob.position.y > position.y && position.x - blob.position.x <= sensingRange && blob.position.y - position.y <= sensingRange) {
                    blBlobDensity++;
                }

                // Apply density to the input vector at index 20, 21, 22, 23
                inputVector.putScalar(20, trBlobDensity * normalizationFactor);
                inputVector.putScalar(21, tlBlobDensity * normalizationFactor);
                inputVector.putScalar(22, brBlobDensity * normalizationFactor);
                inputVector.putScalar(23, blBlobDensity * normalizationFactor);

            }

        }
    
        // Distance from Borders
        inputVector.putScalar(8, position.y); // Distance from North Border
        inputVector.putScalar(9, mapSize - position.y); // Distance from South Border
        inputVector.putScalar(10, mapSize - position.x); // Distance from East Border
        inputVector.putScalar(11, position.x); // Distance from West Border

        // Last action taken
        inputVector.putScalar(24, lastActionTaken);
    
        return inputVector;
    }

    private void checkForFoodEating(List<Food> foods) {
        Iterator<Food> iterator = foods.iterator();
        while (iterator.hasNext()) {
            Food food = iterator.next();
            if (isAdjacent(food.position, blobSize)) {
                eatenAmount++;
                iterator.remove(); // Use iterator to safely remove the food
            }
        }
    }

    private void moveInAdjacentDirection(int index){
         switch (index) {
        case 0:
            position.x -= 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.x += 1;
            }

            break;
        case 1:
            position.x += 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.x -= 1;
            }
            break;
        case 2:
            position.y -= 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.y += 1;
            }
            break;
        case 3:
            position.y += 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.y -= 1;
            }
            break;
        case 4:
            position.x -= 1;
            position.y -= 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.x += 1;
                position.y += 1;
            }
            break;
        case 5:
            position.x += 1;
            position.y -= 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.x -= 1;
                position.y += 1;
            }
            break;
        case 6:
            position.x -= 1;
            position.y += 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.x += 1;
                position.y -= 1;
            }
            break;
        case 7:
            position.x += 1;
            position.y += 1;
            if (checkForBlobCollision(Simulator.getInstance().blobs)){
                position.x -= 1;
                position.y -= 1;
            }
            break;
         }
        
    }

    private boolean checkForBlobCollision(List<Blob> blobs) {
        // Iterate through all blobs and check for collision with the current blob
        for (Blob otherBlob : blobs) {
            if (!otherBlob.equals(this) && distance(position, otherBlob.position) < 10) {
                // Collision detected
                return true;
            }
        }
        // No collision
        return false;
    }


       // Method to check if a position is adjacent to the blob
       private boolean isAdjacent(Point other, int distanceThreshold) {
        return distance(position, other) <= distanceThreshold;
    }

    private double calculateDynamicWeight(double distance) {
        // Parameters for the modified Gaussian function
        double sigma = 0.5; // Adjust this value to control sensitivity
        double mu = 0.0;   // Center of the Gaussian function should be position of the blob
    
        // Modified Gaussian function
        double exponent = -Math.pow(distance - mu, 2) / (2 * Math.pow(sigma, 2));
        double dynamicWeight = Math.exp(exponent);
    
        return dynamicWeight;
    }


    // Helper method to calculate the Euclidean distance between two points
    private double distance(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

      // Method to check whether the blob has eaten
      public int getEatenAmount() {
        return eatenAmount;
    }

    
}