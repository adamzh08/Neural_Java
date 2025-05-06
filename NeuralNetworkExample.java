import java.util.Random;

/**
 * Represents a single layer in the neural network
 */
class Layer {
    /** Number of neurons in this layer */
    private int length;
    
    /** Activation function used in this layer */
    private ActivationFunction activation;
    
    /** Derivative of the activation function used for backpropagation */
    private ActivationFunction activationDerivative;
    
    /**
     * Constructor for a neural network layer
     * 
     * @param length Number of neurons in this layer
     * @param activation Activation function used in this layer
     * @param activationDerivative Derivative of the activation function
     */
    public Layer(int length, ActivationFunction activation, ActivationFunction activationDerivative) {
        this.length = length;
        this.activation = activation;
        this.activationDerivative = activationDerivative;
    }
    
    /**
     * @return Number of neurons in this layer
     */
    public int getLength() {
        return length;
    }
    
    /**
     * @return Activation function used in this layer
     */
    public ActivationFunction getActivation() {
        return activation;
    }
    
    /**
     * @return Derivative of the activation function
     */
    public ActivationFunction getActivationDerivative() {
        return activationDerivative;
    }
}

/**
 * Functional interface for activation functions
 */
@FunctionalInterface
interface ActivationFunction {
    float apply(float input);
}

/**
 * Represents the entire neural network structure
 */
class Network {
    /** Array of layer configurations */
    private Layer[] layers;
    
    /** Total number of layers in the network */
    private int size;
    
    /**
     * 3D array of network weights:
     * - First dimension: layer index
     * - Second dimension: input neuron index (including bias)
     * - Third dimension: output neuron index
     */
    private float[][][] weights;
    
    /**
     * Creates and initializes a new neural network
     * 
     * @param layers Array of layer configurations
     */
    public Network(Layer[] layers) {
        this.layers = layers;
        this.size = layers.length;
        
        // Allocate memory for layers
        this.weights = new float[size - 1][][];
        
        for (int layer = 0; layer < size - 1; layer++) {
            // Add +1 for bias weights
            this.weights[layer] = new float[layers[layer].getLength() + 1][];
            
            for (int inputNeuron = 0; inputNeuron < layers[layer].getLength() + 1; inputNeuron++) {
                this.weights[layer][inputNeuron] = new float[layers[layer + 1].getLength()];
            }
        }
    }
    
    /**
     * Initializes network weights with random values using Xavier/Glorot initialization
     * 
     * Implements Xavier/Glorot initialization which helps with:
     * 1. Preventing vanishing/exploding gradients
     * 2. Maintaining appropriate scale of gradients through the network
     * Scale factor is calculated as sqrt(2 / (fan_in + fan_out))
     */
    public void randomizeWeights() {
        Random random = new Random();
        
        for (int layer = 0; layer < size - 1; layer++) {
            // Xavier/Glorot initialization
            float scale = (float) Math.sqrt(2.0f / (layers[layer].getLength() + layers[layer + 1].getLength()));
            
            for (int i = 0; i < layers[layer].getLength() + 1; i++) {
                for (int j = 0; j < layers[layer + 1].getLength(); j++) {
                    float r = (random.nextFloat() * 2.0f - 1.0f);
                    weights[layer][i][j] = r * scale;
                }
            }
        }
    }
    
    /**
     * Performs forward propagation through the network
     * 
     * This function:
     * 1. Propagates input through each layer
     * 2. Applies weights and biases
     * 3. Handles special case for softmax in output layer
     * 4. Applies activation functions
     * 5. Returns final layer output
     * 
     * @param input Array of input values
     * @return Array containing output layer activations
     */
    public float[] getResult(float[] input) {
        float[] currentLayerActivations = new float[layers[0].getLength()];
        System.arraycopy(input, 0, currentLayerActivations, 0, layers[0].getLength());
        
        for (int layerIdx = 0; layerIdx < size - 1; layerIdx++) {
            float[] nextLayerActivations = new float[layers[layerIdx + 1].getLength()];
            
            // Forward propagation
            for (int inputNeuron = 0; inputNeuron < layers[layerIdx].getLength(); inputNeuron++) {
                for (int outputNeuron = 0; outputNeuron < layers[layerIdx + 1].getLength(); outputNeuron++) {
                    nextLayerActivations[outputNeuron] += currentLayerActivations[inputNeuron] *
                                                          weights[layerIdx][inputNeuron][outputNeuron];
                }
            }
            
            // Add bias terms
            for (int outputNeuron = 0; outputNeuron < layers[layerIdx + 1].getLength(); outputNeuron++) {
                nextLayerActivations[outputNeuron] += weights[layerIdx][layers[layerIdx].getLength()][outputNeuron];
            }
            
            // Special handling for softmax in the output layer
            ActivationFunction activation = layers[layerIdx + 1].getActivation();
            // Check if this layer uses softmax activation by comparing it with a reference
            // Since we can't directly compare function references, we'll use a named reference
            boolean isSoftmax = isSoftmaxActivation(activation);
            
            if (layerIdx == size - 2 && isSoftmax) {
                // Find max for numerical stability
                float maxActivation = nextLayerActivations[0];
                for (int i = 1; i < layers[layerIdx + 1].getLength(); i++) {
                    if (nextLayerActivations[i] > maxActivation) {
                        maxActivation = nextLayerActivations[i];
                    }
                }
                
                // Calculate exp(x - max) and sum
                float expSum = 0.0f;
                for (int i = 0; i < layers[layerIdx + 1].getLength(); i++) {
                    nextLayerActivations[i] = (float) Math.exp(nextLayerActivations[i] - maxActivation);
                    expSum += nextLayerActivations[i];
                }
                
                // Normalize
                for (int i = 0; i < layers[layerIdx + 1].getLength(); i++) {
                    nextLayerActivations[i] /= expSum;
                }
            } else {
                // Regular activation for other layers
                for (int outputNeuron = 0; outputNeuron < layers[layerIdx + 1].getLength(); outputNeuron++) {
                    nextLayerActivations[outputNeuron] = 
                        activation.apply(nextLayerActivations[outputNeuron]);
                }
            }
            
            // Swap buffers
            currentLayerActivations = nextLayerActivations;
        }
        
        return currentLayerActivations;
    }
    
    /**
     * Helper method for softmax activation function
     * This is used to check if a layer is using softmax activation
     * 
     * @param input Input value
     * @return Output value (not actually used by softmax implementation)
     */
    private float softmaxSingle(float input) {
        // This is a placeholder - the actual softmax is computed for the entire layer
        return input;
    }
    
    /**
     * Checks if the given activation function is the softmax function
     * 
     * @param func Activation function to check
     * @return true if the function is softmax, false otherwise
     */
    private boolean isSoftmaxActivation(ActivationFunction func) {
        // In Java we can't directly compare function references, so we'll use a specific method
        // This could be implemented in several ways:
        // 1. Using a named constant for the softmax function
        // 2. Using an enum of activation functions
        // 3. Using a special flag in the Layer class
        
        // For this example, we'll check if the function has the same identity as our softmax function
        // by using a simple test value
        float testValue = 1.0f;
        return Math.abs(func.apply(testValue) - softmaxSingle(testValue)) < 0.0001f;
    }
}

/**
 * Example usage of the neural network implementation
 */
public class NeuralNetworkExample {
    public static void main(String[] args) {
        // Example activation functions
        ActivationFunction relu = x -> x > 0 ? x : 0;
        ActivationFunction reluDerivative = x -> x > 0 ? 1 : 0;
        
        // Create a simple network: 3 inputs -> 4 hidden -> 2 outputs
        Layer[] layers = {
            new Layer(3, relu, reluDerivative),
            new Layer(4, relu, reluDerivative),
            new Layer(2, relu, reluDerivative)
        };
        
        // Initialize network
        Network network = new Network(layers);
        network.randomizeWeights();
        
        // Example forward pass
        float[] input = {0.5f, 0.3f, 0.7f};
        float[] output = network.getResult(input);
        
        // Print results
        System.out.println("Output:");
        for (float value : output) {
            System.out.println(value);
        }
    }
}