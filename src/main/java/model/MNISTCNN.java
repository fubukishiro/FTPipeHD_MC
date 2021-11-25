package model;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.weightinit.impl.XavierInitScheme;

public class MNISTCNN {
    private int inputSize[][] = {{-1, 28, 28}, {-1, 26, 26, 4}, {-1, 13, 13, 4}, {-1, 11, 11, 8}, {-1, 5, 5, 8}};

    public SameDiff makeMNISTNet() {
        SameDiff sd = SameDiff.create();

        //Properties for MNIST dataset:
        int nIn = 28 * 28;
        int nOut = 10;

        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);                 //Shape: [?, 784] - i.e., minibatch x 784 for MNIST
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);             //Shape: [?, 10] - i.e., minibatch x 10 for MNIST

        SDVariable reshaped = in.reshape(-1, 1, 28, 28);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();

        // layer 1: Conv2D with a 3x3 kernel and 4 output channels
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
        SDVariable b0 = sd.zero("b0", 4);

        SDVariable conv1 = sd.cnn().conv2d(reshaped, w0, b0, convConfig);

        // layer 2: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool1 = sd.cnn().maxPooling2d(conv1, poolConfig);

        SDVariable relu1 = sd.nn().relu(pool1, 0);

        // layer 3: Conv2D with a 3x3 kernel and 8 output channels
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
        SDVariable b1 = sd.zero("b1", 8);

        SDVariable conv2 = sd.cnn().conv2d(relu1, w1, b1, convConfig);

        // layer 4: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool2 = sd.cnn().maxPooling2d(conv2, poolConfig);

        SDVariable relu2 = sd.nn().relu(pool2, 0);

        SDVariable flat = relu2.reshape(-1, 5 * 5 * 8);

        // layer 5: Output layer on flattened input
        SDVariable wOut = sd.var("wOut", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT, 5 * 5 * 8, 10);
        SDVariable bOut = sd.zero("bOut", 10);

        SDVariable z = sd.nn().linear("z", flat, wOut, bOut);

        // softmax crossentropy loss function
        SDVariable out = sd.nn().softmax("out", z, 1);
        SDVariable loss = sd.loss().softmaxCrossEntropy("loss", label, out, null);

        sd.setLossVariables(loss);

        return sd;
    }

    /**
     * This is a simple construction of sub model to test the feasibility of the Java NN
     * @param start
     * @param end
     * @return
     */
    public SameDiff simpleMakeSubModel(int start, int end) {
        SameDiff sd = SameDiff.create();

        int[] curInput = inputSize[start];

        int nIn = -1;
        for (int i : curInput) {
            nIn *= i;
        }

        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);
        SDVariable inReshaped = in.reshape(curInput);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();


        for (int i = start; i <= end; i++) {
            String name = "inter_" + i;
            if (i == end) {
                name = "output";
            }
            switch (i) {
                case 0:
                    SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
                    SDVariable b0 = sd.zero("b0", 4);

                    inReshaped = sd.cnn().conv2d(name, inReshaped, w0, b0, convConfig);
                    break;
                case 1:
                    SDVariable pool1 = sd.cnn().maxPooling2d(inReshaped, poolConfig);

                    inReshaped = sd.nn().relu(name, pool1, 0);
                    break;
                case 2:
                    SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
                    SDVariable b1 = sd.zero("b1", 8);

                    inReshaped= sd.cnn().conv2d(name, inReshaped, w1, b1, convConfig);
                    break;
                case 3:
                    SDVariable pool2 = sd.cnn().maxPooling2d(inReshaped, poolConfig);
                    inReshaped = sd.nn().relu(name, pool2, 0);
                    break;
                case 4:
                    int nOut = 10;
                    SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);

                    SDVariable wOut = sd.var("wOut", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT, 5 * 5 * 8, 10);
                    SDVariable bOut = sd.zero("bOut", 10);

                    inReshaped = inReshaped.reshape(-1, 5 * 5 * 8);
                    SDVariable z = sd.nn().linear("z", inReshaped, wOut, bOut);

                    // softmax crossentropy loss function
                    SDVariable out = sd.nn().softmax("output", z, 1);
                    SDVariable loss = sd.loss().softmaxCrossEntropy("loss", label, out, null);

                    sd.setLossVariables(loss);
                    break;
                default:
                    System.out.print("Unknown layer: " + i);
            }
        }
        return sd;
    }
}
