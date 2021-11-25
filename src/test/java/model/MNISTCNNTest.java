package model;

import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class MNISTCNNTest {
    @Test
    public void simpleSubModelCreateTest() {
        MNISTCNN testModel = new MNISTCNN();
        MNISTCNN testSubModel = new MNISTCNN();
        SameDiff model = testModel.makeMNISTNet();
        SameDiff subModel = testSubModel.simpleMakeSubModel(0, 1);
        SameDiff subModel2 = testSubModel.simpleMakeSubModel(2, 3);
        SameDiff subModel3 = testSubModel.simpleMakeSubModel(4, 4);
        System.out.print("Created");
    }

    @Test
    public void subModelFeedForwardTest() throws Exception {
        MNISTCNN testSubModel = new MNISTCNN();
        SameDiff subModel = testSubModel.simpleMakeSubModel(0, 1);
        SameDiff subModel2 = testSubModel.simpleMakeSubModel(2, 3);
        SameDiff subModel3 = testSubModel.simpleMakeSubModel(4, 4);

        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

        while (trainData.hasNext()) {
            // subModel 1
            INDArray input = trainData.next().getFeatures();
            subModel.getVariable("input").setArray(input);
            INDArray output = subModel.getVariable("output").eval();

            // subModel 2
            subModel2.getVariable("input").setArray(output);
            output = subModel2.getVariable("output").eval();

            // subModel 3
            subModel3.getVariable("input").setArray(output);
            output = subModel3.getVariable("loss").eval();

            System.out.print("HI");
        }
    }
}
