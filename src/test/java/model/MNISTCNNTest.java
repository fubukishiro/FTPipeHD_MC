package model;

import org.bytedeco.javacv.FrameFilter;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Adam;

import java.util.HashMap;
import java.util.Map;

public class MNISTCNNTest {
    @Test
    public void simpleSubModelCreateTest() throws Exception{
        MNISTCNN testModel = new MNISTCNN();
        MNISTCNN testSubModel = new MNISTCNN();
        SameDiff model = testModel.makeMNISTNet();
        SameDiff subModel = testSubModel.simpleMakeSubModel(0, 1);
        SameDiff subModel2 = testSubModel.simpleMakeSubModel(2, 3);
        SameDiff subModel3 = testSubModel.simpleMakeSubModel(4, 4);
        System.out.print("Created");

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)                               //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
                .build();

        model.setTrainingConfig(config);

        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

        while (trainData.hasNext()) {

            INDArray input = trainData.next().getFeatures();
            model.getVariable("input").setArray(input);
            INDArray output = model.getVariable("out").eval();
            System.out.print("HI");
        }
    }

    @Test
    public void subModelTrainTest() throws Exception {
        MNISTCNN testSubModel = new MNISTCNN();
        MNISTCNN testSubModel2 = new MNISTCNN();
        MNISTCNN testSubModel3 = new MNISTCNN();
        SameDiff model = testSubModel.makeMNISTNet();

        // Split a five-layer CNN model into three sub-models
        SameDiff subModel = testSubModel.simpleMakeSubModel(0, 1);
        SameDiff subModel2 = testSubModel2.simpleMakeSubModel(2, 3);
        SameDiff subModel3 = testSubModel3.simpleMakeSubModel(4, 4);

        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

        int iteration = 1;
        int epoch = 1;

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)                               //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
                .build();
        model.setTrainingConfig(config);

        subModel.setTrainingConfig(config);
        subModel2.setTrainingConfig(config);
        subModel3.setTrainingConfig(config);

        Map<String, GradientUpdater> gu1 = testSubModel.externalInitializeTraining();
        Map<String, GradientUpdater> gu2 = testSubModel2.externalInitializeTraining();
        Map<String, GradientUpdater> gu3 = testSubModel3.externalInitializeTraining();

        // I want to train these three sub-models in a consecutive order
        // I want to forward the subModel, subModel2 and subModel3 sequentially and then backward them in a reverse order
        while (trainData.hasNext()) {
            DataSet curData = trainData.next();
            INDArray input = curData.getFeatures();
            INDArray label = curData.getLabels();

            // subModel forward
            subModel.getVariable("input").setArray(input);
            INDArray output = subModel.getVariable("output").eval();

            // subModel2 forward
            subModel2.getVariable("input").setArray(output);
            output = subModel2.getVariable("output").eval();

            // subModel3 forward
            subModel3.getVariable("label").setArray(label);
            subModel3.getVariable("input").setArray(output);
            output = subModel3.getVariable("loss").eval();

            // subModel3 backward
            Map<String, INDArray> grads3 = subModel3.calculateGradients(null, subModel3.getVariables().keySet());
            testSubModel3.step();

            // subModel2 backward
            ExternalErrorsFunction fn = SameDiffUtils.externalErrors(subModel2, null, subModel2.getVariable("output"));
            INDArray externalGrad = grads3.get("reshapedInput").reshape(-1, 8, 5, 5);
            Map<String, INDArray> externalGradMap = new HashMap<String, INDArray>();
            externalGradMap.put("output-grad", externalGrad);
            Map<String, INDArray> grad2 = subModel2.calculateGradients(externalGradMap, subModel2.getVariables().keySet()); // TODO: error occured here


            Map<String, INDArray> grad = subModel.calculateGradients(null, subModel.getVariables().keySet());

            testSubModel.step();
            
        }
    }

    @Test
    public void externalInitializeTrainingTest() {
        MNISTCNN testSubModel = new MNISTCNN();
        MNISTCNN testSubModel2 = new MNISTCNN();
        MNISTCNN testSubModel3 = new MNISTCNN();
        SameDiff model = testSubModel.makeMNISTNet();
        SameDiff subModel = testSubModel.simpleMakeSubModel(0, 1);
        SameDiff subModel2 = testSubModel2.simpleMakeSubModel(2, 3);
        SameDiff subModel3 = testSubModel3.simpleMakeSubModel(4, 4);

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)                               //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
                .build();
        model.setTrainingConfig(config);

        subModel.setTrainingConfig(config);
        subModel2.setTrainingConfig(config);
        subModel3.setTrainingConfig(config);

        Map<String, GradientUpdater> gu1 = testSubModel.externalInitializeTraining();
        Map<String, GradientUpdater> gu2 = testSubModel2.externalInitializeTraining();
        Map<String, GradientUpdater> gu3 = testSubModel3.externalInitializeTraining();

    }
}
