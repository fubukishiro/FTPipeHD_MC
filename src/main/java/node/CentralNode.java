package node;

import org.nd4j.autodiff.samediff.SameDiff;

import model.MNISTCNN;

public class CentralNode {
    public static void main(String[] args) {
        MNISTCNN model = new MNISTCNN();
        model.makeMNISTNet();
        System.out.print("Create");
    }
}
