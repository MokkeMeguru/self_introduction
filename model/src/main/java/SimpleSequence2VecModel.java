import org.apache.uima.resource.ResourceInitializationException;
import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class SimpleSequence2VecModel {
    private DataSetIterator dataSetIterator;
    private int dictSize;
    private int labelSize;
    private int featureMaxLength;

    // the network
    private ComputationGraph net;

    public SimpleSequence2VecModel(DataSetIterator dataSetIterator,
                                   int dictSize,
                                   int labelSize) {
        this.dataSetIterator = dataSetIterator;
        this.dictSize = dictSize;
        this.labelSize = labelSize;
    }

    public void loadNetWork(File file) throws IOException {
        net = ModelSerializer.restoreComputationGraph(file);
    }

    public void initNetWork(boolean showUI) {

        // network settings
        final NeuralNetConfiguration.Builder builder =
                new NeuralNetConfiguration.Builder()
                        .seed(140)
                        .updater(new RmsProp(RmsProp.DEFAULT_RMSPROP_LEARNING_RATE))
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final ComputationGraphConfiguration.GraphBuilder graphBuilder =
                builder.graphBuilder()
                        .pretrain(false)
                        .backprop(true)
                        .backpropType(BackpropType.Standard)
                        .addInputs("inputLine")
                        .setInputTypes(InputType.recurrent(dictSize))
                        .addLayer("embeddingEncoder",
                                new EmbeddingLayer.Builder().nIn(dictSize).nOut(dictSize).build(),
                                "inputLine")
                        .addLayer("encoder",
                                new LSTM.Builder()
                                        .nIn(dictSize)
                                        .nOut(dictSize / 4)
                                        .activation(Activation.TANH)
                                        .build(),
                                "embeddingEncoder")
                        .addLayer("output",
                                new RnnOutputLayer.Builder()
                                        .nIn(dictSize / 4)
                                        .nOut(labelSize)
                                        .activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                                        .build(),
                                "encoder")
                        .setOutputs("output");

        net = new ComputationGraph(graphBuilder.build());
        net.init();

        if (showUI) {
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            statsStorage.removeAllListeners();
            uiServer.attach(statsStorage);
            net.setListeners(new StatsListener(statsStorage));
        } else {
            StatsStorage statsStorage = new InMemoryStatsStorage();
            statsStorage.removeAllListeners();
            net.setListeners(new StatsListener(statsStorage));
        }
    }

    public void train() {
        for (int epoch = 1; epoch < 300; ++epoch) {
            System.out.println("Epoch :" + epoch);
            this.dataSetIterator.reset();
            while (this.dataSetIterator.hasNext()) {
                DataSet trainData = this.dataSetIterator.next();
                net.fit(trainData);
            }
        }
    }

    public void saveModel() throws IOException {
        File saveFile = new File("resources/ComputationGraph.zip");
        boolean saveUpdater = true;
        ModelSerializer.writeModel(net, saveFile, saveUpdater);
    }

    public void setFeatureMaxLength(int featureMaxLength) {
        this.featureMaxLength = featureMaxLength;
    }

    private int rawPredictData(double[] doubles) {
        net.rnnClearPreviousState();
        INDArray[] o = net.output(Nd4j.create(new double[][][]{{doubles}}));
        // System.out.println(o[0]);
        System.out.println(o[0].getRow(0).tensorAlongDimension(doubles.length - 1, 0));
        int result;
        result = (int) Nd4j.getExecutioner().exec(
                new IMax(o[0].getRow(0).tensorAlongDimension(doubles.length - 1, 0)),
                1).getDouble(0);
        // System.out.println(result);
        return result;
    }

    public static void main(String... args)
            throws IOException, ResourceInitializationException, InterruptedException {
        // ----------------------------------
        System.out.println("Create DataSet!");
        BaseWordSequenceParser sequenceParser =
                new SimpleWordSequenceLabelCSVParser(
                        new CSVLineSequenceRecordReader(1, ','),
                        new SimpleEnglishTextParser(),
                        new FileSplit(new File("resources/question_source.csv")));
        sequenceParser.runAndSave(1, 2,
                new File("resources/features.csv"),
                new File("resources/label.csv"));

        sequenceParser.addWord("<unk>");
        sequenceParser.addWord("<go>");
        sequenceParser.addWord("<eos>");

        int labelSize = sequenceParser.getLabelSize();
        int dictSize = sequenceParser.getIdWordDict().size();
        int featureMaxLength = sequenceParser.getFeatureMaxLength();
        System.out.println(featureMaxLength);
        System.out.println("Creation Finish!");
        // -----------------------------------
        DataSetIterator wordSequenceParser =
                Sequence2VecDataSetIteratorFactory.createDataSetIterator(
                        labelSize,
                        new FileSplit(new File("resources/features.csv")),
                        new FileSplit(new File("resources/label.csv"))
                );
        // ------------------------------------
        SimpleSequence2VecModel model = new SimpleSequence2VecModel(
                wordSequenceParser,
                dictSize,
                labelSize);

        // ------------------------------------
        // boolean showUI = false;
        // System.out.println("Initialize model");
        // model.setFeatureMaxLength(featureMaxLength);
        // model.initNetWork(showUI);
        // System.out.println("Initialize Finish");
        // -------------------------------------
        // System.out.println("Train model");
        // model.train();
        // System.out.println("Train Finish");
        // -------------------------------------------
        // System.out.println("Save Start");
        // model.saveModel();
        // System.out.println("Save Finish");
        // -------------------------------------------
         System.out.println("load Model");
         model.loadNetWork(new File("resources/ComputationGraph.zip"));
         System.out.println("load Finish");
        // ------------------------------------------
        Scanner scanner = new Scanner(System.in);
        int result;
        CSVLineSequenceRecordReader answerReader = new CSVLineSequenceRecordReader(1, ',');
        answerReader.initialize(new FileSplit(new File("resources/answer_source.csv")));
        Map<Integer, String> answerMap = new HashMap<>();
        while (answerReader.hasNext()) {
            List<Writable> writables = answerReader.next();
            answerMap.put(writables.get(0).toInt(), writables.get(1).toString());
        }
        while (true) {
            System.out.println("> [Exit : type \"QUIT\"]");
            String str = scanner.nextLine();
            if (str.toUpperCase().equals("QUIT")) return;
            double[] doubles =
                    sequenceParser
                            .text2vecs(str, "<unk>")
                            .stream()
                            .mapToDouble(d -> d)
                            .toArray();
            result = model.rawPredictData(doubles);
            System.out.println(">> " + answerMap.get(result + 1));
        }
    }
}
