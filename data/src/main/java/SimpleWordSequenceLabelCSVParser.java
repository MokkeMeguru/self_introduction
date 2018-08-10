import org.apache.uima.resource.ResourceInitializationException;
import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

import static java.util.Collections.max;

public class SimpleWordSequenceLabelCSVParser implements BaseWordSequenceParser {
    private CSVLineSequenceRecordReader recordReader;
    private BaseTextParser textParser;
    private FileSplit inputFile;

    private Map<String, Integer> wordIdDict;
    private Map<Integer, String> idWordDict;
    int currentId;

    private int featureMaxLength;
    private int labelSize;

    private List<List<String>> featuresList;
    private List<Integer> labelList;

    public SimpleWordSequenceLabelCSVParser(CSVLineSequenceRecordReader recordReader, BaseTextParser textParser, FileSplit inputFile) throws IOException, InterruptedException {
        this.recordReader = recordReader;
        this.textParser = textParser;
        this.inputFile = inputFile;
        this.wordIdDict = new HashMap<>();
        this.idWordDict = new HashMap<>();
        this.recordReader.initialize(inputFile);

        this.currentId = 0;
        this.inputFile.reset();
        this.featuresList = new ArrayList<>();
        this.labelList = new ArrayList<>();
    }

    public SimpleWordSequenceLabelCSVParser(CSVLineSequenceRecordReader recordReader, BaseTextParser textParser, FileSplit inputFile,
                                            Map<String, Integer> wordIdDict, Map<Integer, String> idWordDict) throws IOException, InterruptedException {
        this.recordReader = recordReader;
        this.textParser = textParser;
        this.inputFile = inputFile;
        this.wordIdDict = wordIdDict;
        this.idWordDict = idWordDict;
        this.recordReader.initialize(inputFile);

        this.currentId = wordIdDict.size();
        this.inputFile.reset();
        this.featuresList = new ArrayList<>();
        this.labelList = new ArrayList<>();
    }

    @Override
    public void run(int featureIndex, int labelIndex) {
        while(this.recordReader.hasNext()) {
            String text = this.recordReader.next().get(featureIndex).toString();
            int label = this.recordReader.next().get(labelIndex).toInt();
            List<String> features = this.textParser.parse(text);
            features.forEach(str -> {
                addWord(str);
            });
            featuresList.add(features);
            labelList.add(label);
        }
    }

    @Override
    public void save(File featureFile, File labelFile) throws FileNotFoundException {
        PrintStream featureStream = new PrintStream(featureFile);
        AtomicBoolean notfirst = new AtomicBoolean(false);
        featuresList.forEach(stringList -> {
            featureStream.println();
            stringList.forEach(string -> {
                if (notfirst.getAndSet(true)) {
                    featureStream.print(',');
                }
                featureStream.print(wordIdDict.get(string));

            });
        });
        featureStream.flush();
        featureStream.close();
        PrintStream labelStream = new PrintStream(labelFile);
        labelStream.println();
        labelList.forEach(integer -> {
            labelStream.println(integer);
        });
        labelStream.flush();
        labelStream.close();
        featuresList.clear();
        labelList.clear();
    }

    @Override
    public void runAndSave(int featureIndex, int labelIndex, File featureFile, File labelFile) throws FileNotFoundException {
        PrintStream featureStream = new PrintStream(featureFile);
        PrintStream labelStream = new PrintStream(labelFile);
        while(this.recordReader.hasNext()) {
            List<Writable> writables = this.recordReader.next();
            String text = writables.get(featureIndex).toString();
            int label = writables.get(labelIndex).toInt() - 1;
            List<String> features = this.textParser.parse(text);

            if(features.size() > featureMaxLength) featureMaxLength = features.size();

            labelStream.println();
            featureStream.println();
            AtomicBoolean notfirst = new AtomicBoolean(false);
            features.forEach(str -> {
                addWord(str);
                if(notfirst.getAndSet(true)) {
                    featureStream.print(',');
                }
                featureStream.print(wordIdDict.get(str));
            });
            labelStream.print(label);
            labelList.add(label);
        }
        featureStream.flush();
        labelStream.flush();
        featureStream.close();
        labelStream.close();
    }

    @Override
    public void setwordIdDict(Map<String, Integer> wordIdDict, Map<Integer, String> idWordDict) {
        this.wordIdDict = wordIdDict;
        this.idWordDict = idWordDict;
    }

    @Override
    public void setInputFile(FileSplit file) throws IOException, InterruptedException {
        this.inputFile = file;
        this.inputFile.reset();
        this.recordReader.initialize(inputFile);
    }

    @Override
    public Map<String, Integer> getwordIdDict() {
        return this.wordIdDict;
    }

    @Override
    public Map<Integer, String> getIdWordDict() {
        return this.idWordDict;
    }

    @Override
    public int getLabelSize() {
        return max(labelList) + 1;
    }

    @Override
    public void setLabelSize(int labelSize) {
        this.labelSize = labelSize;
    }

    @Override
    public int getFeatureMaxLength() {
        return featureMaxLength;
    }

    @Override
    public int getLabelMaxLength() {
        return 1;
    }

    @Override
    public void setFeatureMaxLength(int featureMaxLength) {
        this.featureMaxLength = featureMaxLength;
    }


    @Override
    public void setLabelMaxLength(int labelMaxLength) {
    }

    @Override
    public int getDictSize() {
        return currentId;
    }

    @Override
    public void addWord(String word) {
        if(!wordIdDict.containsKey(word)) {
            wordIdDict.put(word, currentId);
            idWordDict.put(currentId, word);
            ++currentId;
        }
    }

    @Override
    public List<Integer> text2vecs(String text, String unknown) {
        return this.textParser.parse(text)
                        .stream()
                        .map(str -> {
                            if (wordIdDict.containsKey(str)) {
                                return wordIdDict.get(str);
                            } else {
                                return wordIdDict.get(unknown);
                            }
                        })
                        .collect(Collectors.toList());
    }


    public static void main (String... args) throws ResourceInitializationException, IOException, InterruptedException {
        System.out.println("Start!");
        BaseWordSequenceParser sequenceParser =
                new SimpleWordSequenceLabelCSVParser(
                        new CSVLineSequenceRecordReader(1, ','),
                        new SimpleEnglishTextParser(),
                        new FileSplit(new File("resources/question_source.csv")));
        sequenceParser.runAndSave(1,2,
                new File("resources/features.csv"),
                new File("resources/label.csv"));
        System.out.println("Finish!");
    }
}
