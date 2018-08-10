import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public class Sequence2VecDataSetIteratorFactory {
    public static DataSetIterator createDataSetIterator (
            int labelSize,
            FileSplit featureFile,
            FileSplit labelFile
    ) throws IOException, InterruptedException {
        CSVLineSequenceRecordReader featureSequenceRecordReader =
                new CSVLineSequenceRecordReader(1,',');
        CSVLineSequenceRecordReader labelSequenceRecordReader =
                new CSVLineSequenceRecordReader(1,',');

        featureSequenceRecordReader.initialize(featureFile);
        labelSequenceRecordReader.initialize(labelFile);

        featureSequenceRecordReader.reset();
        labelSequenceRecordReader.reset();

        return new SequenceRecordReaderDataSetIterator(
                        featureSequenceRecordReader,
                        labelSequenceRecordReader,
                        5,
                        labelSize,
                        false,
                        SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
    };
}
