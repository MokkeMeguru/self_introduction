import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;

import java.util.List;

public class SimpleEnglishTextParser implements BaseTextParser {
    private TokenizerFactory tokenizer;

    public SimpleEnglishTextParser() throws ResourceInitializationException {
        this.tokenizer = new UimaTokenizerFactory();
        this.tokenizer.setTokenPreProcessor(new LowCasePreProcessor());
    }

    public SimpleEnglishTextParser(TokenizerFactory tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public List<String> parse(String text) {
        return this.tokenizer.create(text).getTokens();
    }

    public static void main (String... args) throws ResourceInitializationException {
        BaseTextParser textParser = new SimpleEnglishTextParser();
        System.out.println(textParser.parse("Hello, my name is Meguru. Nice to meet you."));
        // => [hello, ,, my, name, is, meguru, ., nice, to, meet, you, .]
    }
}
