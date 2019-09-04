package tr.edu.mu.ceng.ir.word2vec;

import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Word2VecExample {

    private static final String CMD = "run.sh";
    private static Logger logger = LoggerFactory.getLogger(Word2VecExample.class);


    private Word2Vec vec;
    private Map<String,String> actionParamsMap = new LinkedHashMap<>();
    private Properties properties;
    private List<String> stopWords = new ArrayList<>();
    

    public Word2VecExample(){

        actionParamsMap.put("train", "file/directory targetzipfile");
        actionParamsMap.put("load", "sourcezipfile");
        actionParamsMap.put("retrain", "sourcezipfile file/directory targetzipfile");

        readProperties();

    }


    private  void usage() {
        System.out.println("Usage: " + CMD + " ACTION");
        System.out.println("where ACTION is one of:");
        for (String action : actionParamsMap.keySet()) {
            System.out.println(action + " " + actionParamsMap.get(action));
        }
    }

    private void usage(String action){
        System.out.println("Usage: " + action + " " + actionParamsMap.get(action));
    }

    public static void main (String... args) throws IOException {
        Word2VecExample wve = new Word2VecExample();


        System.out.println();

        if (args.length == 0) {
            wve.usage();
            System.exit(0);
        }

        String action = args[0];
        switch (action){
            case "train":
                if (args.length<3){
                    wve.usage(args[0]);
                    System.exit(0);
                }
                wve.train(args[1], args[2]);
                break;
            case "load" :
                if (args.length<2){
                    wve.usage(args[0]);
                    System.exit(0);
                }
                wve.loadVectors(args[1]);
                break;
            case "retrain":
                if (args.length<4){
                    wve.usage(args[0]);
                    System.exit(0);
                }
                wve.retrain(args[1], args[2], args[3]);
                break;
            default:
                wve.usage();
        }



        Scanner input = new Scanner(System.in);

        System.out.println("Do you want to find closest words of a word? (Y/N)");
        String str = input.nextLine();
        if (!str.toLowerCase().equals("y"))
            return;
        while (true){
            System.out.print("(To quit type q) Enter the word: ");
            String word = input.nextLine();
            if (word.toLowerCase().equals("q"))
                break;
            System.out.print("number of closest words: ");
            int n = input.nextInt();
            input.nextLine();
            System.out.println( "Frequency of " + word + " is " + wve.vec.getVocab().wordFrequency(word));
            Collection<String> lst = wve.vec.wordsNearest(word,n);  //get the closest n words for given word
            System.out.println("closest " + n + " words for " + word + " are " + lst);

        }

    }

    private void retrain(String sourceArchiveFile, String sourceFileFolder, String targetArchiveFile) {
        loadVectors(sourceArchiveFile);
        logger.debug("=====Starting Word2Vec Retraining for " + sourceFileFolder + " =====");
        logger.debug("Stop words: " + vec.getStopWords());
        SentenceIterator iter = new FileSentenceIterator((new File(sourceFileFolder)));
        iter.setPreProcessor(new SentencePreProcessor() { //Preprocess input
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });
        vec.setSentenceIterator(iter);
        vec.setElementsLearningAlgorithm(properties.getProperty("word2vec.algorithm","SkipGram").
                equals("CBOW")? new CBOW<>() : new SkipGram<>());

        vec.getConfiguration().setWindow(Integer.parseInt(properties.getProperty("word2vec.window","5")));


        vec.fit();
        logger.debug("=====Training Completed=====");

        logger.debug("=====Serializing word vectors to "+ targetArchiveFile + " =====");
        WordVectorSerializer.writeWord2VecModel(vec,targetArchiveFile);
        logger.debug("=====Serialization Completed=====");
    }

    private void loadVectors(String archiveFile) {
        logger.debug("=====Starting Reading Vectors  from " + archiveFile + " =====");
        vec = WordVectorSerializer.readWord2VecModel(archiveFile,true);
        logger.debug("=====Reading Vectors Completed=====");

    }



    private void train(String sourceFileFolder, String targetArchiveFile) throws IOException {
        logger.debug("=====Starting Word2Vec Training for " + sourceFileFolder + " =====");
        SentenceIterator iter = new FileSentenceIterator((new File(sourceFileFolder)));

        iter.setPreProcessor(new SentencePreProcessor() { //Preprocess input
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        //to remove punctuation marks, numbers, special characters
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());


        
        String stopWordFile = (properties.getProperty("word2vec.stopWordFile",null));
        if (stopWordFile != null){
            readStopWords(stopWordFile);
        }
        logger.debug("Stop words: " + stopWords);
        vec = new Word2Vec.Builder().stopWords(stopWords).
                layerSize(Integer.parseInt(properties.getProperty("word2vec.dimension","100"))). //set the dimension of the vector representation
                windowSize(Integer.parseInt(properties.getProperty("word2vec.window","5"))).
                iterate(iter).  //iterator for the selected corpus
     //           tokenizerFactory(tokenizerFactory).
                elementsLearningAlgorithm(properties.getProperty("word2vec.algorithm","SkipGram").equals("CBOW")
                    ? new CBOW<>() : new SkipGram<>()).   //SkipGram or CBOW algorithm
                build();


        vec.fit(); //performs training
        logger.debug("=====Training Completed=====");
        logger.debug("Vocabulary size: " + vec.getVocab().numWords());
        logger.debug("# of documents: " + vec.getVocab().totalNumberOfDocs());
        logger.debug("# of word occurrences: " + vec.getVocab().totalWordOccurrences());


        logger.debug("=====Serializing word vectors to " + targetArchiveFile+" =====");
        WordVectorSerializer.writeWord2VecModel(vec,targetArchiveFile);
        logger.debug("=====Serialization Completed=====");

    }

    private void readStopWords(String stopWordFile) throws IOException {

        try (BufferedReader br = new BufferedReader(new FileReader(stopWordFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                stopWords.addAll(Arrays.asList(values));
            }
        }
    }


    private  void readProperties() {

       this.properties = new Properties();

        Path path = Paths.get("config.properties");

        if (Files.exists(path) && Files.isReadable(path))

            try (InputStream input = Files.newInputStream(Paths.get("config.properties"))) {
                properties.load(input);
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        else
            try (InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties")) {

                if (input == null) {
                    System.out.println("Sorry, unable to find config.properties in class path");
                    return;
                }
                properties.load(input);

            } catch (IOException ex) {
                ex.printStackTrace();
            }

    }

}
