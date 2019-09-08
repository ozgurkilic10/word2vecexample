package tr.edu.mu.ceng.ir.word2vec;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;

import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


import org.deeplearning4j.text.sentenceiterator.SentenceIterator;


import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
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

        actionParamsMap.put("train", "file/directory targetmodelfile/folder");
        actionParamsMap.put("load", "sourcezipfile");
        actionParamsMap.put("retrain", "sourcezipfile file/directory targetmodelfile/folder");

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

        logger.debug( "Max Memory = " + Runtime.getRuntime().maxMemory() / 1024 / 1024 + " GB ");
        logger.debug( "Total Memory = " + Runtime.getRuntime().totalMemory() / 1024 / 1024 + " GB ");
        logger.debug( "Number of processors = " + Runtime.getRuntime().availableProcessors());
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


    private void loadVectors(String targetFileFolder) throws IOException {
        logger.debug("=====Starting Reading Vectors  from " + targetFileFolder + " =====");
        if (properties.getProperty("word2vec.modelCompressed","true").toLowerCase().equals("true")) {
            vec = WordVectorSerializer.readWord2VecModel(targetFileFolder, true);
        }else{
            readUnCompressed(targetFileFolder);
        }
        logger.debug("=====Reading Vectors Completed=====");

    }



    private void train(String sourceFileFolder, String targetArchiveFile) throws IOException {
        logger.debug("=====Starting Word2Vec Training for " + sourceFileFolder + " =====");
        SentenceIterator iter = new FileSentenceIterator((new File(sourceFileFolder)));

        //Set preprocessor to lowercase tokens
        if (properties.getProperty("word2vec.preprocessor","true").toLowerCase().equals("true")) {
            iter.setPreProcessor(new SentencePreProcessor() { //Preprocess input
                @Override
                public String preProcess(String sentence) {
                    //CommonPreprocessor ile de bu yapilabilir
                    return sentence.toLowerCase();
                }
            });
        }

        String stopWordFile = (properties.getProperty("word2vec.stopWordFile",null));
        if (stopWordFile != null){
            readStopWords(stopWordFile);
        }
        logger.debug("Stop words: " + stopWords);


        File confFile = new File("config.json");
        VectorsConfiguration configuration = null;
        if (confFile.exists()) {
            InputStream stream = new FileInputStream(confFile);
            StringBuilder builder = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    builder.append(line);
                }
            }

            configuration = VectorsConfiguration.fromJson(builder.toString().trim());
        }
        Word2Vec.Builder builder = null;
        if (configuration != null){
            builder = new Word2Vec.Builder(configuration);
            logger.debug("======using confiuration=======");
            logger.debug(configuration.toJson());
        }else{
            builder = new Word2Vec.Builder().
                    layerSize(Integer.parseInt(properties.getProperty("word2vec.dimension","100"))). //set the dimension of the vector representation
                    windowSize(Integer.parseInt(properties.getProperty("word2vec.window","5"))).
                            minWordFrequency(Integer.parseInt(properties.getProperty("word2vec.minWordFrequency","5")));

        }

        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        vec = builder.stopWords(stopWords).iterate(iter).tokenizerFactory(tokenizerFactory).
                elementsLearningAlgorithm(properties.getProperty("word2vec.algorithm","SkipGram").equals("CBOW")
                    ? new CBOW<>() : new SkipGram<>()).   //SkipGram or CBOW algorithm
                build();


        vec.fit(); //performs training
        logger.debug("=====Training Completed=====");
        logger.debug("Vocabulary size: " + vec.getVocab().numWords());
        logger.debug("# of documents: " + vec.getVocab().totalNumberOfDocs());
        logger.debug("# of word occurrences: " + vec.getVocab().totalWordOccurrences());

        logger.debug( "Max Memory = " + Runtime.getRuntime().maxMemory() / 1024 / 1024 + " GB ");
        logger.debug( "Total Memory = " + Runtime.getRuntime().totalMemory() / 1024 / 1024 + " GB ");
        logger.debug( "Free Memory = " + Runtime.getRuntime().freeMemory() / 1024 / 1024 + " GB ");

        logger.debug("=====Serializing word vectors to " + targetArchiveFile+" =====");
        if (properties.getProperty("word2vec.modelCompressed","true").toLowerCase().equals("true")) {
            WordVectorSerializer.writeWord2VecModel(vec, targetArchiveFile);
        }else {
            writeUnCompressed(targetArchiveFile);
        }

        logger.debug("=====Serialization Completed=====");

    }

    private void retrain(String sourceArchiveFile, String sourceFileFolder, String targetArchiveFile) throws IOException {
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

        if (properties.getProperty("word2vec.modelCompressed","true").toLowerCase().equals("true")) {
            WordVectorSerializer.writeWord2VecModel(vec, targetArchiveFile);
        }else {
            writeUnCompressed(targetArchiveFile);
        }
        logger.debug("=====Serialization Completed=====");


    }



    private void readStopWords(String stopWordFile) throws IOException {

        try (BufferedReader br = new BufferedReader(new FileReader(stopWordFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.replaceAll(" ","").split(",");
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

    public void writeUnCompressed(String targetFolder) throws IOException {
        Path path  = Paths.get(targetFolder);
        if (!Files.exists(path))
            Files.createDirectory(path);
        WordVectorSerializer.writeWordVectors(vec.lookupTable(), new File(targetFolder+"/syn0.txt"));

        // writing out syn1
        INDArray syn1 = ((InMemoryLookupTable<VocabWord>) vec.getLookupTable()).getSyn1();

        if (syn1 != null)
            try (PrintWriter writer = new PrintWriter(new FileWriter( new File(targetFolder+"/syn1.txt")))) {
                for (int x = 0; x < syn1.rows(); x++) {
                    INDArray row = syn1.getRow(x);
                    StringBuilder builder = new StringBuilder();
                    for (int i = 0; i < row.length(); i++) {
                        builder.append(row.getDouble(i)).append(" ");
                    }
                    writer.println(builder.toString().trim());
                }
            }

        // writing out syn1
        INDArray syn1Neg = ((InMemoryLookupTable<VocabWord>) vec.getLookupTable()).getSyn1Neg();

        if (syn1Neg != null)
            try (PrintWriter writer = new PrintWriter(new FileWriter( new File(targetFolder+"/syn1Neg.txt")))) {
                for (int x = 0; x < syn1Neg.rows(); x++) {
                    INDArray row = syn1Neg.getRow(x);
                    StringBuilder builder = new StringBuilder();
                    for (int i = 0; i < row.length(); i++) {
                        builder.append(row.getDouble(i)).append(" ");
                    }
                    writer.println(builder.toString().trim());
                }
            }

        // writing out huffman codes
        try (PrintWriter writer = new PrintWriter(new FileWriter(new File(targetFolder+"/codes.txt")))) {
            for (int i = 0; i < vec.getVocab().numWords(); i++) {
                VocabWord word = vec.getVocab().elementAtIndex(i);
                StringBuilder builder = new StringBuilder(WordVectorSerializer.encodeB64(word.getLabel())).append(" ");
                for (int code : word.getCodes()) {
                    builder.append(code).append(" ");
                }

                writer.println(builder.toString().trim());
            }
        }


        // writing out huffman tree
        try (PrintWriter writer = new PrintWriter(new File(targetFolder+"/huffman.txt"))) {
            for (int i = 0; i < vec.getVocab().numWords(); i++) {
                VocabWord word = vec.getVocab().elementAtIndex(i);
                StringBuilder builder = new StringBuilder(WordVectorSerializer.encodeB64(word.getLabel())).append(" ");
                for (int point : word.getPoints()) {
                    builder.append(point).append(" ");
                }

                writer.println(builder.toString().trim());
            }
        }

        // writing out word frequencies
        try (PrintWriter writer = new PrintWriter(new File(targetFolder+"/frequencies.txt"))) {
            for (int i = 0; i < vec.getVocab().numWords(); i++) {
                VocabWord word = vec.getVocab().elementAtIndex(i);
                StringBuilder builder = new StringBuilder(WordVectorSerializer.encodeB64(word.getLabel())).append(" ")
                        .append(word.getElementFrequency()).append(" ")
                        .append(vec.getVocab().docAppearedIn(word.getLabel()));

                writer.println(builder.toString().trim());
            }
        }


        try (FileOutputStream writer = new FileOutputStream((targetFolder+"/config.json"))){
            writer.write(vec.getConfiguration().toJson().getBytes(StandardCharsets.UTF_8));
        }
    }

    private void readUnCompressed(String targetFileFolder) throws IOException {
        logger.debug("Trying full model restoration...");


        File fileSyn0 = new File(targetFileFolder+"/syn0.txt");
        File fileSyn1 = new File(targetFileFolder+"/syn1.txt");
        File fileCodes = new File(targetFileFolder+"/codes.txt");
        File fileHuffman = new File(targetFileFolder+"/huffman.txt");
        File fileFreq = new File(targetFileFolder+"/frequencies.txt");


        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();

        if (originalPeriodic)
            Nd4j.getMemoryManager().togglePeriodicGc(false);

        Nd4j.getMemoryManager().setOccasionalGcFrequency(50000);

        try {
            InputStream stream = new FileInputStream(targetFileFolder+"/config.json");
            StringBuilder builder = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    builder.append(line);
                }
            }

            VectorsConfiguration configuration = VectorsConfiguration.fromJson(builder.toString().trim());

            // we read first 4 files as w2v model
            vec = WordVectorSerializer.readWord2VecFromText(fileSyn0, fileSyn1, fileCodes, fileHuffman, configuration);

            if (fileFreq.exists()) {
                // we read frequencies from frequencies.txt, however it's possible that we might not have this file
                stream = new FileInputStream(fileFreq);
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] split = line.split(" ");
                        VocabWord word = vec.getVocab().tokenFor(WordVectorSerializer.decodeB64(split[0]));
                        word.setElementFrequency((long) Double.parseDouble(split[1]));
                        word.setSequencesCount((long) Double.parseDouble(split[2]));
                    }
                }

            }
            if (fileSyn1.exists()) {
                stream = new FileInputStream(fileSyn1);

                try (InputStreamReader isr = new InputStreamReader(stream);
                     BufferedReader reader = new BufferedReader(isr)) {
                    String line = null;
                    List<INDArray> rows = new ArrayList<>();
                    while ((line = reader.readLine()) != null) {
                        String[] split = line.split(" ");
                        double array[] = new double[split.length];
                        for (int i = 0; i < split.length; i++) {
                            array[i] = Double.parseDouble(split[i]);
                        }
                        rows.add(Nd4j.create(array));
                    }

                    // it's possible to have full model without syn1Neg
                    if (!rows.isEmpty()) {
                        INDArray syn1Neg = Nd4j.vstack(rows);
                        ((InMemoryLookupTable) vec.getLookupTable()).setSyn1Neg(syn1Neg);
                    }
                }

            }

        } finally {
            if (originalPeriodic)
                Nd4j.getMemoryManager().togglePeriodicGc(true);
            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);
        }
    }


}
