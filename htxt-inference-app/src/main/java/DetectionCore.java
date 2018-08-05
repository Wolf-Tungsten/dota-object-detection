import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMap;
import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem;

import com.google.protobuf.TextFormat;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;

import com.sun.image.codec.jpeg.JPEGCodec;
import com.sun.image.codec.jpeg.JPEGImageEncoder;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;

/**
 * Java inference for the Object Detection API at:
 * https://github.com/tensorflow/models/blob/master/research/object_detection/
 */
public class DetectionCore {
    private static int STRIDE = 250;
    private static int LENGTH = 500;
    private static int MAX_BATCH_SIZE = 4;

    private static String[] labels = {"", "plane", "helicopter"};
    private String modelPath;
    private SavedModelBundle model;

    private Session session;
    public DetectionCore() {
        modelPath = Paths.get(DetectionCore.class.getClassLoader().getResource("model/saved_model/").getPath().substring(1)).toString();
        try {
            model = SavedModelBundle.load(modelPath, "serve");
            printSignature(model);
            session = model.session();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public ArrayList<BBox> detect(String imagePath) throws Exception{

        BufferedImage img = loadImage(imagePath);
        int width = Math.max(img.getWidth(), LENGTH);
        int height = Math.max(img.getHeight(), LENGTH);
        BufferedImage paddedImage = paddingImage(img, width, height);

        // 生成分割网格
        ArrayList<Integer> xList = new ArrayList<Integer>();
        ArrayList<Integer> yList = new ArrayList<Integer>();
        for (int i = 0; i * STRIDE + LENGTH <= width; i++) {
            xList.add(i * STRIDE);
        }
        for (int i = 0; i * STRIDE + LENGTH <= height; i++) {
            yList.add(i * STRIDE);
        }

        if (xList.get(xList.size() - 1) + LENGTH + 1< width) {
            xList.add(width - LENGTH);
        }
        if (yList.get(yList.size() - 1) + LENGTH + 1< height) {
            yList.add(height - LENGTH);
        }


        ArrayList<BBox> result = new ArrayList<BBox>();

        ArrayList<BufferedImage> splitImage = new ArrayList<>();
        ArrayList<Integer> minXs = new ArrayList<>();
        ArrayList<Integer> minYs = new ArrayList<>();



        for (int x:xList) {
            for (int y:yList){
                splitImage.add(paddedImage.getSubimage(x, y, LENGTH, LENGTH));
                minXs.add(x);
                minYs.add(y);
            }
        }

        ArrayList<BufferedImage> imagesBatch = new ArrayList<>();
        ArrayList<Integer> minXsBatch = new ArrayList<>();
        ArrayList<Integer> minYsBatch = new ArrayList<>();


        int processed = 0;
        for (BufferedImage image:splitImage) {
            int index = splitImage.indexOf(image);

            imagesBatch.add(image);
            minXsBatch.add(minXs.get(index));
            minYsBatch.add(minYs.get(index));
            processed += 1;

            drawProcessBar((double)(processed)/(splitImage.size()));
            if (imagesBatch.size() >= MAX_BATCH_SIZE || processed >= splitImage.size()) {
                ArrayList<ArrayList<BBox>> resultList= detectInBatch(makeImagesTensor(imagesBatch), minXsBatch, minYsBatch);
                for (ArrayList<BBox> list:resultList) {
                    result.addAll(list);
                    for ( BBox bBox:list) {
                        System.out.println(bBox.label);
                    }
                }
                imagesBatch.clear();
                minXsBatch.clear();
                minYsBatch.clear();
            }
        }

        return BBox.bBoxFilter(result);
    }

    public static void drawProcessBar (double process) {
        System.out.print("\r当前进度：[-");
        System.out.print(process*100);
        System.out.print("%-]");
    }

    public ArrayList<BBox> detectInSlice(Tensor<UInt8> imageSlice, int minX, int minY) {
        List<Tensor<?>> outputs = null;
        outputs = session.runner()
                        .feed("image_tensor", imageSlice)
                        .fetch("detection_scores").fetch("detection_classes").fetch("detection_boxes")
                        .run();
        ArrayList<BBox> objects = new ArrayList<BBox>();
        try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
             Tensor<Float> classesT = outputs.get(1).expect(Float.class);
             Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
            // All these tensors have:
            // - 1 as the first dimension
            // - maxObjects as the second dimension
            // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
            // This can be verified by looking at scoresT.shape() etc.
            int maxObjects = (int) scoresT.shape()[1];
            float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
            float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
            float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
            // Print all objects whose score is at least 0.5.
            for (int i = 0; i < scores.length; ++i) {
                if (scores[i] < 0.5) {
                    continue;
                }
                //System.out.printf("\tFound %-20s (score: %.4f)\n", labels[(int) classes[i]], scores[i]);
                BBox bbox = new BBox (
                        (int) (boxes[i][0] * LENGTH + minX),
                        (int) (boxes[i][1] * LENGTH + minY),
                        (int) (boxes[i][2] * LENGTH + minX),
                        (int) (boxes[i][3] * LENGTH + minY)
                );
                bbox.score = scores[i];
                bbox.label = labels[(int) classes[i]];
                objects.add(bbox);
            }
        }
        return objects;
    }

    public ArrayList<ArrayList<BBox>> detectInBatch(Tensor<UInt8> imageBatch, ArrayList<Integer> minXs, ArrayList<Integer> minYs) {
        List<Tensor<?>> outputs = null;
        int batch_size = minXs.size();
        outputs = session.runner()
                .feed("image_tensor", imageBatch)
                .fetch("detection_scores").fetch("detection_classes").fetch("detection_boxes")
                .run();
        ArrayList<ArrayList<BBox>> objectsList = new ArrayList<>();
        try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
             Tensor<Float> classesT = outputs.get(1).expect(Float.class);
             Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
            // All these tensors have:
            // - 1 as the first dimension
            // - maxObjects as the second dimension
            // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
            // This can be verified by looking at scoresT.shape() etc.
            int maxObjects = (int) scoresT.shape()[1];
            for (int i = 0; i < batch_size; i++) {
                ArrayList<BBox> objects = new ArrayList<>();
                float[] scores = scoresT.copyTo(new float[batch_size][maxObjects])[i];
                float[] classes = classesT.copyTo(new float[batch_size][maxObjects])[i];
                float[][] boxes = boxesT.copyTo(new float[batch_size][maxObjects][4])[i];
                // Print all objects whose score is at least 0.5.
                for (int j = 0; j < scores.length; ++j) {
                    if (scores[j] < 0.5) {
                        continue;
                    }
                    //System.out.printf("\tFound %-20s (score: %.4f)\n", labels[(int) classes[i]], scores[i]);
                    BBox bbox = new BBox(
                            (int) (boxes[j][0] * LENGTH + minXs.get(i)),
                            (int) (boxes[j][1] * LENGTH + minYs.get(i)),
                            (int) (boxes[j][2] * LENGTH + minXs.get(i)),
                            (int) (boxes[j][3] * LENGTH + minYs.get(i))
                    );
                    bbox.score = scores[j];
                    bbox.label = labels[(int) classes[j]];
                    objects.add(bbox);
                }
                objectsList.add(objects);
            }
        }
        return objectsList;

    }


    private static void printSignature(SavedModelBundle model) throws Exception {
        MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
        SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
        int numInputs = sig.getInputsCount();
        int i = 1;
        System.out.println("MODEL SIGNATURE");
        System.out.println("Inputs:");
        for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
            TensorInfo t = entry.getValue();
            System.out.printf(
                    "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
                    i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
        }
        int numOutputs = sig.getOutputsCount();
        i = 1;
        System.out.println("Outputs:");
        for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
            TensorInfo t = entry.getValue();
            System.out.printf(
                    "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
                    i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
        }
        System.out.println("-----------------------------------------------");
    }

    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }



    private static Tensor<UInt8> makeImageTensor(BufferedImage img) throws IOException {
        img = copyImage(img);
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[] {BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
    }

    private static Tensor<UInt8> makeImagesTensor(ArrayList<BufferedImage> images) throws IOException {
        final int CHANNELS = 3;
        int batch_size = images.size();
        byte[] data = new byte[batch_size * LENGTH * LENGTH * CHANNELS];
        for (int i = 0; i < batch_size; i++) {
            BufferedImage img = copyImage(images.get(i));
            byte[] singleData = ((DataBufferByte) img.getData().getDataBuffer()).getData();
            for (int j = 0; j < LENGTH * LENGTH * CHANNELS; j++) {
                data[ i * (LENGTH * LENGTH * CHANNELS) + j] = singleData[j];
            }
        }
        long[] shape = new long[] {batch_size, LENGTH, LENGTH, CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));

    }

    private static BufferedImage paddingImage(BufferedImage img, int width, int height) {
        BufferedImage dimg =new BufferedImage(width, height, img.getType());
        Graphics2D g = dimg.createGraphics();
        g.drawImage(img, null, 0, 0);
        return dimg;
    }

    private static BufferedImage copyImage(BufferedImage img) {
        BufferedImage dimg =new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
        Graphics2D g = dimg.createGraphics();
        g.drawImage(img, null, 0, 0);
        return dimg;
    }

    private static BufferedImage loadImage(String imagePath) throws Exception{
        BufferedImage img = ImageIO.read(new File(imagePath));
        if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new IOException(
                    String.format(
                            "Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
                            img.getType(), imagePath));
        }
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        bgr2rgb(data);
        return img;
    }

    private static void outputImage(String filename, BufferedImage img) throws Exception{

        FileOutputStream fos = new FileOutputStream(filename);
        JPEGImageEncoder encoder = JPEGCodec.createJPEGEncoder(fos);
        encoder.encode(img);
        fos.close();

    }
}