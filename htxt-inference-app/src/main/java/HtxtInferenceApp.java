import java.nio.file.Paths;
import java.util.ArrayList;

public class HtxtInferenceApp {

    public static void main(String[] args) throws Exception {
        System.out.println("hello htxt");
        String testImagePath = Paths.get(DetectionCore.class.getClassLoader().getResource("1.tif").getPath().substring(1)).toString();
        DetectionCore detectionCore = new DetectionCore();
        ArrayList<BBox> result = detectionCore.detect(testImagePath);
        for (BBox bbox:result) {
            System.out.printf("发现 - %s - 在 {(%d, %d),(%d, %d)}\n", bbox.label,
                    bbox.minX, bbox.minY, bbox.maxX, bbox.maxY);
        }
    }
}
