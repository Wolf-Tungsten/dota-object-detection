import java.lang.reflect.Array;
import java.util.*;

public class BBox {

    public static double THRESHOLD = 0.75;

    public int minX;
    public int minY;
    public int maxX;
    public int maxY;
    public double score;
    public String label;


    public BBox(int minX, int minY, int maxX, int maxY) {
        this.minX = minX;
        this.minY = minY;
        this.maxX = maxX;
        this.maxY = maxY;
    }

    public int getArea() {
        return ( maxX - minX ) * ( maxY - minX);
    }

    public static double unionScore( BBox bbox1, BBox bbox2) {
        int unionMinX = bbox1.minX > bbox2.minX ? bbox1.minX : bbox2.minX;
        int unionMinY = bbox1.minY > bbox2.minY ? bbox1.minY : bbox2.minY;
        int unionMaxX = bbox1.maxX < bbox2.maxY ? bbox1.maxX : bbox2.maxX;
        int unionMaxY = bbox1.maxY < bbox2.maxX ? bbox1.maxY : bbox2.maxY;

        if ( unionMaxX > unionMinX && unionMaxY > unionMinY ) {
            BBox unionPart = new BBox( unionMinX, unionMinY, unionMaxX, unionMaxY );
            double maxArea = bbox1.getArea() > bbox2.getArea() ? bbox1.getArea() : bbox2.getArea();
            return unionPart.getArea() / maxArea;
        } else {
            return 0;
        }
    }

    public static ArrayList<BBox> bBoxFilter(ArrayList<BBox> bBoxes) {
        ArrayList<Integer> toRemove = new ArrayList<Integer>();
        boolean changed = true;
        while (changed) {
            changed = false;
            toRemove.clear();
            for (int i = 0; i < bBoxes.size() - 1; i++) {
                for (int j = i + 1; j < bBoxes.size(); j++) {
                    if (unionScore(bBoxes.get(i), bBoxes.get(j)) > THRESHOLD) {
                        changed = true;
                        toRemove.add(bBoxes.get(i).getArea() > bBoxes.get(j).getArea() ? i : j);
                    }
                }
            }
            if (changed) {
                for (int i = 0; i < toRemove.size(); i++) {
                    bBoxes.remove(toRemove.get(i));
                }
            }
        }
        return bBoxes;
    }
}
