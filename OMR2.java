import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

public class OMR2 {
    static {
        File file = new File("D:/hihih1/opencv/build/java/x64/opencv_java480.dll");
        System.load(file.getAbsolutePath());
    }

    public static int getX(Rect rect) {
        return rect.x;
    }

    public static int getY(Rect rect) {
        return rect.y;
    }

    public static int getH(Rect rect) {
        return rect.height;
    }

    public static int getXVer1(MatOfPoint contour) {
        Rect boundingRect = Imgproc.boundingRect(contour);
        return boundingRect.x * boundingRect.width;
    }

    public static List<Mat> cropImage(Mat img) {
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(imgGray, imgGray, new org.opencv.core.Size(5, 5), 0);

        // dò canh
        Mat edges = new Mat();
        Imgproc.Canny(imgGray, edges, 50, 150);

        // tìm đường viền
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Mat> ansBlocks = new ArrayList<>();
        int xOld = 0, yOld = 0, wOld = 0, hOld = 0;
        if (contours.size() > 0) {
            Collections.sort(contours, Comparator.comparingInt(OMR2::getXVer1));

            for (MatOfPoint contour : contours) {
                Rect rect = Imgproc.boundingRect(contour);
                int xCurr = rect.x;
                int yCurr = rect.y;
                int wCurr = rect.width;
                int hCurr = rect.height;

                if (wCurr * hCurr >= 944505) {
                    int checkXYMin = xCurr * yCurr - xOld * yOld;
                    int checkXYMax = (xCurr + wCurr) * (yCurr + hCurr) - (xOld + wOld) * (yOld + hOld);

                    if (ansBlocks.isEmpty() || (checkXYMin > 20000 && checkXYMax > 20000)) {
                        Mat croppedImage = new Mat(img, new Rect(xCurr, yCurr, wCurr, hCurr));
                        ansBlocks.add(croppedImage);

                        xOld = xCurr;
                        yOld = yCurr;
                        wOld = wCurr;
                        hOld = hCurr;
                    }
                }

            }

            Collections.sort(contours, new Comparator<MatOfPoint>() {
                @Override
                public int compare(MatOfPoint contour1, MatOfPoint contour2) {
                    int x1 = Imgproc.boundingRect(contour1).x;
                    int x2 = Imgproc.boundingRect(contour2).x;
                    return Integer.compare(x1, x2);
                }
            });
        }
        return ansBlocks;

    }

    public static Mat combineImages(List<Mat> images) {
        // // Tìm tổng chiều rộng và chiều cao tối đa trong số tất cả các hình ảnh
        int totalWidth = 0;
        int maxHeight = 0;
        for (Mat image : images) {
            int width = image.width();
            int height = image.height();

            totalWidth += width;
            maxHeight = Math.max(maxHeight, height);
        }

        // Tạo một Mat mới để chứa ảnh đã kết hợp
        Mat combinedImage = Mat.zeros(maxHeight, totalWidth, CvType.CV_8UC3);

        // Copy từng ảnh đã crop vào ảnh ghép
        int x = 0;
        for (Mat image : images) {
            int width = image.width();

            Mat roi = combinedImage.submat(0, image.height(), x, x + width);
            image.copyTo(roi);

            x += width;
        }

        return combinedImage;
    }

    public static List<Mat> processAnsBlocks(List<Mat> ansBlocks) {
        List<Mat> listAnswers = new ArrayList<>();

        for (Mat ansBlock : ansBlocks) {
            int offset1 = (int) Math.ceil(ansBlock.rows() / 6);

            for (int i = 0; i < 6; i++) {
                Mat boxImg = new Mat(ansBlock, new Rect(0, i * offset1, ansBlock.cols(), offset1));
                int heightBox = boxImg.rows();

                boxImg = boxImg.submat(14, heightBox - 14, 0, boxImg.cols());
                int offset2 = (int) Math.ceil(boxImg.rows() / 5);

                for (int j = 0; j < 5; j++) {
                    listAnswers.add(boxImg.submat(j * offset2, (j + 1) * offset2, 0, boxImg.cols()));
                }
            }
        }

        return listAnswers;
    }

    public static List<Mat> processListAns(List<Mat> listAnswers) {
        List<Mat> listChoices = new ArrayList<>();
        int offset = 80;
        int start = 120;

        for (Mat answerImg : listAnswers) {
            for (int i = 0; i < 4; i++) {
                Mat bubbleChoice = new Mat(answerImg,
                        new org.opencv.core.Rect(start + i * offset, 0, offset, answerImg.rows()));

                // Convert to grayscale
                Imgproc.cvtColor(bubbleChoice, bubbleChoice, Imgproc.COLOR_BGR2GRAY);

                Imgproc.threshold(bubbleChoice, bubbleChoice, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);

                Imgproc.resize(bubbleChoice, bubbleChoice, new org.opencv.core.Size(28, 28));
                bubbleChoice = bubbleChoice.reshape(1); // Reshape to a single channel Mat
                listChoices.add(bubbleChoice);
            }
        }

        if (listChoices.size() != 480) {
            throw new IllegalArgumentException("Length of listChoices must be 480");
        }

        return listChoices;
    }

    public static String mapAnswer(int idx) {
        String answerCircle;
        if (idx % 4 == 0) {
            answerCircle = "A";
        } else if (idx % 4 == 1) {
            answerCircle = "B";
        } else if (idx % 4 == 2) {
            answerCircle = "C";
        } else {
            answerCircle = "D";
        }
        return answerCircle;
    }
    

    public static void main(String[] args) {
        OMR2 omr2 = new OMR2();
        Mat img = Imgcodecs.imread("test7.jpg");
        List<Mat> croppedImages = omr2.cropImage(img);
        List<Mat> listAns = omr2.processAnsBlocks(croppedImages);
        List<Mat> processListAns = omr2.processListAns(listAns);
        List answeredStudents = new ArrayList<>();
        for (int i = 0; i < processListAns.size(); i++) {
            Mat ans = processListAns.get(i);
            int tes = Core.countNonZero(ans);
            String mappedAnswer = mapAnswer(i);
            if(tes > 100){
                answeredStudents.add(ans);
                System.out.println("Câu trả lời " + (i /4 +1) + ": " + mappedAnswer);
                System.out.println(tes);
            }
        }
        
    }
}