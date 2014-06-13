//
//  main.cpp
//  TesteOpenCV
//
//  Created by Sasha Nicolas Da Rocha Pinheiro on 5/30/14.
//  Copyright (c) 2014 Sasha Nicolas Da Rocha Pinheiro. All rights reserved.
//

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int estado = 0; //0=calibra, 1=detecta
CvCapture* capture = cvCreateCameraCapture(0);

class CalibraCamera{
public:
    int n_boards, board_dt, board_w, board_h, board_n;
    CvSize board_sz;
    CvMat* image_points, *object_points, *point_counts, *intrinsic_matrix, *distortion_coeffs;
    
    CalibraCamera(int w, int h, int n){
        board_w = w; // Board width in squares
        board_h = h; // Board height
        board_dt = 20;
        n_boards = n; // Number of boards to find
        board_n = board_w * board_h; //number of squares
        board_sz = cvSize( board_w, board_h );
        
        image_points = cvCreateMat( n_boards*board_n, 2, CV_32FC1 );
        object_points = cvCreateMat( n_boards*board_n, 3, CV_32FC1 );
        point_counts = cvCreateMat( n_boards, 1, CV_32SC1 );
        intrinsic_matrix = cvCreateMat( 3, 3, CV_32FC1 );
        distortion_coeffs = cvCreateMat( 5, 1, CV_32FC1 );
    }
    
    void Calibrate(){
        CvPoint2D32f* corners = new CvPoint2D32f[ board_n ];
        int corner_count;
        int successes = 0;
        int step, frame = 0;
        
//        IplImage *image = cvQueryFrame( capture );
        
        IplImage *image = cvQueryFrame( capture );
        
        cvSetImageROI(image, cvRect(300, 300, 600, 600));
        IplImage *tmp = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
        cvCopy(image, tmp, NULL);
        cvResetImageROI(image);
        image = cvCloneImage(tmp);
        
        IplImage *gray_image = cvCreateImage( cvGetSize( image ), 8, 1 );
        
        // Capture Corner views loop until we've got n_boards
        // successful captures (all corners on the board are found)
        
        while( successes < n_boards ){
            // Skp every board_dt frames to allow user to move chessboard
            if( frame++ % board_dt == 0 ){
                // Find chessboard corners:
                int found = cvFindChessboardCorners( image, board_sz, corners,
                                                    &corner_count, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );
                
                // Get subpixel accuracy on those corners
                cvCvtColor( image, gray_image, CV_BGR2GRAY );
                cvFindCornerSubPix( gray_image, corners, corner_count, cvSize( 11, 11 ),
                                   cvSize( -1, -1 ), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
                
                // Draw it
                cvDrawChessboardCorners( image, board_sz, corners, corner_count, found );
                cvShowImage( "Calibration", image );
                
                // If we got a good board, add it to our data
                if( corner_count == board_n ){
                    step = successes*board_n;
                    for( int i=step, j=0; j < board_n; ++i, ++j ){
                        CV_MAT_ELEM( *image_points, float, i, 0 ) = corners[j].x;
                        CV_MAT_ELEM( *image_points, float, i, 1 ) = corners[j].y;
                        CV_MAT_ELEM( *object_points, float, i, 0 ) = j/board_w;
                        CV_MAT_ELEM( *object_points, float, i, 1 ) = j%board_w;
                        CV_MAT_ELEM( *object_points, float, i, 2 ) = 0.0f;
                    }
                    CV_MAT_ELEM( *point_counts, int, successes, 0 ) = board_n;
                    successes++;
                }
            }
            
            // Handle pause/unpause and ESC
            int c = cvWaitKey( 15 );
            if( c == 'p' ){
                c = 0;
                while( c != 'p' && c != 27 ){
                    c = cvWaitKey( 250 );
                }
            }
//            image = cvQueryFrame( capture ); // Get next image
            image = cvQueryFrame( capture );
            cvSetImageROI(image, cvRect(300, 300, 600, 600));
            IplImage *tmp = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
            cvCopy(image, tmp, NULL);
            cvResetImageROI(image);
            image = cvCloneImage(tmp);
        } // End collection while loop
        
        // Allocate matrices according to how many chessboards found
        CvMat* object_points2 = cvCreateMat( successes*board_n, 3, CV_32FC1 );
        CvMat* image_points2 = cvCreateMat( successes*board_n, 2, CV_32FC1 );
        CvMat* point_counts2 = cvCreateMat( successes, 1, CV_32SC1 );
        
        // Transfer the points into the correct size matrices
        for( int i = 0; i < successes*board_n; ++i ){
            CV_MAT_ELEM( *image_points2, float, i, 0) = CV_MAT_ELEM( *image_points, float, i, 0 );
            CV_MAT_ELEM( *image_points2, float, i, 1) = CV_MAT_ELEM( *image_points, float, i, 1 );
            CV_MAT_ELEM( *object_points2, float, i, 0) = CV_MAT_ELEM( *object_points, float, i, 0 );
            CV_MAT_ELEM( *object_points2, float, i, 1) = CV_MAT_ELEM( *object_points, float, i, 1 );
            CV_MAT_ELEM( *object_points2, float, i, 2) = CV_MAT_ELEM( *object_points, float, i, 2 );
        }
        
        for( int i=0; i < successes; ++i ){
            CV_MAT_ELEM( *point_counts2, int, i, 0 ) = CV_MAT_ELEM( *point_counts, int, i, 0 );
        }
        cvReleaseMat( &object_points );
        cvReleaseMat( &image_points );
        cvReleaseMat( &point_counts );
        
        // At this point we have all the chessboard corners we need
        // Initiliazie the intrinsic matrix such that the two focal lengths
        // have a ratio of 1.0
        
        CV_MAT_ELEM( *intrinsic_matrix, float, 0, 0 ) = 1.0;
        CV_MAT_ELEM( *intrinsic_matrix, float, 1, 1 ) = 1.0;
        
        // Calibrate the camera
        cvCalibrateCamera2( object_points2, image_points2, point_counts2, cvGetSize( image ),
                           intrinsic_matrix, distortion_coeffs, NULL, NULL, CV_CALIB_FIX_ASPECT_RATIO );
        
        // Save the intrinsics and distortions
        cvSave( "sashaintri.xml", intrinsic_matrix );
        cvSave( "sashadistor.xml", distortion_coeffs );

    }
};

class DetectaPadrao{
public:
    
};


int main(){
    
    cv::VideoCapture cap;
    cap.open(0);
    
    if( !cap.isOpened() )
    {
        std::cerr << "***Could not initialize capturing...***\n";
        std::cerr << "Current parameter's value: \n";
        return -1;
    }
    
    CalibraCamera cc(5,4,8);
    cc.Calibrate();
    DetectaPadrao dp;
    
    
    cv::Mat frame;
    while(1){
        cap >> frame;
        if(frame.empty()){
            std::cerr<<"frame is empty"<<std::endl;
            break;
        }
        
        if(estado == 0){
            
        }
        
        cv::imshow("", frame);
        
        
        cv::waitKey(10);
    }
    
    return 1;
}