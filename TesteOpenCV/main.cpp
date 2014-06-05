//
//  main.cpp
//  TesteOpenCV
//
//  Created by Sasha Nicolas Da Rocha Pinheiro on 5/30/14.
//  Copyright (c) 2014 Sasha Nicolas Da Rocha Pinheiro. All rights reserved.
//

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

int main()
{
    cv::VideoCapture cap;
    cap.open(0);

    if( !cap.isOpened() )
    {
        std::cerr << "***Could not initialize capturing...***\n";
        std::cerr << "Current parameter's value: \n";
        return -1;
    }

    cv::Mat frame;
    while(1){
        cap >> frame;
        if(frame.empty()){
            std::cerr<<"frame is empty"<<std::endl;
            break;
        }

        cv::imshow("", frame);
        cv::waitKey(10);
    }

    return 1;
}