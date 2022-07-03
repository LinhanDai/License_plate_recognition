#License Plate Recognition

1. Compile the project

    1.1 run "make build" , and "cd build" 
    
    1.2 "cmake.." and "make -j*"

2. Use compiled library
   
   After compiling, libocr.so will be generated in the build folder,For the different compiling dependent environment, 
   you need to copy the compiled "libocr.so" to the actual project directory for use
   
3. Modify the configuration file

   The performance and speed of the algorithm can be adjusted by modifying the parameters in the two 
   folders configs/yolov4-licence.cfg and ocr.yaml