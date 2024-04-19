## About

This project is used to control 6DOF manipullator using arm detecting 2-cameras system in real time. In this project manipulator has 6 servos connected to raspberry pi pico, and pico connected to the computer. This code is a part of my uni's final project. Not every part of it can be understandable, you can find more info about it in the `final.pdf`. It is available only in polish tho (and it's overstricted in some ways by my university).

Overall flow of the program is to get frames from cameras, detect hand on these frames, estimate elbow using image processing, create and send data frame to manipulator, and that's it. `main.py` and `pico.py` are two separate programs, and you should run `main.py` at your main machine and `pico.py` at your microcontroller operating the manipulator.

## Computer program

When starting program you'd have cameras connected to your machine, and preferably UART or smth (although you can connect it later). I had cameras attached to a table, one at the leg, one at the top. So cameras are rotated in the program. As the background I used just plain black paper for better visibility of the arm. Overall detection process is too complicated to talk about, code is pretty fine with the terms of comments, though it's not always self-explanatory.

Hand is detected with MediaPipe hands model (it's pretty easy part). I then use canny edge detection and Hugh Line Transform to figure out direction of the forearm. Using golden ratio of human proportions I estimate it's length and thus elbow position.

When all landmarks are detected we delve into calculating angles between lines created from their endpoints. Like angle between upperarm and forearm. We get 6 angles (each for each degree of freedom in manipulator) and map them to servos' duties. For the next step we need a checksum to ensure that only correct data is used in the microcontroller, CRC-16 handles it. We create a data frame looking like `S [CRC-16] [duty-1] [duty-2] [duty-3] [duty-4] [duty-5] [duty-6] E`, example frame looks like `S FB6F 5000 5000 6000 5000 6000 8000 E`. This bad boy is then sent to the microcontroller via your selected output.

## Manipulator

When booting your manipulator with code from `pico/pico.py` it'll initialize all the core stuff and get to starting position. It'll wait for some fresh data on stdin and check if there're start and end signs, checksum, and body of the frame with exact 6 integers. If the frame is ok, we then proceed to moving part. Moves are divided into 40 steps to smoothen the movement. That's basically it.