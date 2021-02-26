#!/bin/bash

echo "start" > uptime-out1.txt
echo "start" > top-out1.txt

while true
do
sleep 2
echo `uptime` >> uptime-out.txt
echo `top -bcn1 -u fos -w512` >> top-out1.txt 
done
