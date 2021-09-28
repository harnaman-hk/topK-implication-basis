For running the algorithm for stopping at the size of k+1,
two additional parameters are given, the filename, and any negative integer.

for eg.
./algo connect.txt 0.1 0.1 strong frequent 8 none connect/connect_topk_10_80.txt -1


For running the algorithm for calculating the precision and recall at some points of time,along with the file name, the time points are passed as arguments
for eg.
./algo connect.txt 0.1 0.1 strong frequent 8 none connect/connect_topk_10_80.txt 1 5 10 30 


For running the algorithm till completion, only filename is passed as additional arguments
 eg.
./algo connect.txt 0.1 0.1 strong frequent 8 none connect/connect_topk_10_80.txt 
