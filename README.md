# Google Street View Demography

We collected 14000 geolocations from IndianaPolis area tagged with their demographic characteristics. The three characteristics we worked on are income level, poverty level and crime level, each between 1 and 7.

For each geolocations we collected 4 google streetview images facing north, west, south and east each with the dimension 256x256, which results in more than 60K images to train on.

We trained a network similar to Alexnet but with three parallel output with loss layers to get the result of 3 classification results in a single net. We could reach upto accuracy of 55% at this moment.
