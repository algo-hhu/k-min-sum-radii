# k-min-sum-radii

Befehl zum erstellen der Shared Library

```sh
g++ -fopenmp -shared -o clustering.so -fPIC kMSR/wrapper.cpp kMSR/k_MSR.cpp kMSR/point.cpp kMSR/gonzalez.cpp kMSR/ball.cpp kMSR/cluster.cpp kMSR/yildirim.cpp kMSR/welzl.cpp kMSR/heuristic.cpp kMSR/util.cpp
```