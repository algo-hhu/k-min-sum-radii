// heuristic.h

#ifndef HEURISTIC_H
#define HEURISTIC_H

#include "ball.h"
#include "cluster.h"
#include "point.h"

std::vector<Cluster> gonzales(std::vector<Point> &points, int k, int seed);
std::vector<Cluster> kMeansPlusPlus(std::vector<Point> &points, int k, int seed);
std::vector<Cluster> heuristik(std::vector<Point> &points, int k);

#endif  // HEURISTIC_H