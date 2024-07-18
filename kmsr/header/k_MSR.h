// k_MSR.h

#ifndef K_MSR_H
#define K_MSR_H

#include "ball.h"
#include "cluster.h"
#include "point.h"

bool containsPoint(const Point& p, const std::vector<Ball>& balls);

std::vector<Ball> selection(const std::vector<Point>& points, int k,
                            const std::vector<int>& u,
                            const std::vector<double>& radii, double epsilon);

std::vector<std::vector<double>> getRadii(double rmax, int k, double epsilon);

std::vector<std::vector<int>> getU(int n, int k, double epsilon,
                                   int numUVectors, int seed);

std::vector<Cluster> clustering(const std::vector<Point>& points, int k,
                                double epsilon, int numUVectors,
                                int numRadiiVectors, int seed);

std::vector<std::vector<double>> getRandomRadii(double rmax, int k,
                                                double epsilon,
                                                int numRadiiVectors, int seed);
#endif  // K_MSR_H
