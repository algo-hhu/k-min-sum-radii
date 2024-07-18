#include <Python.h>

#include <iostream>

#include "header/heuristic.h"
#include "header/k_MSR.h"
#include "header/point.h"
#include "header/welzl.h"

using namespace std;

vector<Point> arrayToVector(double *array, int numPoints, int dimension)
{
  vector<Point> points;
  for (int i = 0; i < numPoints; i++)
  {
    vector<double> coordinates;
    for (int j = 0; j < dimension; j++)
    {
      coordinates.push_back(array[i * dimension + j]);
    }
    points.push_back(Point(coordinates, i));
  }
  return points;
}

int exportCluster(vector<Cluster> clusters, int *labels, double *centers, double *radii, int dimensions)
{
  int numClusters = 0;

  for (size_t i = 0; i < clusters.size(); i++)
  {
    for (size_t j = 0; j < clusters[i].getPoints().size(); j++)
    {
      labels[clusters[i].getPoints()[j].getPosition()] = i;
    }
    // If the cluster actually has points we
    // - Consider it a cluster
    // - Add the center and radius to the output arrays
    if (clusters[i].getPoints().size() > 0)
    {
      numClusters++;

      Ball ball = findMinEnclosingBall(clusters[i].getPoints());

      for (size_t j = 0; j < ball.getCenter().getCoordinates().size(); j++)
      {
        centers[i * dimensions + j] = ball.getCenter().getCoordinates()[j];
      }

      radii[i] = ball.getRadius();
    }
  }
  return numClusters;
}

extern "C"
{

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  double schmidt_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      double epsilon,
      int numUVectors,
      int numRadiiVectors,
      int *numClusters,
      int *labels,
      double *centers,
      double *radii)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);
    vector<Cluster> bestCluster(k);

    double cost =
        clustering(points, k, epsilon, numUVectors, numRadiiVectors, bestCluster);

    *numClusters = exportCluster(bestCluster, labels, centers, radii, dimension);

    return cost;
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  double heuristic_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters,
      int *labels,
      double *centers,
      double *radii)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = heuristik(points, k);

    *numClusters = exportCluster(cluster, labels, centers, radii, dimension);

    return cost(cluster);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  double gonzales_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters,
      int *labels,
      double *centers,
      double *radii)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = gonzales(points, k);

    *numClusters = exportCluster(cluster, labels, centers, radii, dimension);

    return cost(cluster);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  double kmeans_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters,
      int *labels,
      double *centers,
      double *radii)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = kMeansPlusPlus(points, k);

    *numClusters = exportCluster(cluster, labels, centers, radii, dimension);

    return cost(cluster);
  }

} // extern "C"

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _coremodule = {
    PyModuleDef_HEAD_INIT,
    "kmsr._core",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__core(void)
{
  return PyModule_Create(&_coremodule);
}
