#include <Python.h>

#include <iostream>

#include "header/heuristic.h"
#include "header/k_MSR.h"
#include "header/point.h"

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

void exportCluster(vector<Cluster> clusters, int *numClusters, int *labels, double *centers)
{
  *numClusters = 0;

  for (int i = 0; i < *numClusters; i++)
  {
    for (size_t j = 0; j < clusters[i].getPoints().size(); j++)
    {
      labels[clusters[i].getPoints()[j].getPosition()] = i;
    }
    if (clusters[i].getPoints().size() > 0)
    {
      *numClusters++;
    }
  }
}

extern "C"
{

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  void schmidt_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      double epsilon,
      int numUVectors,
      int numRadiiVectors,
      int *numClusters,
      int *labels,
      double *centers)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster =
        clustering(points, k, epsilon, numUVectors, numRadiiVectors);

    exportCluster(cluster, numClusters, labels, centers);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  void heuristic_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters,
      int *labels,
      double *centers)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = heuristik(points, k);

    exportCluster(cluster, numClusters, labels, centers);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  void gonzales_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters,
      int *labels,
      double *centers)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = gonzales(points, k);

    exportCluster(cluster, numClusters, labels, centers);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  void kmeans_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters,
      int *labels,
      double *centers)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = kMeansPlusPlus(points, k);

    exportCluster(cluster, numClusters, labels, centers);
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
