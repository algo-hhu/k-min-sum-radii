#include <Python.h>

#include "header/heuristic.h"
#include "header/k_MSR.h"
#include "header/point.h"

using namespace std;

struct PointData
{
  double *coordinates;
  int dimension;
};

struct ClusterData
{
  PointData *points;
  int numPoints;
};

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
    points.push_back(Point(coordinates));
  }
  return points;
}

ClusterData *clusterToArray(vector<Cluster> clusters, int *numClusters)
{
  *numClusters = clusters.size();

  ClusterData *clusterData = new ClusterData[*numClusters];
  for (int i = 0; i < *numClusters; i++)
  {
    const vector<Point> &clusterPoints = clusters[i].getPoints();
    clusterData[i].numPoints = clusterPoints.size();
    clusterData[i].points = new PointData[clusterPoints.size()];
    for (size_t j = 0; j < clusterPoints.size(); j++)
    {
      const vector<double> &coords = clusterPoints[j].getCoordinates();
      clusterData[i].points[j].dimension = coords.size();
      clusterData[i].points[j].coordinates = new double[coords.size()];
      for (size_t k = 0; k < coords.size(); k++)
      {
        clusterData[i].points[j].coordinates[k] = coords[k];
      }
    }
  }
  return clusterData;
}

extern "C"
{

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  ClusterData *
  schmidt_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      double epsilon,
      int numUVectors,
      int numRadiiVectors,
      int *numClusters)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster =
        clustering(points, k, epsilon, numUVectors, numRadiiVectors);

    return clusterToArray(cluster, numClusters);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  ClusterData *
  heuristic_wrapper(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = heuristik(points, k);

    return clusterToArray(cluster, numClusters);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  ClusterData *
  runGonzales(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = gonzales(points, k);

    return clusterToArray(cluster, numClusters);
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  __declspec(dllexport)
#endif
  ClusterData *
  runKMeans(
      double *pointArray,
      int numPoints,
      int dimension,
      int k,
      int *numClusters)
  {
    vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

    vector<Cluster> cluster = kMeansPlusPlus(points, k);

    return clusterToArray(cluster, numClusters);
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
