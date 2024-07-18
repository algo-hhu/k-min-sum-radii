#include "header/heuristic.h"
#include "header/k_MSR.h"
#include "header/point.h"

using namespace std;

struct PointData {
  double* coordinates;
  int dimension;
};

struct ClusterData {
  PointData* points;
  int numPoints;
};

vector<Point> arrayToVector(double* array, int numPoints, int dimension) {
  vector<Point> points;
  for (int i = 0; i < numPoints; i++) {
    vector<double> coordinates;
    for (int j = 0; j < dimension; j++) {
      coordinates.push_back(array[i * dimension + j]);
    }
    points.push_back(Point(coordinates));
  }
  return points;
}

ClusterData* clusterToArray(vector<Cluster> clusters, int* numClusters) {
  *numClusters = clusters.size();

  ClusterData* clusterData = new ClusterData[*numClusters];
  for (int i = 0; i < *numClusters; i++) {
    const vector<Point>& clusterPoints = clusters[i].getPoints();
    clusterData[i].numPoints = clusterPoints.size();
    clusterData[i].points = new PointData[clusterPoints.size()];
    for (int j = 0; j < clusterPoints.size(); j++) {
      const vector<double>& coords = clusterPoints[j].getCoordinates();
      clusterData[i].points[j].dimension = coords.size();
      clusterData[i].points[j].coordinates = new double[coords.size()];
      for (int k = 0; k < coords.size(); k++) {
        clusterData[i].points[j].coordinates[k] = coords[k];
      }
    }
  }
  return clusterData;
}

extern "C" {

ClusterData* schmidt_wrapper(double* pointArray, int numPoints, int dimension,
                             int k, double epsilon, int numUVectors,
                             int numRadiiVectors, int* numClusters) {
  vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

  vector<Cluster> cluster =
      clustering(points, k, epsilon, numUVectors, numRadiiVectors);

  return clusterToArray(cluster, numClusters);
}

ClusterData* heuristic_wrapper(double* pointArray, int numPoints, int dimension,
                               int k, int* numClusters) {
  vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

  vector<Cluster> cluster = heuristik(points, k);

  return clusterToArray(cluster, numClusters);
}

ClusterData* gonzales_wrapper(double* pointArray, int numPoints, int dimension,
                              int k, int* numClusters) {
  vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

  vector<Cluster> cluster = gonzales(points, k);

  return clusterToArray(cluster, numClusters);
}

ClusterData* kmeans_wrapper(double* pointArray, int numPoints, int dimension,
                            int k, int* numClusters) {
  vector<Point> points = arrayToVector(pointArray, numPoints, dimension);

  vector<Cluster> cluster = kMeansPlusPlus(points, k);

  return clusterToArray(cluster, numClusters);
}

void free_cluster_data(ClusterData* clusterData, int numClusters) {
  for (int i = 0; i < numClusters; i++) {
    for (int j = 0; j < clusterData[i].numPoints; j++) {
      delete[] clusterData[i].points[j].coordinates;
    }
    delete[] clusterData[i].points;
  }
  delete[] clusterData;
}
}
