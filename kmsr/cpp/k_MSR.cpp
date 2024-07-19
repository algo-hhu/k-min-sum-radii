#include <omp.h>

#include <random>

#include "../header/k_MSR.h"
#include "../header/util.h"
#include "../header/gonzales.h"
#include "../header/heuristic.h"
#include "../header/yildirim.h"

using namespace std;

// Calculates a vector of vectors of radii for a given maximum radius, number of balls k, and accuracy epsilon.
vector<vector<double>> getRadii(double rmax, int k, double epsilon)
{
  vector<vector<double>> result;
  vector<int> indices(k - 1, 0);
  vector<double> set;

  // Calculate the number of radii needed to ensure sufficient coverage
  int limit = ceil(logBase((k / epsilon), (1 + epsilon)));

  // Create the set of radii whose permutations will be formed.
  for (int i = 0; i <= limit; i++)
  {
    set.push_back(pow((1 + epsilon), i) * (epsilon / k) * rmax);
  }

  // Create all possible permutations of 'set' with 'rmax' as the first element.
  while (true)
  {
    vector<double> current;

    // The maximum radius is always added as the first radius in the combination
    current.push_back(rmax);

    for (int idx : indices)
    {
      current.push_back(set[idx]);
    }
    result.push_back(current);

    int next = k - 2;
    while (next >= 0 && ++indices[next] == static_cast<int>(set.size()))
    {
      indices[next] = 0;
      next--;
    }
    if (next < 0)
    {
      break;
    }
  }

  return result;
}

vector<vector<double>> getRandomRadii(double rmax, int k, double epsilon,
                                      int numRadiiVectors, int seed)
{
  vector<double> set;

  // Calculate the number of radii needed to ensure sufficient coverage
  int limit = ceil(logBase((k / epsilon), (1 + epsilon)));

  // Create the set of radii whose permutations will be formed.
  for (int i = 0; i <= limit; i++)
  {
    set.push_back(pow((1 + epsilon), i) * (epsilon / k) * rmax);
  }

  vector<vector<double>> result(numRadiiVectors);

  // Initialize a Mersenne Twister generator with the seed of 'rd'.
  mt19937 gen(seed);

  // Define a uniform distribution for integers between 0 and set.size()-1.
  uniform_int_distribution<> distrib(0, set.size() - 1);

  // Generate numVectors number of vectors.
  for (int i = 0; i < numRadiiVectors; i++)
  {
    vector<double> currentVector(k);
    currentVector[0] = rmax;

    // Fill the vector with random values determined by the random number generator.
    for (size_t j = 1; j < currentVector.size(); j++)
    {
      currentVector[j] = set[distrib(gen)];
    }
    result[i] = currentVector;
  }

  return result;
}

// Generates a list of vectors, each containing random integers between 0 and k-1.
vector<vector<int>> getU(int n, int k, double epsilon, int numUVectors, int seed)
{
  // Calculate the length of each vector based on the given parameters k and epsilon.
  int length =
      min(n, static_cast<int>((32 * k * (1 + epsilon)) / (pow(epsilon, 3))));

  vector<vector<int>> result(numUVectors);

  // Initialize a Mersenne Twister generator with the seed of 'rd'.
  mt19937 gen(seed);

  // Define a uniform distribution for integers between 0 and k-1.
  uniform_int_distribution<> distrib(0, k - 1);

  // Generate numVectors number of vectors.
  for (int i = 0; i < numUVectors; i++)
  {
    vector<int> currentVector(length);

    // Fill the vector with random values determined by the random number generator.
    for (int &value : currentVector)
    {
      value = distrib(gen);
    }
    result[i] = currentVector;
  }

  return result;
}

// Creates 'k' balls that contain all the given points.
vector<Ball> selection(const vector<Point> &points, int k, const vector<int> &u,
                       const vector<double> &radii, double epsilon)
{
  vector<Ball> balls(k, Ball(points.front().getCoordinates().size()));
  vector<vector<Point>> Si(k);
  double lambda = 1 + epsilon + 2 * sqrt(epsilon);

  for (size_t i = 0; i < u.size(); i++)
  {
    bool addedPoint = false;

    // Add the first point in 'points' that is not contained in 'X' or 'R' to 'S_ui'.
    for (Point p : points)
    {
      if (!containsPoint(p, balls))
      {
        Si[u[i]].push_back(p);
        addedPoint = true;
        break;
      }
    }

    // If no point was added, abort the process and return the balls.
    if (!addedPoint)
    {
      return balls;
    }

    // If the size of 'S_ui' is greater than or equal to 2, find the ball that encloses all points in 'S_ui' and increase its radius by the factor Lambda.
    if (Si[u[i]].size() >= 2)
    {
      Ball b = findMEB(Si[u[i]], epsilon);
      b.setRadius(b.getRadius() * lambda);
      balls[u[i]] = b;
    }
    else
    {
      balls[u[i]] = Ball(Si[u[i]][0], (epsilon / (1 + epsilon)) * radii[u[i]]);
    }
  }
  return balls;
}

// Main function that calculates the clusters.
vector<Cluster> clustering(const vector<Point> &points, int k, double epsilon,
                  int numUVectors, int numRadiiVectors, int seed)
{
  vector<Cluster> bestCluster(k);
  double rmax = gonzalesrmax(points, k, seed);

  // Calculate the radii and u values based on 'rmax', 'k', and 'epsilon'.
  vector<vector<double>> radii =
      getRandomRadii(rmax, k, epsilon, numRadiiVectors, seed);
  vector<vector<int>> u = getU(points.size(), k, epsilon, numUVectors, seed);

  // Initialize the 'bestCluster' by making all points part of a cluster.
  bestCluster[0].setPoints(points);
  double bestCost = cost(bestCluster);

#pragma omp parallel for collapse(2) schedule(dynamic) \
    shared(bestCluster, bestCost)
  // Calculate the clusters with the lowest cost for all combinations of 'radii' and 'u'.
  for (size_t i = 0; i < radii.size(); i++)
  {
    for (size_t j = 0; j < u.size(); j++)
    {
      vector<double> r = radii[i];
      vector<int> ui = u[j];

      // Calculate balls based on the radii and 'u' values.
      vector<Ball> localBalls = selection(points, k, ui, r, epsilon);

      // Check if all points are covered by the selected balls.
      if (containsAllPoints(points, localBalls))
      {
        // Create clusters based on the selected balls.
        vector<Cluster> localCluster(k);
        for (Point p : points)
        {
          for (int c = 0; c < k; c++)
          {
            if (localBalls[c].contains(p))
            {
              localCluster[c].addPoint(p);
              break;
            }
          }
        }

        // localCluster = mergeCluster(localCluster);
        // Calculate the cost for the local clusters.
        double localCost = cost(localCluster);

#pragma omp critical
        {
          // Update the best cluster and cost value if a better one is found.
          if (localCost < bestCost)
          {
            bestCost = localCost;
            bestCluster = localCluster;
          }
        }
      }
    }
  }

  return mergeCluster(bestCluster);
}
