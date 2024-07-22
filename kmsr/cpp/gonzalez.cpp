#include <algorithm>

#include "../header/gonzalez.h"
#include "../header/point.h"

using namespace std;

double gonzalezrmax(const vector<Point> &points, int k, int seed)
{
  srand(seed);
  vector<Point> centers;

  // Choose the first point randomly from the input points
  centers.push_back(points[rand() % points.size()]);

  vector<double> distances(points.size());

  // Calculate the maximum distance for the initial assignment
  for (size_t i = 0; i < points.size(); i++)
  {
    distances[i] = Point::distance(points[i], centers[0]);
  }

  // Iterate to find the remaining k - 1 centers
  while (static_cast<int>(centers.size()) < k)
  {
    // Choose the next center based on the maximum distance
    size_t nextCenterIndex = distance(
        distances.begin(), max_element(distances.begin(), distances.end()));
    Point nextCenter = points[nextCenterIndex];
    centers.push_back(nextCenter);

    // Update the distances for each point to its nearest center
    for (size_t i = 0; i < points.size(); i++)
    {
      distances[i] = min(distances[i], Point::distance(points[i], nextCenter));
    }
  }

  // Calculate the maximum distance of the points to their centers
  double maxDistance = 0.0;
  for (size_t i = 0; i < points.size(); i++)
  {
    maxDistance = max(maxDistance, distances[i]);
  }

  return maxDistance;
}
