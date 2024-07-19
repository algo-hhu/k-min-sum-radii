#include <cmath>
#include <vector>

#include "../header/ball.h"
#include "../header/miniball.h"

using namespace std;

// Function object to map a Point iterator to the corresponding
// coordinate iterator
struct PointCoordAccessor
{
  typedef vector<Point>::const_iterator Pit;
  typedef vector<double>::const_iterator Cit;

  inline Cit operator()(Pit it) const { return it->getCoordinates().begin(); }
};

Ball findMinEnclosingBall(const vector<Point> &points)
{
  int dimension = static_cast<int>(points.front().getCoordinates().size());

  Miniball::Miniball<PointCoordAccessor> mb(dimension, points.begin(),
                                            points.end());

  // Get the center and radius of the computed Minimum Enclosing Ball
  const double *center_coords = mb.center();
  double radius = sqrt(mb.squared_radius());

  // Convert the center to a Point object
  vector<double> center_vector(center_coords, center_coords + dimension);
  Point center_point(center_vector);

  // Create and return the Ball
  return Ball(center_point, radius);
}
