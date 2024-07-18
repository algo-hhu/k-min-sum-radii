#include <cmath>
#include <vector>

#include "../header/ball.h"
#include "../header/miniball.h"

using namespace std;

// Funktionsobjekt, um einen Point-Iterator auf den entsprechenden
// Koordinaten-Iterator abzubilden
struct PointCoordAccessor {
  typedef vector<Point>::const_iterator Pit;
  typedef vector<double>::const_iterator Cit;

  inline Cit operator()(Pit it) const { return it->getCoordinates().begin(); }
};

Ball findMinEnclosingBall(const vector<Point> &points) {
  int dimension = points.front().getCoordinates().size();

  Miniball::Miniball<PointCoordAccessor> mb(dimension, points.begin(),
                                            points.end());

  // Hole das Zentrum und den Radius des berechneten Minimum Enclosing Balls
  const double *center_coords = mb.center();
  double radius = sqrt(mb.squared_radius());

  // Konvertiere das Zentrum zu einem Point-Objekt
  vector<double> center_vector(center_coords, center_coords + dimension);
  Point center_point(center_vector);

  // Erstelle und gebe den Ball zurück
  return Ball(center_point, radius);
}
