// point.h

#ifndef POINT_H
#define POINT_H

#include <stdexcept>
#include <vector>
#include <cmath>

class Point {
 public:
  Point(std::vector<double> coords);
  Point(std::vector<double> coords, int pos);
  Point(int dimension);

  double distanceTo(const Point& other) const;
  double squaredDistanceToOrigin() const;
  static double distance(const Point& p1, const Point& p2);
  const std::vector<double>& getCoordinates() const { return coordinates; }
  void setCoordinates(std::vector<double> newCoordinates) {
    coordinates = newCoordinates;
  }
  int getPosition() const { return position; }
  void setPosition(int newPosition) { position = newPosition; }

  std::string print() const;

  bool operator<(const Point& other) const;
  bool operator==(const Point& other) const;
  bool operator!=(const Point& other) const;
  Point operator+(const Point& other) const;
  Point operator*(double scalar) const;

 private:
  std::vector<double> coordinates;
  int position = -1;
};

#endif  // POINT_H
