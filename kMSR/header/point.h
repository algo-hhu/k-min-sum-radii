// point.h

#ifndef POINT_H
#define POINT_H

#include <stdexcept>
#include <vector>

class Point {
 public:
  Point();
  Point(std::vector<double> coords);

  double distanceTo(const Point& other) const;
  static double distance(const Point& p1, const Point& p2);
  const std::vector<double>& getCoordinates() const { return coordinates; }
  void setCoordinates(std::vector<double> newCoordinates) {
    coordinates = newCoordinates;
  }

  bool operator<(const Point& other) const;
  bool operator==(const Point& other) const;
  bool operator!=(const Point& other) const;
  Point operator+(const Point& other) const;
  Point operator*(double scalar) const;

 private:
  std::vector<double> coordinates;
};

#endif  // POINT_H
