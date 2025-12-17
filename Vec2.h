#pragma once
#include <cmath>
#include <ostream>
#include "CudaConfig.h"

struct Vec2
{
    float x, y;

    // --- Constructors ---
    HD
    constexpr Vec2() : x(0), y(0) {}
    HD
    constexpr Vec2(float x, float y) : x(x), y(y) {}

    // --- Basic arithmetic ---
    HD
    constexpr Vec2 operator+(const Vec2& other) const {
        return { x + other.x, y + other.y };
    }

    HD
    constexpr Vec2 operator-(const Vec2& other) const {
        return { x - other.x, y - other.y };
    }

    HD
    constexpr Vec2 operator*(float scalar) const {
        return { x * scalar, y * scalar };
    }

    HD
    constexpr Vec2 operator/(float scalar) const {
        return { x / scalar, y / scalar };
    }

    // Compound assignments
    HD
    Vec2& operator+=(const Vec2& other) {
        x += other.x; y += other.y;
        return *this;
    }

    HD
    Vec2& operator-=(const Vec2& other) {
        x -= other.x; y -= other.y;
        return *this;
    }

    HD
    Vec2& operator*=(float scalar) {
        x *= scalar; y *= scalar;
        return *this;
    }

    HD
    Vec2& operator/=(float scalar) {
        x /= scalar; y /= scalar;
        return *this;
    }

    // --- Dot product ---
    HD
    constexpr float dot(const Vec2& other) const {
        return x * other.x + y * other.y;
    }

    // --- Magnitude ---
    HD
    float length() const {
        return std::sqrt(x * x + y * y);
    }

    HD
    float lengthSquared() const {
        return x * x + y * y;
    }

    // --- Normalization ---
    HD
    Vec2 normalized() const {
        float len = length();
        return (len == 0.0f) ? Vec2(0, 0) : Vec2(x / len, y / len);
    }

    HD
    void normalize() {
        float len = length();
        if (len != 0.0f) { x /= len; y /= len; }
    }

    // --- Comparison ---
    HD
    constexpr bool operator==(const Vec2& other) const {
        return x == other.x && y == other.y;
    }

    HD
    constexpr bool operator!=(const Vec2& other) const {
        return !(*this == other);
    }
};

// Optional: scalar * vector (commutative multiply)
HD
inline Vec2 operator*(float scalar, const Vec2& v) {
    return { v.x * scalar, v.y * scalar };
}

//stream a vector
inline std::ostream& operator<<(std::ostream& os, const Vec2& v)
{
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}