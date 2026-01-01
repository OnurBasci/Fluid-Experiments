#pragma once
#include <cmath>
#include <ostream>
#include "CudaConfig.h"

struct Vec2
{
    float x, y;

    // --- Constructors ---
    CUDA_HD
    constexpr Vec2() : x(0), y(0) {}
    CUDA_HD
    constexpr Vec2(float x, float y) : x(x), y(y) {}

    // --- Basic arithmetic ---
    CUDA_HD
    constexpr Vec2 operator+(const Vec2& other) const {
        return { x + other.x, y + other.y };
    }

    CUDA_HD
    constexpr Vec2 operator-(const Vec2& other) const {
        return { x - other.x, y - other.y };
    }

    CUDA_HD
    constexpr Vec2 operator*(float scalar) const {
        return { x * scalar, y * scalar };
    }

    CUDA_HD
    constexpr Vec2 operator/(float scalar) const {
        return { x / scalar, y / scalar };
    }

    // Compound assignments
    CUDA_HD
    Vec2& operator+=(const Vec2& other) {
        x += other.x; y += other.y;
        return *this;
    }

    CUDA_HD
    Vec2& operator-=(const Vec2& other) {
        x -= other.x; y -= other.y;
        return *this;
    }

    CUDA_HD
    Vec2& operator*=(float scalar) {
        x *= scalar; y *= scalar;
        return *this;
    }

    CUDA_HD
    Vec2& operator/=(float scalar) {
        x /= scalar; y /= scalar;
        return *this;
    }

    // --- Dot product ---
    CUDA_HD
    constexpr float dot(const Vec2& other) const {
        return x * other.x + y * other.y;
    }

    // --- Magnitude ---
    CUDA_HD
    float length() const {
        return std::sqrt(x * x + y * y);
    }

    CUDA_HD
    float lengthSquared() const {
        return x * x + y * y;
    }

    // --- Normalization ---
    CUDA_HD
    Vec2 normalized() const {
        float len = length();
        return (len == 0.0f) ? Vec2(0, 0) : Vec2(x / len, y / len);
    }

    CUDA_HD
    void normalize() {
        float len = length();
        if (len != 0.0f) { x /= len; y /= len; }
    }

    // --- Comparison ---
    CUDA_HD
    constexpr bool operator==(const Vec2& other) const {
        return x == other.x && y == other.y;
    }

    CUDA_HD
    constexpr bool operator!=(const Vec2& other) const {
        return !(*this == other);
    }
};

// Optional: scalar * vector (commutative multiply)
CUDA_HD
inline Vec2 operator*(float scalar, const Vec2& v) {
    return { v.x * scalar, v.y * scalar };
}

//stream a vector
inline std::ostream& operator<<(std::ostream& os, const Vec2& v)
{
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}