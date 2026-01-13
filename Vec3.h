#pragma once
#include <cmath>
#include <ostream>
#include "CudaConfig.h"

struct Vec3
{
    float x, y, z;

    // --- Constructors ---
    CUDA_HD
        constexpr Vec3() : x(0), y(0), z(0) {}

    CUDA_HD
        constexpr Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    // --- Basic arithmetic ---
    CUDA_HD
        constexpr Vec3 operator+(const Vec3& other) const {
        return { x + other.x, y + other.y, z + other.z };
    }

    CUDA_HD
        constexpr Vec3 operator-(const Vec3& other) const {
        return { x - other.x, y - other.y, z - other.z };
    }

    CUDA_HD
        constexpr Vec3 operator*(float scalar) const {
        return { x * scalar, y * scalar, z * scalar };
    }

    CUDA_HD
        constexpr Vec3 operator/(float scalar) const {
        return { x / scalar, y / scalar, z / scalar };
    }

    // --- Compound assignments ---
    CUDA_HD
        Vec3& operator+=(const Vec3& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    CUDA_HD
        Vec3& operator-=(const Vec3& other) {
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }

    CUDA_HD
        Vec3& operator*=(float scalar) {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }

    CUDA_HD
        Vec3& operator/=(float scalar) {
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }

    // --- Dot product ---
    CUDA_HD
        constexpr float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // --- Cross product ---
    CUDA_HD
        constexpr Vec3 cross(const Vec3& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    // --- Magnitude ---
    CUDA_HD
        float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    CUDA_HD
        float lengthSquared() const {
        return x * x + y * y + z * z;
    }

    // --- Normalization ---
    CUDA_HD
        Vec3 normalized() const {
        float len = length();
        return (len == 0.0f) ? Vec3(0, 0, 0) : Vec3(x / len, y / len, z / len);
    }

    CUDA_HD
        void normalize() {
        float len = length();
        if (len != 0.0f) {
            x /= len; y /= len; z /= len;
        }
    }

    // --- Comparison ---
    CUDA_HD
        constexpr bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    CUDA_HD
        constexpr bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }
};

// Optional: scalar * vector (commutative multiply)
CUDA_HD
inline Vec3 operator*(float scalar, const Vec3& v) {
    return { v.x * scalar, v.y * scalar, v.z * scalar };
}

// Stream a vector (host-only)
inline std::ostream& operator<<(std::ostream& os, const Vec3& v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}
