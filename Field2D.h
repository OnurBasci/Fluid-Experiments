#include <vector>
#include <cassert>

template<typename T>
class Field2D {
public:
    Field2D() : width(0), height(0) {}
    Field2D(size_t h, size_t w) { resize(h, w); }

    void resize(size_t h, size_t w) {
        height = h;
        width = w;
        data.resize(w * h);
    }

    inline T& operator()(size_t y, size_t x) { return data[y * width + x]; }

    // Optional: clear all values to zero
    void fill(const T& value) {
        std::fill(data.begin(), data.end(), value);
    }

    // ======= 2D indexing support: field[y][x] =======

    // Row proxy to allow field[row][col]
    class Row {
    public:
        Row(T* row_ptr, size_t w) : ptr(row_ptr), width(w) {}

        T& operator[](size_t x) {
            return ptr[x];
        }
        const T& operator[](size_t x) const {
            return ptr[x];
        }

    private:
        T* ptr;
        size_t width;
    };

    Row operator[](size_t y) {
        return Row(&data[y * width], width);
    }
    const Row operator[](size_t y) const {
        return Row(const_cast<T*>(&data[y * width]), width);
    }

    // Direct flat access
    T& at(size_t y, size_t x) { return data[y * width + x]; }
    const T& at(size_t y, size_t x) const { return data[y * width + x]; }

    // Return pointer to raw data if needed
    T* raw() { return data.data(); }
    const T* raw() const { return data.data(); }

    size_t getWidth()  const { return width; }
    size_t getHeight() const { return height; }

private:
    size_t width, height;
    std::vector<T> data;
};