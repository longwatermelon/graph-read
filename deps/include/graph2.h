#pragma once
#include <string>
#include <vector>
#include <functional>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>

namespace graph
{
    struct DataPoint2
    {
        glm::vec2 p{ 0.f };
        size_t shape{ 0 };
    };

    struct Graph2Shape
    {
        Graph2Shape(const std::vector<glm::vec2> &points, glm::vec3 col)
            : points(points), col(col) {}

        std::vector<glm::vec2> points;
        glm::vec3 col{ 0.f };
    };

    class Graph2
    {
    public:
        Graph2() = default;
        Graph2(const std::string &data_fp);
        ~Graph2();

        void render(SDL_Renderer *rend, SDL_Rect r, const std::function<float(float)> &func) const;
        void render_shape(SDL_Renderer *rend, SDL_Rect r, const DataPoint2 &p) const;

        void load(const std::string &config);

        const std::vector<DataPoint2> &data() const { return m_data; }
        glm::vec2 min() const { return m_min; }
        glm::vec2 max() const { return m_max; }
        glm::vec2 step() const { return m_step; }

        void add_shape(const Graph2Shape &shape);

        void set_espace(float espace) { m_espace = espace; }
        void set_shape_size(float size) { m_shape_dim = size; }

    private:
        float gx2scr(float x, SDL_Rect r) const;
        float gy2scr(float y, SDL_Rect r) const;

    private:
        glm::vec2 m_min{ 0.f }, m_max{ 0.f };
        glm::vec2 m_step{ 0.f };
        std::vector<DataPoint2> m_data;

        std::vector<Graph2Shape> m_shapes;

        float m_espace{ 20.f };
        float m_shape_dim{ 6.f };
    };
}

