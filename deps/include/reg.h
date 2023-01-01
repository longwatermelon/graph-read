#pragma once
#include <cstdio>
#include <vector>
#include <array>
#include <functional>
#include <glm/glm.hpp>

namespace reg
{
    // N: num features
    template <size_t N>
    struct DataPoint
    {
        DataPoint()
        {
            features.fill(0.f);
        }

        DataPoint(const std::array<float, N> &features, float y)
            : features(features), y(y) {}

        std::array<float, N> features;
        float y{ 0.f };
    };

    namespace general
    {
        float calc_mean(const std::vector<float> &values);
        float calc_sd(const std::vector<float> &values);
        void zscore_normalize(std::vector<float> &features, float &sd, float &mean);

        template <size_t N>
        float dot(const std::array<float, N> &a, const std::array<float, N> &b)
        {
            float sum = 0.f;
            for (size_t i = 0; i < a.size(); ++i)
                sum += a[i] * b[i];
            return sum;
        }

        // N: num features
        template <size_t N>
        void feature_scale(std::vector<DataPoint<N>> &data, std::array<float, N> &sd, std::array<float, N> &mean)
        {
            for (size_t c = 0; c < N; ++c)
            {
                std::vector<float> features;
                features.reserve(data.size());

                for (size_t r = 0; r < data.size(); ++r)
                    features.emplace_back(data[r].features[c]);

                zscore_normalize(features, sd[c], mean[c]);

                for (size_t r = 0; r < data.size(); ++r)
                    data[r].features[c] = features[r];
            }
        }

        // N: num features
        template <size_t N>
        float cost(const std::vector<DataPoint<N>> &data,
                   const std::array<float, N> &vw,
                   const std::function<float(const DataPoint<N>&)> &err_f,
                   float lambda = 0.f)
        {
            float cost = 0.f;
            for (const auto &p : data)
                cost += err_f(p);
            cost /= 2.f * data.size();

            float reg_term = 0.f;
            for (const auto &w : vw)
                reg_term += w * w;
            reg_term *= lambda / (2.f * data.size());

            return cost + reg_term;
        }

        // N: num features
        // func parameters: vw, vx, b
        template <size_t N>
        void descend(std::array<float, N> &vw, float &b, float a,
                     const std::vector<DataPoint<N>> &data,
                     const std::function<float(const std::array<float, N>&,
                                               const std::array<float, N>&,
                                               float)> &func,
                     float lambda = 0.f)
        {
            // Calculate new b
            float db_j = 0.f;
            for (size_t i = 0; i < data.size(); ++i)
                db_j += func(vw, data[i].features, b) - data[i].y;
            db_j /= data.size();
            float b_new = b - a * db_j;

            // Calculate new vw
            std::array<float, N> vw_new;
            for (size_t j = 0; j < N; ++j)
            {
                float dw_j = 0.f;
                for (size_t i = 0; i < data.size(); ++i)
                    dw_j += (func(vw, data[i].features, b) - data[i].y) * data[i].features[j];
                dw_j /= data.size();
                dw_j += lambda / data.size() * vw[j];

                vw_new[j] = vw[j] - a * dw_j;
            }

            b = b_new;
            vw = vw_new;
        }
    }

    namespace linear
    {
        float f_wb(const std::array<float, 1> &w, const std::array<float, 1> &features, float b);
    }

    namespace multilinear
    {
        template <size_t N>
        float f_wb(const std::array<float, N> &vw, const std::array<float, N> &features,
                   float b)
        {
            // w \dot x + b
            float res = b;
            for (size_t i = 0; i < N; ++i)
                res += vw[i] * features[i];
            return res;
        }
    }

    namespace logistic
    {
        float f_wb(const std::array<float, 1> &w, const std::array<float, 1> &features, float b);
        float loss(float prediction, float data_y);
    }

    namespace multilogistic
    {
        float f_wb(const std::array<float, 2> &w, const std::array<float, 2> &features, float b);
    }
}

