#include <sstream>
#include <reg.h>

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
}

#define BLACK_THRESHOLD 50
#define IS_BLACK(pix) (pix[0] <= BLACK_THRESHOLD && pix[1] <= BLACK_THRESHOLD && pix[2] <= BLACK_THRESHOLD)

bool g_verbose = false;
bool g_fscale = false;

template <size_t N>
struct Equation
{
    Equation()
    {
        vw.fill(0.f);
        vx.fill(0.f);
        xpows.fill(0.f);
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss.precision(2);
        for (size_t i = 0; i < N; ++i)
            ss << vw[i] << "x^" << xpows[i] << " + ";
        ss << b;
        return reduce_str(ss.str());
    }

    std::array<float, N> vw, vx;
    std::array<float, N> xpows;
    float b{ 0.f };

private:
    std::string reduce_str(const std::string &repr) const
    {
        // pow: coeff
        std::unordered_map<float, float> pow_coeffs;
        for (size_t i = 0; i < N; ++i)
            pow_coeffs[xpows[i]] = 0.f;

        std::stringstream ss(repr);
        for (size_t i = 0; i < N; ++i)
        {
            char tmp;
            float coeff;
            float pow;
            ss >> coeff >> tmp >> tmp >> pow >> tmp;
            pow_coeffs[pow] += coeff;
        }

        std::stringstream ssout;
        ssout.precision(2);
        for (const auto &[pow, coeff] : pow_coeffs)
        {
            if (std::abs(coeff) > 1e-4f)
            {
                if (std::abs(coeff - 1.f) > 1e-2f)
                    ssout << coeff;
                ssout << 'x';

                if ((int)pow != 1)
                    ssout << '^' << '{' << pow << '}';
                ssout << " + ";
            }
        }

        if (std::abs(b) > 1e-3f)
            ssout << b;

        std::string res = ssout.str();

        if (res[res.size() - 1] == ' ')
            res = res.substr(0, res.size() - 3);

        return res;
    }
};

bool load_image(std::vector<std::array<unsigned char, 3>> &image, const std::string &path, int &x, int &y)
{
    int n;
    unsigned char* data = stbi_load(path.c_str(), &x, &y, &n, 0);
    if (data != nullptr)
    {
        for (int i = 0; i < x * y * n; i += n)
        {
            std::array<unsigned char, 3> tmp;
            for (int j = 0; j < 3; ++j)
                tmp[j] = data[i + j];

            image.emplace_back(tmp);
        }
    }
    stbi_image_free(data);
    return (data != nullptr);
}

// N: num terms
template <size_t N>
Equation<N> make_polynomial()
{
    Equation<N> eq;

    for (size_t i = 0; i < N; ++i)
        eq.xpows[i] = i + 1;

    return eq;
}

template <size_t N>
Equation<N - 1> make_polynomial_invpow()
{
    Equation<N - 1> eq;

    for (size_t i = 0; i < N - 1; ++i)
        eq.xpows[i] = 1.f / (i + 2);

    return eq;
}

// Returns cost
template <size_t N>
float fit_eq(Equation<N> &eq, size_t iters, float a, std::vector<reg::DataPoint<N>> data,
        std::array<float, N> &sd, std::array<float, N> &mean)
{
    auto f_xraise = [eq](std::array<float, N> x){
        for (size_t i = 0; i < N; ++i)
            x[i] = std::pow(x[i], eq.xpows[i]);
        return x;
    };

    if (g_fscale)
    {
        for (auto &dp : data)
            dp.features = f_xraise(dp.features);

        reg::general::feature_scale<N>(data, sd, mean);
    }

    for (size_t i = 0; i < iters; ++i)
    {
        reg::general::descend<N>(eq.vw, eq.b, a, data,
            [f_xraise](const std::array<float, N> &vw,
               const std::array<float, N> &vx,
               float b){
            // Don't raise vx to power, data already has been raised
            return reg::general::dot(vw, g_fscale ? vx : f_xraise(vx)) + b;
        });
    }

    float cost = reg::general::cost<N>(data,
            [eq, f_xraise](const reg::DataPoint<N> &dp){
        return std::pow((reg::general::dot(eq.vw, g_fscale ? dp.features : f_xraise(dp.features)) + eq.b) - dp.y, 2);
    });

    if (g_verbose)
        printf("Fit %s: Cost = %f\n", eq.to_string().c_str(), cost);

    return cost;
}

// N: new feature num
// No: orig feature num
template <size_t N, size_t No>
std::vector<reg::DataPoint<N>> reduce_data(const std::vector<reg::DataPoint<No>> &data)
{
    std::vector<reg::DataPoint<N>> res;
    res.reserve(data.size());

    for (const auto &p : data)
    {
        reg::DataPoint<N> pn;
        for (size_t i = 0; i < N; ++i)
            pn.features[i] = p.features[i];
        pn.y = p.y;
        res.emplace_back(pn);
    }

    return res;
}

struct BestFitInfo
{
    BestFitInfo() = default;
    template <size_t N>
    BestFitInfo(Equation<N> eq, const std::vector<reg::DataPoint<N>> &data,
            size_t iters, float a)
    {
        std::array<float, N> sd_tmp, mean_tmp;
        cost = fit_eq(eq, iters, a, data, sd_tmp, mean_tmp);

        for (size_t i = 0; i < N; ++i)
        {
            sd.emplace_back(sd_tmp[i]);
            mean.emplace_back(mean_tmp[i]);
            xpows.emplace_back(eq.xpows[i]);
        }

        eq_display = eq.to_string();
    }

    std::string eq_display;
    float cost;
    std::vector<float> sd, mean, xpows;
};

template <size_t N>
BestFitInfo find_best_fit(const std::vector<reg::DataPoint<N>> &data)
{
    BestFitInfo res;
    res.cost = std::numeric_limits<float>::max(); // for N == 0

    int iters = 1000;
    float a = .1f;

    if constexpr (N > 0)
    {
        BestFitInfo fit = BestFitInfo(make_polynomial<N>(), data, iters, a);

        if constexpr (N == 1)
            return fit;
        else
        {
            auto reduced_data = reduce_data<N - 1, N>(data);

            BestFitInfo fitni = BestFitInfo(make_polynomial_invpow<N>(), reduced_data, iters, a);
            BestFitInfo fitn = find_best_fit<N - 1>(reduced_data);

            res = (fit.cost < fitni.cost && fit.cost < fitn.cost) ? fit : (fitni.cost < fitn.cost ? fitni : fitn);
        }
    }

    return res;
}

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        fprintf(stderr, "Error: no file provided\n");
        exit(EXIT_FAILURE);
    }

    glm::vec2 gmin(0.f, 0.f),
              gmax(1.f, 1.f);
    for (int i = 2; i < argc; ++i)
    {
        if (strcmp(argv[i], "-v") == 0)
            g_verbose = true;

        if (strcmp(argv[i], "-fs") == 0)
            g_fscale = true;

        if (strcmp(argv[i], "-dim") == 0)
        {
            std::stringstream ss(argv[++i]);
            char c;
            ss >> gmin.x >> c >> gmin.y >> c >> gmax.x >> c >> gmax.y;
        }
    }

    // Define constants
    constexpr size_t max_terms = 3;

    // Load image
    int w, h;
    std::vector<std::array<unsigned char, 3>> imgdata;
    load_image(imgdata, argv[1], w, h);

    // Setup graph & data points for regression
    std::vector<reg::DataPoint<max_terms>> data;
    for (size_t i = 0; i < imgdata.size(); ++i)
    {
        if (IS_BLACK(imgdata[i]))
        {
            float x = (float)(i % w) / (w - 1) * (gmax.x - gmin.x) + gmin.x;
            float y = (1.f - (float)(i - (i % w)) / w / (h - 1)) * (gmax.y - gmin.y) + gmin.y;

            reg::DataPoint<max_terms> dp;
            dp.features.fill(x);
            dp.y = y;
            data.emplace_back(dp);
        }
    }

    // Find best fit graph out of all possible polynomials
    BestFitInfo best_fit = find_best_fit<max_terms>(data);
    printf("x from [%.2f,%.2f], y from [%.2f,%.2f]\n",
            gmin.x, gmax.x, gmin.y, gmax.y);

    if (g_fscale)
    {
        std::stringstream scaled_eq;
        scaled_eq.precision(2);
        size_t index = 0;
        for (size_t i = 0; i < best_fit.eq_display.size(); ++i)
        {
            char c = best_fit.eq_display[i];
            if (c == 'x')
            {
                scaled_eq << "((x^{" << best_fit.xpows[index] << "} - "
                          << best_fit.mean[index]
                          << ") / " << best_fit.sd[index] << ")";
                ++index;
            }
            else
            {
                if (c == '^')
                {
                    while (best_fit.eq_display[i++] != '}')
                        ;
                }

                scaled_eq << best_fit.eq_display[i];
            }
        }

        printf("y = %s\n", scaled_eq.str().c_str());
    }
    else
    {
        printf("y = %s\n", best_fit.eq_display.c_str());
    }

    printf("Accuracy: %.2f%%\n", (1.f - best_fit.cost / .5f) * 100.f);

    return 0;
}

