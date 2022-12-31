#include <sstream>
#include <reg.h>
#include <graph2.h>

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
}

#define BLACK_THRESHOLD 50
#define IS_BLACK(pix) (pix[0] <= BLACK_THRESHOLD && pix[1] <= BLACK_THRESHOLD && pix[2] <= BLACK_THRESHOLD)

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
        ss << b << '\n';
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
            ssout << coeff << "x^" << pow << " + ";
        ssout << b << '\n';
        return ssout.str();
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
std::vector<Equation<N>> all_possible_powers(int maxp)
{
    std::vector<Equation<N>> equations;

    if constexpr (N > 0)
    {
        std::vector<Equation<N - 1>> next_term_eqs = all_possible_powers<N - 1>(maxp);

        // Each eq in next_term_eqs needs to be paired with every possible power
        // of the current term
        for (const auto &eq : next_term_eqs)
        {
            // Initialize base equation's xpows[0:N-1] whose
            // last term power will be modified from [1,maxp]
            Equation<N> e;
            for (size_t j = 0; j < N - 1; ++j)
                e.xpows[j] = eq.xpows[j];

            for (int pow = 1; pow <= maxp; ++pow)
            {
                e.xpows[N - 1] = pow;
                equations.emplace_back(e);
            }
        }
    }

    if (equations.empty())
        equations.emplace_back(Equation<N>());

    return equations;
}

// Returns cost
template <size_t N>
float fit_eq(Equation<N> &eq, size_t iters, float a, const std::vector<reg::DataPoint<N>> &data)
{
    auto f_xraise = [eq](std::array<float, N> x){
        for (size_t i = 0; i < N; ++i)
            x[i] = std::pow(x[i], eq.xpows[i]);
        return x;
    };

    for (size_t i = 0; i < iters; ++i)
    {
        reg::general::descend<N>(eq.vw, eq.b, a, data,
            [f_xraise](const std::array<float, N> &vw,
               const std::array<float, N> &vx,
               float b){
            return reg::general::dot(vw, f_xraise(vx)) + b;
        });
    }

    return reg::general::cost<N>(data,
            [eq, f_xraise](const reg::DataPoint<N> &dp){
        return std::pow((reg::general::dot(eq.vw, f_xraise(dp.features)) + eq.b) - dp.y, 2);
    });
}

// N: num terms
// Returns pair { eq: cost }
template <size_t N>
std::pair<Equation<N>, float> find_best_fit(int maxp, const std::vector<reg::DataPoint<N>> &data)
{
    std::vector<Equation<N>> eqs = all_possible_powers<N>(maxp);

    float min_cost = std::numeric_limits<float>::max();
    Equation<N> min_cost_e;
    for (auto &eq : eqs)
    {
        float cost = fit_eq(eq, 1000, .1f, data);
        if (cost < min_cost)
        {
            min_cost = cost;
            min_cost_e = eq;
        }
    }

    return { min_cost_e, min_cost };
}

// N: new feature num
// No: orig feature num
template <size_t N, size_t No>
std::vector<reg::DataPoint<N>> reduce_data(const std::vector<reg::DataPoint<No>> &data)
{
    std::vector<reg::DataPoint<N>> res;
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

template <size_t N>
std::pair<std::string, float> find_best_fit_all(int maxp, const std::vector<reg::DataPoint<N>> &data)
{
    std::pair<std::string, float> res{ "", std::numeric_limits<float>::max() };

    if constexpr (N > 0)
    {
        std::pair<Equation<N>, float> best_n = find_best_fit<N>(maxp, data);
        std::pair<std::string, float> best_next_n = find_best_fit_all<N - 1>(maxp, reduce_data<N - 1, N>(data));

        bool best_n_better = best_n.second < best_next_n.second;
        res.first = best_n_better ? best_n.first.to_string() : best_next_n.first;
        res.second = best_n_better ? best_n.second : best_next_n.second;
    }

    return res;
}

/*
   Start with a single feature, and iterate from smallest power (1 for now)
   to highest power (10 for now). After that, increment the number of terms
   in the polynomial and find every possible combination of the powers of
   the terms in the polynomial, recording their equation and cost.
 */
int main(int argc, char **argv)
{
    if (argc == 1)
    {
        fprintf(stderr, "Error: no file provided\n");
        exit(EXIT_FAILURE);
    }

    // Define constants
    constexpr size_t max_terms = 3;
    size_t max_pow = 3;

    // Load image
    int w, h;
    std::vector<std::array<unsigned char, 3>> imgdata;
    load_image(imgdata, argv[1], w, h);

    // Setup graph & data points for regression
    std::vector<reg::DataPoint<max_terms>> data;
    std::string graph_config = "min 0 0\nmax 1 1\nstep .2 .2\n";
    for (size_t i = 0; i < imgdata.size(); ++i)
    {
        if (IS_BLACK(imgdata[i]))
        {
            float x = (float)(i % w) / (w - 1);
            float y = 1.f - (float)(i - (i % w)) / w / (h - 1);

            reg::DataPoint<max_terms> dp;
            dp.features.fill(x);
            dp.y = y;
            data.emplace_back(dp);

            graph_config += "data " + std::to_string(x) +
                            " " + std::to_string(y) + "0\n";
        }
    }

    // Find best fit graph out of all possible polynomials

    std::pair<std::string, float> best_fit = find_best_fit_all<max_terms>(max_pow, data);
    printf("Equation: %s\nCost: %f\n", best_fit.first.c_str(), best_fit.second);

#ifdef GRAPHICS
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("Graph",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            600, 600, SDL_WINDOW_SHOWN);
    SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    graph::Graph2 graph;
    graph.add_shape(graph::Graph2Shape(
        {
            { 0.f, 0.f }, { 1.f, 1.f },
            { 1.f, 0.f }, { 0.f, 1.f }
        },
        { 1.f, 0.f, 0.f }
    ));
    graph.set_shape_size(2.f);

    graph.load(graph_config);

    bool running = true;
    SDL_Event evt;

    while (running)
    {
        while (SDL_PollEvent(&evt))
        {
            switch (evt.type)
            {
            case SDL_QUIT:
                running = false;
                break;
            }
        }

        SDL_RenderClear(rend);

        graph.render(rend, { 0, 0, 600, 600 }, [](float x){ return 0.f; });

        SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
        SDL_RenderPresent(rend);
    }

    SDL_DestroyRenderer(rend);
    SDL_DestroyWindow(win);
    SDL_Quit();
#endif

    return 0;
}

