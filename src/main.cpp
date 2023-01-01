#include <sstream>
#include <fstream>
#include <reg.h>

#ifdef GRAPHICS
#include <SDL2/SDL.h>
#include <graph2.h>
#endif

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
}

#define BLACK_THRESHOLD 50
#define IS_BLACK(pix) (pix[0] <= BLACK_THRESHOLD && pix[1] <= BLACK_THRESHOLD && pix[2] <= BLACK_THRESHOLD)

bool g_verbose = false;
bool g_fscale = false;
bool g_latex = false;
bool g_graph = false;

glm::vec2 g_gmin(0.f), g_gmax(0.f);

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
        for (size_t i = 0; i < N; ++i)
            ss << vw[i] << "x^{" << xpows[i] << "} + ";
        ss << b;
        return ss.str();
    }

    std::array<float, N> vw, vx;
    std::array<float, N> xpows;
    float b{ 0.f };
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

// Returns cost
template <size_t N>
float fit_eq(Equation<N> &eq, size_t iters, float a, std::vector<reg::DataPoint<N>> data)
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
            // Don't raise vx to power, data already has been raised
            return reg::general::dot(vw, g_fscale ? vx : f_xraise(vx)) + b;
        });
    }

    float cost = reg::general::cost<N>(data, eq.vw,
            [eq, f_xraise](const reg::DataPoint<N> &dp){
        /* printf("x %f predict %f y %f\n", dp.features[0], reg::general::dot(eq.vw, g_fscale ? dp.features : f_xraise(dp.features)) + eq.b, dp.y); */
        return std::pow((reg::general::dot(eq.vw, g_fscale ? dp.features : f_xraise(dp.features)) + eq.b) - dp.y, 2);
    });

    if (g_verbose)
        printf("Fit %s: Cost = %f\n", eq.to_string().c_str(), std::sqrt(cost));

    return std::sqrt(cost);
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
    BestFitInfo(std::vector<reg::DataPoint<N>> data,
            size_t iters, float a, size_t attempts)
    {
        // Feature scale
        Equation<N> eq = make_polynomial<N>();

        auto f_xraise = [eq](std::array<float, N> x){
            for (size_t i = 0; i < N; ++i)
                x[i] = std::pow(x[i], eq.xpows[i]);
            return x;
        };

        if (g_fscale)
        {
            for (auto &dp : data)
                dp.features = f_xraise(dp.features);

            std::array<float, N> sd_arr, mean_arr;
            reg::general::feature_scale<N>(data, sd_arr, mean_arr);

            for (size_t i = 0; i < N; ++i)
            {
                sd.emplace_back(sd_arr[i]);
                mean.emplace_back(mean_arr[i]);
            }
        }

        // Random points for gradient descent
        cost = std::numeric_limits<float>::max();

        for (size_t i = 0; i < attempts; ++i)
        {
            Equation<N> eq_test = make_polynomial<N>();
            for (auto &w : eq_test.vw)
                w = (rand() % 1000) / 10.f - 50.f;

            eq_test.b = (rand() % 1000) / 10.f - 50.f;
            float test_cost = fit_eq(eq_test, iters, a, data);

            if (test_cost < cost)
            {
                cost = test_cost;
                eq = eq_test;
            }
        }

        if (g_verbose) printf("\n");

        for (size_t i = 0; i < N; ++i)
            xpows.emplace_back(eq.xpows[i]);

        eq_display = eq.to_string();

#ifdef GRAPHICS
        float min_x = std::numeric_limits<float>::max(),
              max_x = std::numeric_limits<float>::min();

        for (const auto &dp : data)
        {
            if (dp.features[0] < min_x) min_x = dp.features[0];
            if (dp.features[0] > max_x) max_x = dp.features[0];
        }

        std::stringstream graph_config;
        graph_config << "min " << min_x << ' ' << g_gmin.y
                     << "\nmax " << max_x << ' ' << g_gmax.y
                     << "\nstep " << (g_gmax.x - g_gmin.x) / 5.f
                     << ' ' << (g_gmax.y - g_gmin.y) / 5.f << "\n";
        for (const auto &dp : data)
        {
            float prediction = reg::general::dot(eq.vw, g_fscale ? dp.features : f_xraise(dp.features)) + eq.b;
            graph_config << "data " << dp.features[0] << ' ' << prediction << " 0\n";
            graph_config << "data " << dp.features[0] << ' ' << dp.y << " 1\n";
        }

        std::string graph_name = "graphs/" + std::to_string((int)eq.xpows[eq.xpows.size() - 1]);
        std::ofstream ofs(graph_name);
        ofs << graph_config.str();
        ofs.close();
#endif
    }

    std::string to_string(bool latex)
    {
        std::stringstream scaled_eq;
        size_t index = 0;
        for (size_t i = 0; i < eq_display.size(); ++i)
        {
            char c = eq_display[i];
            if (c == 'x')
            {
                if (latex)
                {
                    scaled_eq << "\\frac{x^{" << xpows[index] << "} - "
                              << mean[index] << "}{" << sd[index] << "}";
                }
                else
                {
                    scaled_eq << "((x^{" << xpows[index] << "} - "
                              << mean[index]
                              << ") / " << sd[index] << ")";
                }
                ++index;
            }
            else
            {
                if (c == '^')
                {
                    while (eq_display[i++] != '}')
                        ;
                }

                scaled_eq << eq_display[i];
            }
        }

        return scaled_eq.str();
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

    int iters = 10000;
    float a = .05f;

    if constexpr (N > 0)
    {
        BestFitInfo fit = BestFitInfo(data, iters, a, 20);

        if constexpr (N == 1)
            return fit;
        else
        {
            auto reduced_data = reduce_data<N - 1, N>(data);
            BestFitInfo fitn = find_best_fit<N - 1>(reduced_data);

            res = fitn.cost < fit.cost ? fitn : fit;
        }
    }

    return res;
}

int main(int argc, char **argv)
{
    srand(time(0));
    if (argc == 1)
    {
        fprintf(stderr, "Error: no file provided\n");
        exit(EXIT_FAILURE);
    }

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
            ss >> g_gmin.x >> c >> g_gmin.y >> c >> g_gmax.x >> c >> g_gmax.y;
        }

        if (strcmp(argv[i], "-latex") == 0)
            g_latex = true;

        if (strcmp(argv[i], "-graph") == 0)
            g_graph = true;
    }

    // Define constants
    constexpr size_t max_terms = 10;

    // Load image
    int w, h;
    std::vector<std::array<unsigned char, 3>> imgdata;
    load_image(imgdata, argv[1], w, h);

    // Setup graph & data points for regression
    std::vector<reg::DataPoint<max_terms>> data;
    for (int x = 0; x < w; x += 2)
    {
        for (int y = 0; y < h; ++y)
        {
            if (IS_BLACK(imgdata[y * w + x]))
            {
                int i = 0;
                while (y + i < h && IS_BLACK(imgdata[x + (y + i) * w]))
                    ++i;
                int middle_y = y + i / 2;
                float gx = (float)(x) / (w - 1) * (g_gmax.x - g_gmin.x) + g_gmin.x;
                float gy = (1.f - (float)(middle_y) / (h - 1)) * (g_gmax.y - g_gmin.y) + g_gmin.y;

                reg::DataPoint<max_terms> dp;
                dp.features.fill(gx);
                dp.y = gy;
                data.emplace_back(dp);
                break;
            }
        }
    }

    // Find best fit graph out of all possible polynomials
    BestFitInfo best_fit = find_best_fit<max_terms>(data);
    printf("x from [%.2f,%.2f], y from [%.2f,%.2f]\n",
            g_gmin.x, g_gmax.x, g_gmin.y, g_gmax.y);

    std::string eq_display;
    if (g_fscale) eq_display = best_fit.to_string(false);
    else eq_display = best_fit.eq_display;
    printf("y = %s\n", eq_display.c_str());

    if (g_latex && g_fscale)
        printf("y = %s\n", best_fit.to_string(true).c_str());

    float sum_y = 0.f;
    for (const auto &dp : data)
        sum_y += dp.y * dp.y;
    sum_y = std::sqrt(sum_y / data.size());

    printf("Accuracy: %f%% | Cost: %f\n", (1.f - best_fit.cost / sum_y) * 100.f, best_fit.cost);

#ifdef GRAPHICS
    if (g_graph)
    {
        SDL_Init(SDL_INIT_VIDEO);
        SDL_Window *win = SDL_CreateWindow("Graph",
                SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                800, 800, SDL_WINDOW_SHOWN);
        SDL_Renderer *rend = SDL_CreateRenderer(win, -1,
                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

        size_t i = eq_display.find_last_of("^");
        std::stringstream ss(eq_display.substr(i, eq_display.size() - i));
        char tmp;
        int pow;
        ss >> tmp >> tmp >> pow;

        std::string filename = "graphs/" + std::to_string(pow);
        graph::Graph2 graph(filename);
        graph.add_shape(graph::Graph2Shape(
            {
                { 0.f, 0.f }, { 1.f, 1.f },
                { 1.f, 0.f }, { 0.f, 1.f }
            },
            { 1.f, 0.f, 0.f }
        ));
        graph.add_shape(graph::Graph2Shape(
            {
                { .5f, 0.f }, { 0.f, 1.f },
                { .5f, 0.f }, { 1.f, 1.f },
                { 0.f, 1.f }, { 1.f, 1.f }
            },
            { 0.f, .5f, 1.f }
        ));
        graph.set_shape_size(4.f);

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

            graph.render(rend, { 0, 0, 800, 800 }, [](float x){ return 0.f; });

            SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
            SDL_RenderPresent(rend);
        }

        SDL_DestroyRenderer(rend);
        SDL_DestroyWindow(win);
        SDL_Quit();
    }
#endif

    return 0;
}

