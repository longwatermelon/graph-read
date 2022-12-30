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

    // Load image
    int w, h;
    std::vector<std::array<unsigned char, 3>> imgdata;
    load_image(imgdata, argv[1], w, h);

    // Setup graph & data points for regression
    std::vector<reg::DataPoint<1>> data;
    std::string graph_config = "min 0 0\nmax 1 1\nstep .2 .2\n";
    for (size_t i = 0; i < imgdata.size(); ++i)
    {
        if (IS_BLACK(imgdata[i]))
        {
            float x = (float)(i % w) / (w - 1);
            float y = 1.f - (float)(i - (i % w)) / w / (h - 1);
            data.emplace_back(reg::DataPoint<1>({ x }, y));
            graph_config += "data " + std::to_string(x) +
                            " " + std::to_string(y) + "0\n";
        }
    }

    std::vector<Equation<3>> eqs = all_possible_powers<3>(3);
    for (const auto &eq : eqs)
    {
        for (float p : eq.xpows)
            printf("%f ", p);
        printf("\n");
    }

    /* std::array<float, 1> vw; */
    /* vw.fill(0.f); */
    /* float b = 0.f; */

    /* for (size_t i = 0; i < 1000; ++i) */
    /* { */
    /*     reg::general::descend<1>(vw, b, .1f, data, */
    /*             [&](const std::array<float, 1> &vw, */
    /*                 const std::array<float, 1> &vx, */
    /*                 float b){ */
    /*         return vw[0] * vx[0] + b; */
    /*     }); */
    /* } */

    /* printf("%.2fx + %.2f\n", vw[0], b); */

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

