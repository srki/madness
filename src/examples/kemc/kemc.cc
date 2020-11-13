#include <madness/world/MADworld.h>
#include <madness/mra/mra.h>

using namespace madness;

int main(int argc, char *argv[]) {
    const double thresh = 1e-10;

    if (argc != 9) {
        std::cerr << "Need 8 args " << std::endl;
        return -1;
    }

    int k = std::stoi(argv[1]);
    auto basePath = std::string{argv[2]};
    int v1Start = std::stoi(argv[3]);
    int v1Size = std::stoi(argv[4]);
    int v1Step = std::stoi(argv[5]);
    int v2Start = std::stoi(argv[6]);
    int v2Size = std::stoi(argv[7]);
    int v2Step = std::stoi(argv[8]);

    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world, argc, argv);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_autorefine(false);
//    FunctionDefaults<3>::set_cubic_cell(-L, L);
    FunctionDefaults<3>::set_max_refine_level(30);


    /* region Read the functions */
    real_function_3d v1[v1Size], v2[v2Size];

    auto startRead = std::chrono::steady_clock::now();
    for (int i = 0; i < v1Size; i++) {
        v1[i] = real_factory_3d(world).empty();

        std::ostringstream treePath;
        treePath << basePath << std::setw(4) << std::setfill('0') << (v1Start + i * v1Step);
        v1[i].load_tree_random_coeff(treePath.str(), i);

    }

    for (int i = 0; i < v2Size; i++) {
        v2[i] = real_factory_3d(world).empty();

        std::ostringstream treePath;
        treePath << basePath << std::setw(4) << std::setfill('0') << (v2Start + i * v2Step);
        v2[i].load_tree_random_coeff(treePath.str(), i);
    }
    auto endRead = std::chrono::steady_clock::now();
    /* endregion */

    /* region Derivation */
    auto startDiff = std::chrono::steady_clock::now();
    real_derivative_3d dOp[3] = {free_space_derivative<double, 3>(world, 0),
                                 free_space_derivative<double, 3>(world, 1),
                                 free_space_derivative<double, 3>(world, 2)};

    real_function_3d dv1[v1Size][3];
    real_function_3d dv2[v1Size][3];

    for (int i = 0; i < v1Size; i++) {
        for (int j = 0; j < 3; j++) {
            dv1[i][j] = dOp[j](v1[i], false);
        }
    }

    for (int i = 0; i < v2Size; i++) {
        for (int j = 0; j < 3; j++) {
            dv2[i][j] = dOp[j](v2[i], false);
        }
    }

    world.gop.fence();
    auto endDiff = std::chrono::steady_clock::now();
    /* endregion */


    /* region Compress */
    auto startCompress = std::chrono::steady_clock::now();
    for (int i = 0; i < v1Size; i++) {
        for (int j = 0; j < 3; j++) {
            dv1[i][j].compress(false);
        }
    }

    for (int i = 0; i < v2Size; i++) {
        for (int j = 0; j < 3; j++) {
            dv2[i][j].compress(false);
        }
    }

    world.gop.fence();
    auto endCompress = std::chrono::steady_clock::now();
    /* endregion */


    /* region inner product */
    bool print = false;
    double result = 0.0;
    auto startInner = std::chrono::steady_clock::now();
    for (int l = 0; l < 3; l++) {
        for (int i = 0; i < v1Size; i++) {
            for (int j = 0; j < v2Size; j++) {
                auto t = dv1[i][l].inner(dv2[j][l]);
                result += t;
                if (print) { std::cout << t << " "; };
            }
            if (print) { std::cout << std::endl; }
        }
        if (print) { std::cout << std::endl; }
    }
    auto endInner = std::chrono::steady_clock::now();
    /* endregion */

    std::cout << "Result: " << result << std::endl;
    std::cout << "LOG;"
              << std::chrono::duration_cast<std::chrono::milliseconds>(endRead - startRead).count() << ";"
              << std::chrono::duration_cast<std::chrono::milliseconds>(endDiff - startDiff).count() << ";"
              << std::chrono::duration_cast<std::chrono::milliseconds>(endCompress - startCompress).count() << ";"
              << std::chrono::duration_cast<std::chrono::milliseconds>(endInner - startInner).count() << std::endl;

    finalize();
}