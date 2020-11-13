#include <madness/world/MADworld.h>
#include <madness/mra/mra.h>

using namespace madness;

const double L = 16.0;   // box size [-L,L]^3

int main(int argc, char *argv[]) {
    int k = 10;
    double thresh = 1e-10;

    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    startup(world, argc, argv);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_autorefine(false);
    FunctionDefaults<3>::set_cubic_cell(-L, L);
    FunctionDefaults<3>::set_max_refine_level(30);

    real_function_3d f = real_factory_3d(world).empty();
    f.load_tree_random_coeff(argv[1]);

    std::cout << "---------" << std::endl;
    f.print_tree();

    real_derivative_3d D0 = free_space_derivative<double,3>(world, 0);
    real_function_3d df = D0(f);

    std::cout << "---------" << std::endl;
    df.print_tree();

    df.compress();

    std::cout << "---------" << std::endl;
    df.print_tree();

    std::cout << df.inner(df) << std::endl;

    finalize();
}