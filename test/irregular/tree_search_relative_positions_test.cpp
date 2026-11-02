#include "packingsolver/irregular/instance_builder.hpp"
#include "packingsolver/irregular/optimize.hpp"
#include "irregular/tree_search_relative_positions.hpp"

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

using namespace packingsolver;
using namespace packingsolver::irregular;
namespace fs = boost::filesystem;

namespace
{

Instance read_instance(const std::string& name)
{
    InstanceBuilder instance_builder;
    instance_builder.read((fs::path("data") / "irregular" / "tests" / (name + ".json")).string());
    return instance_builder.build();
}

/** One entry per copy of each item type of an instance. */
std::vector<ItemTypeId> all_items(const Instance& instance)
{
    std::vector<ItemTypeId> item_type_ids;
    for (ItemTypeId item_type_id = 0; item_type_id < instance.number_of_item_types(); ++item_type_id)
        for (ItemPos copy = 0; copy < instance.item_type(item_type_id).copies; ++copy)
            item_type_ids.push_back(item_type_id);
    return item_type_ids;
}

}

struct IrregularTreeSearchRelativePositionsTestParams
{
    /** Name of the instance, in 'data/irregular/tests'. */
    std::string name;

    /** Expected status, for all the items of the instance. */
    TreeSearchRelativePositionsStatus expected_status;

    /** For the open dimension objectives, expected bound. */
    LengthDbl expected_open_dimension_bound = 0;
};

inline std::ostream& operator<<(std::ostream& os, const IrregularTreeSearchRelativePositionsTestParams& test_params)
{
    os << test_params.name;
    return os;
}

class IrregularTreeSearchRelativePositionsTest: public testing::TestWithParam<IrregularTreeSearchRelativePositionsTestParams> { };

TEST_P(IrregularTreeSearchRelativePositionsTest, IrregularTreeSearchRelativePositions)
{
    IrregularTreeSearchRelativePositionsTestParams test_params = GetParam();
    std::cout << "Instance: " << test_params.name << std::endl;
    Instance instance = read_instance(test_params.name);
    bool open_dimension = (instance.objective() == Objective::OpenDimensionX
            || instance.objective() == Objective::OpenDimensionY);

    // The same result is expected with a single thread and with several
    // ones. The pieces of the pairs are only computed once.
    TreeSearchRelativePositionsCache cache;
    for (Counter number_of_threads: {1, 4}) {
        TreeSearchRelativePositionsParameters parameters;
        parameters.verbosity_level = 0;
        parameters.number_of_threads = number_of_threads;
        parameters.cache = &cache;
        TreeSearchRelativePositionsOutput output = tree_search_relative_positions(
                instance, 0, all_items(instance), parameters);
        EXPECT_EQ(output.status, test_params.expected_status);
        if (!open_dimension)
            continue;
        EXPECT_NEAR(output.open_dimension_bound, test_params.expected_open_dimension_bound, 1e-4);
        // If the positions of the solution are a packing, its open dimension
        // is the bound.
        Solution solution(instance);
        if (tree_search_relative_positions_solution(instance, 0, output, solution)) {
            EXPECT_NEAR(
                    (instance.objective() == Objective::OpenDimensionX)? solution.x_max(): solution.y_max(),
                    output.open_dimension_bound,
                    1e-4);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
        Irregular,
        IrregularTreeSearchRelativePositionsTest,
        testing::ValuesIn(std::vector<IrregularTreeSearchRelativePositionsTestParams>{
            {
                // Two 55 x 55 squares don't fit in a 100 x 100 bin.
                "feasibility_squares_too_large",
                TreeSearchRelativePositionsStatus::Infeasible
            }, {
                // Four 50 x 50 squares exactly fit in a 100 x 100 bin.
                "feasibility_squares_exact_fit",
                TreeSearchRelativePositionsStatus::Feasible
            }, {
                // Five 50 x 50 squares don't fit in a 100 x 100 bin.
                "feasibility_squares_one_too_many",
                TreeSearchRelativePositionsStatus::Infeasible
            }, {
                // Step of size 20 x 20: with the rotation by 180 degrees, two
                // steps exactly fit in a 20 x 30 bin, interlocked.
                "feasibility_steps_interlocked",
                TreeSearchRelativePositionsStatus::Feasible
            }, {
                // Without the rotation, two steps don't fit in a 20 x 30 bin.
                "feasibility_steps_without_rotation",
                TreeSearchRelativePositionsStatus::Infeasible
            }, {
                // Every placement of a cross of span 70 in a 100 x 100 bin
                // covers the center of the bin, so two crosses don't fit.
                "feasibility_crosses",
                TreeSearchRelativePositionsStatus::Infeasible
            }, {
                // One cross and a 25 x 25 square in a corner fit.
                "feasibility_cross_and_square",
                TreeSearchRelativePositionsStatus::Feasible
            }, {
                // Two 60 x 40 rectangles can't be side by side in a 100 x 100
                // bin, so three of them don't fit.
                "feasibility_rectangles_core_bound",
                TreeSearchRelativePositionsStatus::Infeasible
            }, {
                // Two 60 x 50 rectangles fit exactly in a 100 x 100 bin.
                "feasibility_rectangles_exact_fit",
                TreeSearchRelativePositionsStatus::Feasible
            }, {
                // The rotations by 0 and 180 degrees of a square are the
                // same, and the ones of the (single) step are symmetric
                // through the bin: a single rotation of each item is left.
                // The square fits in the notch of the step.
                "feasibility_step_and_square_symmetries",
                TreeSearchRelativePositionsStatus::Feasible
            }, {
                // Two steps, a 30 x 30 square and a 60 x 25 rectangle fit in a
                // 100 x 100 bin.
                "feasibility_steps_square_rectangle",
                TreeSearchRelativePositionsStatus::Feasible
            }, {
                // Five items of poly2a, with four rotations each: the
                // infeasible set needs many nodes.
                "feasibility_poly2a_five_items_infeasible",
                TreeSearchRelativePositionsStatus::Infeasible
            }, {
                "feasibility_poly2a_five_items_feasible",
                TreeSearchRelativePositionsStatus::Feasible
            }, {
                // In a strip of height 30, two steps take a length of 20 with
                // the rotation by 180 degrees (interlocked).
                "open_dimension_x_steps_interlocked",
                TreeSearchRelativePositionsStatus::Feasible,
                20
            }, {
                // Without the rotation, they take a length of 30 (one in the
                // notch of the other).
                "open_dimension_x_steps_without_rotation",
                TreeSearchRelativePositionsStatus::Feasible,
                30
            }, {
                // Same as above, in a strip of width 30.
                "open_dimension_y_steps_interlocked",
                TreeSearchRelativePositionsStatus::Feasible,
                20
            }, {
                "open_dimension_y_steps_without_rotation",
                TreeSearchRelativePositionsStatus::Feasible,
                30
            }, {
                // The five first items of fu, without rotation, in a strip of
                // height 38.
                "open_dimension_x_fu_five_items",
                TreeSearchRelativePositionsStatus::Feasible,
                17.8889
            }}));

TEST(IrregularTreeSearchRelativePositions, CoreBound)
{
    // Two 60 x 40 rectangles can't be side by side in a 100 x 100 bin, so
    // each one is widened to 100 x 40; three of them don't fit, which the core
    // bound proves at the root.
    Instance instance = read_instance("feasibility_rectangles_core_bound");
    TreeSearchRelativePositionsParameters parameters;
    parameters.verbosity_level = 0;
    TreeSearchRelativePositionsOutput output = tree_search_relative_positions(
            instance, 0, all_items(instance), parameters);
    EXPECT_EQ(output.status, TreeSearchRelativePositionsStatus::Infeasible);
    EXPECT_EQ(output.number_of_nodes, 1);
    EXPECT_EQ(output.number_of_linear_programs, 0);
}

TEST(IrregularTreeSearchRelativePositions, Packing)
{
    // When the relaxation is feasible, the positions of its solution are
    // returned; here, without exact fits, they are a packing.
    Instance instance = read_instance("feasibility_steps_square_rectangle");
    TreeSearchRelativePositionsParameters parameters;
    parameters.verbosity_level = 0;
    TreeSearchRelativePositionsOutput output = tree_search_relative_positions(
            instance, 0, all_items(instance), parameters);
    ASSERT_EQ(output.status, TreeSearchRelativePositionsStatus::Feasible);
    ASSERT_EQ(output.items.size(), 4);
    Solution solution(instance);
    EXPECT_TRUE(tree_search_relative_positions_solution(instance, 0, output, solution));
}

TEST(IrregularTreeSearchRelativePositions, OpenDimensionUpperBound)
{
    // With an upper bound of 20, there is no better solution, so the bound
    // is 20; with an upper bound of 25, the solution of length 20 is found.
    Instance instance = read_instance("open_dimension_x_steps_interlocked");
    TreeSearchRelativePositionsParameters parameters;
    parameters.verbosity_level = 0;
    parameters.open_dimension_upper_bound = 20;
    TreeSearchRelativePositionsOutput output = tree_search_relative_positions(
            instance, 0, all_items(instance), parameters);
    EXPECT_EQ(output.status, TreeSearchRelativePositionsStatus::Infeasible);
    EXPECT_NEAR(output.open_dimension_bound, 20, 1e-6);
    parameters.open_dimension_upper_bound = 25;
    output = tree_search_relative_positions(instance, 0, all_items(instance), parameters);
    EXPECT_EQ(output.status, TreeSearchRelativePositionsStatus::Feasible);
    EXPECT_NEAR(output.open_dimension_bound, 20, 1e-6);
}

TEST(IrregularTreeSearchRelativePositions, OptimizeExactOpenDimension)
{
    // In 'Exact' mode, the automatic selection runs the tree search on
    // relative positions for the open dimension objectives, which proves
    // that two steps without rotation take a length of 30, whereas the area
    // bound is 20.
    Instance instance = read_instance("open_dimension_x_steps_without_rotation");
    OptimizeParameters parameters;
    parameters.verbosity_level = 0;
    parameters.optimization_mode = OptimizationMode::Exact;
    irregular::Output output = optimize(instance, parameters);
    EXPECT_TRUE(output.is_proven_optimal());
    EXPECT_NEAR(output.solution_pool.best().x_max(), 30, 1e-6);
    EXPECT_NEAR(output.open_dimension_x_bound, 30, 1e-6);
}

TEST(IrregularTreeSearchRelativePositions, OptimizeExactInfeasible)
{
    // In 'Exact' mode, the automatic selection runs the tree search on
    // relative positions for a single bin feasibility problem, which proves
    // that two 55 x 55 squares don't fit in a 100 x 100 bin.
    Instance instance = read_instance("feasibility_squares_too_large");
    OptimizeParameters parameters;
    parameters.verbosity_level = 0;
    parameters.optimization_mode = OptimizationMode::Exact;
    irregular::Output output = optimize(instance, parameters);
    EXPECT_TRUE(output.is_proven_infeasible);
}
