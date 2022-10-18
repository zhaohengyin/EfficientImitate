#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <sys/time.h>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count;            // How many time it is visited.
            int action_num;             // How many actions it can take.
            float reward;               // The reward of transferring to this state.
            float value_sum;            // The sum of value after many back propagations.
            float prior;
            int use_q_init;
            float q_init;
            int data_idx_0, data_idx_1;
            int best_action;

            std::vector<int> children_index;   // The identifiers of the children.
            std::vector<CNode>* ptr_node_pool; // This pool is shared by one tree, from which we can retrieve
                                               // any node quickly.

            CNode();
            CNode(float prior, int action_num, std::vector<CNode> *ptr_node_pool);
            ~CNode();
            void expand_q_init(float reward, int data_idx_0, int data_idx_1, const std::vector<float>& q_init);
            void expand(float reward_sum, int data_idx_0, int data_idx_1);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float get_mean_q(int isRoot, float parent_q, float discount);
            void print_out();

            int expanded();

            float value();

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num, action_num, pool_size;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int root_num, int action_num, int pool_size);
            ~CRoots();

            void prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises,
                         const std::vector<std::vector<float>>& q_inits, const std::vector<float> &reward_sums);
            void prepare_no_noise(const std::vector<float> &reward_sums,
                                  const std::vector<std::vector<float>>& q_inits);
            void clear();
            std::vector<std::vector<int>> get_trajectories();
            std::vector<std::vector<int>> get_distributions();
            std::vector<float> get_values();

    };

    class CSearchResults{
        public:
            int num;    // How many roots.
            std::vector<int> data_idx_0, data_idx_1, last_actions, search_lens;
            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*>> search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };


    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount);
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, float value, float discount);
    void cmulti_back_propagate(int data_index_1, float discount, const std::vector<float> &rewards, const std::vector<float> &values, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float total_children_visit_counts, float parent_mean_q, float pb_c_base, float pb_c_init, float discount);
    void cmulti_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);
}

#endif