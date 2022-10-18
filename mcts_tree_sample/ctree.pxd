# distutils: language=c++
from libcpp.vector cimport vector


cdef extern from "cminimax.cpp":
    pass


cdef extern from "cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        float maximum, minimum, value_delta_max

        void set_delta(float value_delta_max)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst

        void set_delta(float value_delta_max)

cdef extern from "cnode.cpp":
    pass


cdef extern from "cnode.h" namespace "tree":
    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, int action_num, vector[CNode]* ptr_node_pool) except +
        int visit_count, to_play, action_num, data_idx_0, data_idx_1, best_action
        float reward, prior, value_sum
        vector[int] children_index;
        vector[CNode]* ptr_node_pool;

        void expand(float reward_sums, int data_idx_0, int data_idx_1)
        void expand_q_init(float reward_sums, int data_idx_0, int data_idx_1, vector[float] q_inits)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)
        float get_mean_q(int isRoot, float parent_q, float discount)

        int expanded()
        float value()
        vector[int] get_trajectory()
        vector[int] get_children_distribution()
        CNode* get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, int action_num, int pool_size) except +
        int root_num, action_num, pool_size
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(float root_exploration_fraction, const vector[vector[float]] &noises,
                     const vector[vector[float]] &q_inits, const vector[float] &reward_sums)

        void prepare_no_noise(const vector[float] &reward_sums,
                              const vector[vector[float]] &q_inits)

        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[float] get_values()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] data_idx_0, data_idx_1, last_actions, search_lens
        vector[CNode*] nodes
        # vector[vector[CNode*]] search_paths

    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, float value, float discount)
    void cmulti_back_propagate(int data_index_1, float discount, vector[float] rewards, vector[float] values,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
    # int cselect_child(CNode &root, CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init)
    # float cucb_score(CNode &parent, CNode &child, CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init)
    void cmulti_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)


# cdef extern from "cresults.cpp":
#     pass
#
#
# cdef extern from "cresults.h" namespace "search":
#     cdef cppclass CSearchResults:
#         CSearchResults() except +
#         vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions
#         vector[CNode] nodes
#         vector[vector[CNode]] search_paths
#
#     void cmulti_traverse(vector[CNode] roots, vector[CMinMaxStats] min_max_stats_lst, int num, vector[int] histories_len, vector[vector[int]] action_histories, int pb_c_base, float pb_c_init, CSearchResults &results)
