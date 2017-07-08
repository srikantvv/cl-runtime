#ifndef HSA_SIGNAL_H_INCLUDED
#define HSA_SIGNAL_H_INCLUDED


typedef int64_t hsa_signal_value_t;

typedef struct hsa_signal_s {
    uint64_t handle;
} hsa_signal_t;

cl_int hsa_signal_create(
    hsa_signal_value_t initial_value,
    uint32_t num_consumers,
    int consumers, //Consumers is yet to be  defined
    hsa_signal_t *signal) {
}

cl_int hsa_signal_destroy(
    hsa_signal_t signal
);

typedef enum {
    HSA_WAIT_STATE_BLOCKED = 0,
    HSA_WAIT_STATE_ACTIVE = 1
} hsa_wait_state_t;

typedef enum {
    HSA_SIGNAL_CONDITION_EQ = 0,
    HSA_SIGNAL_CONDITION_NE = 1,
    HSA_SIGNAL_CONDITION_LT = 2,
    HSA_SIGNAL_CONDITION_GTE = 3
} hsa_signal_condition_t;

hsa_signal_value_t hsa_signal_wait_scacquire(
    hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint
);

hsa_signal_value_t hsa_signal_wait_relaxed(
    hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint
);

#endif
