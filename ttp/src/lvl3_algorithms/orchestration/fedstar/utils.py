#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs


# Custom


##################
# Configurations #
##################


#############
# Functions #
#############

def generate_single_bin_seq(length_of_seq: int):
    bin_seq = np.zeros(length_of_seq)
    idx_to_set_to_one = np.random.randint(len(length_of_seq))
    bin_seq[idx_to_set_to_one] = 1
    return bin_seq

def generate_participant_bin_seq(num_of_seqs: int, len_of_seq: int):
    participants_binary_seq = np.concatenate(
                            [generate_single_bin_seq(len_of_seq) for _ in range(num_of_seqs)])
    return participants_binary_seq

def calc_si(idx, n, number_of_participants):
    if idx <= (n % number_of_participants):
        return 1 + math.floor(n/number_of_participants)
    else:
        return math.floor(n/number_of_participants)

############################################
# Orchestration Class - <INSERT NAME HERE> #
############################################

# Replicate for each class defined



###########
# Scripts #
###########