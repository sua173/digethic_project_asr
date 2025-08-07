# Calculate the number of recognition errors
# using Levenshtein distance

import numpy as np
import copy


def calculate_error(hypothesis, reference):
    """Calculate Levenshtein distance and output
        substitution, deletion, and insertion errors
    hypothesis:       Recognition result (list of tokens)
    reference:        Ground truth (list of tokens)
    total_error:      Total number of errors
    substitute_error: Number of substitution errors
    delete_error:     Number of deletion errors
    insert_error:     Number of insertion errors
    len_ref:          Number of tokens in reference
    """
    # Get lengths of hypothesis and reference sequences
    len_hyp = len(hypothesis)
    len_ref = len(reference)

    # Create cumulative cost matrix
    # Each element contains cumulative values of
    # total cost, substitution cost, deletion cost,
    # and insertion cost in dictionary format.
    cost_matrix = [
        [
            {"total": 0, "substitute": 0, "delete": 0, "insert": 0}
            for j in range(len_ref + 1)
        ]
        for i in range(len_hyp + 1)
    ]

    # Initialize first column and first row
    for i in range(1, len_hyp + 1):
        # Vertical transition represents deletion operation
        cost_matrix[i][0]["delete"] = i
        cost_matrix[i][0]["total"] = i
    for j in range(1, len_ref + 1):
        # Horizontal transition represents insertion operation
        cost_matrix[0][j]["insert"] = j
        cost_matrix[0][j]["total"] = j

    # Calculate cumulative costs for remaining cells
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            #
            # Calculate cost for each operation
            #
            # Diagonal transition: if characters don't match,
            # substitution increases cumulative cost by 1
            substitute_cost = cost_matrix[i - 1][j - 1]["total"] + (
                0 if hypothesis[i - 1] == reference[j - 1] else 1
            )
            # Vertical transition: deletion increases cumulative cost by 1
            delete_cost = cost_matrix[i - 1][j]["total"] + 1
            # Horizontal transition: insertion increases cumulative cost by 1
            insert_cost = cost_matrix[i][j - 1]["total"] + 1

            # Determine which operation (substitution, deletion, insertion)
            # results in minimum cumulative cost
            cost = [substitute_cost, delete_cost, insert_cost]
            min_index = np.argmin(cost)

            if min_index == 0:
                # When substitution gives minimum cumulative cost

                # Copy cumulative cost information from source cell
                cost_matrix[i][j] = copy.copy(cost_matrix[i - 1][j - 1])
                # If characters don't match,
                # increase cumulative substitution cost by 1
                cost_matrix[i][j]["substitute"] += (
                    0 if hypothesis[i - 1] == reference[j - 1] else 1
                )
            elif min_index == 1:
                # When deletion gives minimum cumulative cost

                # Copy cumulative cost information from source cell
                cost_matrix[i][j] = copy.copy(cost_matrix[i - 1][j])
                # Increase cumulative deletion cost by 1
                cost_matrix[i][j]["delete"] += 1
            else:
                # When insertion gives minimum cumulative cost

                # Copy cumulative cost information from source cell
                cost_matrix[i][j] = copy.copy(cost_matrix[i][j - 1])
                # Increase cumulative insertion cost by 1
                cost_matrix[i][j]["insert"] += 1

            # Update cumulative total cost (substitution + deletion + insertion)
            cost_matrix[i][j]["total"] = cost[min_index]

    # The bottom-right element of the cost matrix contains final costs.
    total_error = cost_matrix[len_hyp][len_ref]["total"]
    substitute_error = cost_matrix[len_hyp][len_ref]["substitute"]
    # Deletion errors = insertion cost
    delete_error = cost_matrix[len_hyp][len_ref]["insert"]
    # Insertion errors = deletion cost
    insert_error = cost_matrix[len_hyp][len_ref]["delete"]

    # Return error counts and reference length
    # (used as denominator when calculating error rates)
    return (total_error, substitute_error, delete_error, insert_error, len_ref)


if __name__ == "__main__":
    import sys
    
    # Check if two command line arguments are provided
    if len(sys.argv) == 3:
        # Get strings from command line arguments
        hyp = sys.argv[1]
        ref = sys.argv[2]
    else:
        # Use default example if no arguments provided
        print("Usage: python levenshtein.py \"hypothesis\" \"reference\"")
        print("Using default example:")
        ref = "SAID COULSON HE'S SENT FOR"
        hyp = "SAY COOL SON HE'S SENDED FOUR"

    # Split each string into individual characters as list
    hyp_list = list(hyp)
    ref_list = list(ref)

    # Calculate error counts
    total, substitute, delete, insert, ref_length = calculate_error(hyp_list, ref_list)

    # Output error counts and error rates (100 * error_count / reference_length)
    print("REF: %s" % ref)
    print("HYP: %s" % hyp)
    print(
        "#TOKEN(REF): %d, #ERROR: %d, #SUB: %d, #DEL: %d, #INS: %d"
        % (ref_length, total, substitute, delete, insert)
    )
    print(
        "UER: %.2f, SUBR: %.2f, DELR: %.2f, INSR: %.2f"
        % (
            100.0 * total / ref_length,
            100.0 * substitute / ref_length,
            100.0 * delete / ref_length,
            100.0 * insert / ref_length,
        )
    )
    
    # Also calculate at word level for reference
    hyp_words = hyp.split()
    ref_words = ref.split()
    total_w, substitute_w, delete_w, insert_w, ref_length_w = calculate_error(hyp_words, ref_words)
    print("\nWord-level metrics:")
    print(
        "#WORDS(REF): %d, #ERROR: %d, #SUB: %d, #DEL: %d, #INS: %d"
        % (ref_length_w, total_w, substitute_w, delete_w, insert_w)
    )
    print(
        "WER: %.2f, SUBR: %.2f, DELR: %.2f, INSR: %.2f"
        % (
            100.0 * total_w / ref_length_w if ref_length_w > 0 else 0,
            100.0 * substitute_w / ref_length_w if ref_length_w > 0 else 0,
            100.0 * delete_w / ref_length_w if ref_length_w > 0 else 0,
            100.0 * insert_w / ref_length_w if ref_length_w > 0 else 0,
        )
    )
