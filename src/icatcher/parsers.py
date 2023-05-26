import re
from . import reverse_classes

def parse_illegal_transitions_file(path, skip_header=True):
    """
    given a path to a csv file, parse the illegal transitions
    :param path: path to csv file
    :param skip_header: whether to skip the header row
    :return: illegal_transitions, corrected_transitions
    """
    illegal_transitions = []
    corrected_transitions = []
    if path is not None:
        with open(path) as f:
            rows = f.readlines()
        # skip header
        if skip_header:
            rows = rows[1:]
        for row in rows:
            row = row.strip().split(",")
            row = [x.replace('"', '') for x in row]
            if len(row) != 2:
                raise ValueError("Illegal transitions file needs to have exactly two columns.")
            ilegal, corrected = row
            try:
                # Separate by each digit or negative number, if "-" appears, it should be with the next digit
                bad_transition = [int(x) for x in re.findall(r'-?\d', ilegal)]
            except ValueError:
                raise ValueError("Illegal transitions file needs to have only integers.")
            # Check if the illegal transitions are valid
            for x in bad_transition:
                if x not in reverse_classes.keys():
                    raise ValueError("Illegal transitions file needs to only have valid classes.")
            illegal_transitions.append(bad_transition)
            try:
                good_transition = [int(x) for x in re.findall(r'-?\d', corrected)]
            except ValueError:
                raise ValueError("Illegal transitions file needs to have only integers.")
            for x in good_transition:
                if x not in reverse_classes.keys():
                    raise ValueError("Illegal transitions file needs to only have valid classes.")
            if len(good_transition) != len(bad_transition):
                raise ValueError("One or more illegal transitions have a different length than its corrected transition.")
            corrected_transitions.append(good_transition)
    return illegal_transitions, corrected_transitions
