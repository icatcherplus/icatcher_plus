import re

def parse_illegal_transitions_file(path, skip_header=True):
    illegal_transitions = []
    corrected_transitions = []
    if path is not None:
        with open(path, newline='') as f:
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
                ilegal_transitions = [int(x) for x in re.findall(r'-?\d', ilegal)]
                illegal_transitions.append(ilegal_transitions)

                # Treat the entire corrected transition as a single entity
                corrected_transition = [int(x) for x in re.findall(r'-?\d', corrected)]
                corrected_transitions.append(corrected_transition)
            except ValueError:
                raise ValueError("Illegal transitions file needs to have only integers.")
    return illegal_transitions, corrected_transitions
