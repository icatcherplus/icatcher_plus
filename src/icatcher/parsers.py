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
            row = row.split(",")
            row = [x.strip() for x in row]
            if len(row) != 2:
                raise ValueError("Illegal transitions file needs to have exactly two columns.")
            ilegal, corrected = row
            try:
                illegal_transitions.append([int(x) for x in ilegal])
                corrected_transitions.append([int(x) for x in corrected])
            except ValueError:
                raise ValueError("Illegal transitions file needs to have only integers.")
    return illegal_transitions, corrected_transitions