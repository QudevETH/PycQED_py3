import re


def flatten_list(lis):
    """
    Help function for using:create_experiment_list_pyGSTi

    """
    from collections import Iterable
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten_list(item):
                yield x
        else:
            yield item


def create_experiment_list_pyGSTi_qudev(filename, qb_names=[''],
                                        pygstiGateList=None):
    """
    Extracting list of experiments from .txt file
    !!!! For 2 qbs, this function assumes qb_names[0] is the control qb and
    qb_names[1] is the target. !!!

    Parameters:

    filename: string
        Name of the .txt file. File must be formatted in the way as done by
        pyGSTi.
        One gatesequence per line, formatted as e.g.:Gx(Gy)^2Gx.

    Returns:

    Nested list containing all gate sequences for experiment. Every gate
    sequence is also a nested list, []

    """
    if pygstiGateList is None:
        experiments = open(filename)
        sequences = experiments.read().split("\n")
    else:
        sequences = pygstiGateList

    if len(qb_names) == 1:
        RO_str = "RO " + qb_names[0]
    else:
        RO_str = "RO mux"

    experimentlist = []
    for i in range(len(sequences)):

        clean_seq = sequences[i].strip()
        gateseq = []

        if "{}" in clean_seq or clean_seq == '':
            gateseq.insert(0, RO_str)
            experimentlist.append(gateseq)

        if "(" in clean_seq:
            prepfiducial = []
            germs = []
            measfiducial = []

            if "^" in clean_seq:
                power = int(re.findall(r"\d+", clean_seq)[0])
                result = re.split(r"[(]|\)\^\d", clean_seq)
            else:
                power = 1
                result = re.split("[()]", clean_seq)

            append_pycqed_gate(result[0], prepfiducial, qb_names)
            append_pycqed_gate(result[1], germs, qb_names)
            append_pycqed_gate(result[2], measfiducial, qb_names)

            if len(prepfiducial) != 0:
                gateseq.append(prepfiducial)
            if len(germs) != 0:
                gateseq.append(germs*power)
            if len(measfiducial) != 0:
                gateseq.append(measfiducial)

            gateseq.append([RO_str])
            gateseq = list(flatten_list(gateseq))
            experimentlist.append(gateseq)
        elif ("Gi" in clean_seq) or ("Gx" in clean_seq) or ("Gy" in clean_seq) \
                or ("Gz" in clean_seq) or ("Gcphase" in clean_seq):
            loopseq = []
            append_pycqed_gate(clean_seq, loopseq, qb_names)

            gateseq.append(loopseq)
            gateseq.append([RO_str])
            gateseq = list(flatten_list(gateseq))
            experimentlist.append(gateseq)

    if pygstiGateList is None:
        if len(experimentlist) < (len(sequences)-2):
            print(len(experimentlist))
            print(len(sequences))
            print("Length list of experiments too short, "
                  "probably something wrong")
        experiments.close()
    else:
        if len(experimentlist) != len(sequences):
            print(len(experimentlist))
            print(len(sequences))
            print("Length list of experiments too short, "
                  "probably something wrong")

    return experimentlist


def append_pycqed_gate(pygsti_gate_str, gate_list, qb_names=['']):

    if len(qb_names) == 1:
        qb_name = qb_names[0]
        regsplit = pygsti_gate_str.split('G')[1::]
        for i in range(len(regsplit)):
            if regsplit[i] == "i":
                gate_list.append("I " + qb_name)
            elif regsplit[i] == "x":
                gate_list.append("X90 " + qb_name)
            elif regsplit[i] == "y":
                gate_list.append("Y90 " + qb_name)
            elif regsplit[i] == "z":
                gate_list.append("Z90 " + qb_name)
            else:
                raise ValueError('Unknown pygsti gate type "{}"'.format(
                    pygsti_gate_str))

    elif len(qb_names) == 2:
        regsplit = pygsti_gate_str.split('G')[1::]
        for i in range(len(regsplit)):
            if regsplit[i] == "ii":
                gate_list.append("I " + qb_names[0])
                gate_list.append("Is " + qb_names[1])
            elif regsplit[i] == "ix":
                gate_list.append("I " + qb_names[0])
                gate_list.append("X90s " + qb_names[1])
            elif regsplit[i] == "iy":
                gate_list.append("I " + qb_names[0])
                gate_list.append("Y90s " + qb_names[1])
            elif regsplit[i] == "iz":
                gate_list.append("I " + qb_names[0])
                gate_list.append("Z90s " + qb_names[1])
            elif regsplit[i] == "xi":
                gate_list.append("X90 " + qb_names[0])
                gate_list.append("Is " + qb_names[1])
            elif regsplit[i] == "yi":
                gate_list.append("Y90 " + qb_names[0])
                gate_list.append("Is " + qb_names[1])
            elif regsplit[i] == "zi":
                gate_list.append("Z90 " + qb_names[0])
                gate_list.append("Is " + qb_names[1])
            elif regsplit[i] == "xx":
                gate_list.append("X90 " + qb_names[0])
                gate_list.append("X90s " + qb_names[1])
            elif regsplit[i] == "yy":
                gate_list.append("Y90 " + qb_names[0])
                gate_list.append("Y90s " + qb_names[1])
            elif regsplit[i] == "zz":
                gate_list.append("Z90 " + qb_names[0])
                gate_list.append("Z90s " + qb_names[1])
            elif regsplit[i] == "cphase":
                gate_list.append("CZ {} {}".format(qb_names[1], qb_names[0]))
            else:
                raise ValueError('Unknown pygsti gate type "{}"'.format(
                    pygsti_gate_str))
    else:
        raise ValueError('This functions works only up to 2 qubits.')