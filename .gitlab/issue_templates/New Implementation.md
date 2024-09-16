<!--
Implementation issues are used break-up a large piece of (new) software work into small, discrete tasks. For instance, when developing a new calibration routine, distinct tasks could be: (1) a QuantumExperiment that enables the measurement of the parameters of interest, (2) An analysis class extracting/fitting the relevant parameters, (3) The integration of (1) and (2) within an automated calibration routine.

Implementation issues are usually refined and modified as the software project evolves.
-->

## Why are we doing this work
<!--
A brief explanation of the why, not the what or how. Assume the reader doesn't know
the background and won't have time to dig-up information from comment threads.
-->

## Relevant links (optional)
<!--
Information that the developer might need to refer to when implementing the issue.

- This QuantumExperiment expands upon previous work / issues:
  - #123 - solving problem A, which is a prerequisite for this QuantumExperiment.
  - #456 - refactoring enabling the new QuantumExperiment to be used.
-->


## Implementation plan
<!--
The implentation plan is where the break-up of a lrager piece of work happens.

=> Try to keep the individual MRs as small as possible.

The plan can also call-out responsibilities for other team members or teams and
can be split into smaller MRs to simplify the code review process.

e.g. building on the example above :

- MR 1: Integrate drivers for new QuantumExperiment
- [ ] ~zurich-instruments ~hardware Step 1
- [ ] ~qcodes Step 2
- MR 2: Implement QuantumExperiment Measurement
- [ ] ~pycqed-core Step 1
- [ ] ~critical Step 2
- [ ] ~enhancement Step 3
- MR 3: Implement QuantumExperiment Analysis
- [ ] ~big Step 1
- [ ] ~gui Step 2
- MR 4: Integrate the new QuantumExperiment in automated calibration routines
- [ ] ~calibration ~usability Step 1

-->

## Verification steps
<!--
Add verification steps to help team members test the implementation. This is
particularly useful during the MR review.

It is possible to verify things using a 'virtual' and/or 'physical' setup.


1. Verification types necessary in this issue: (remove the ones that do not apply)
  - [ ] Virtual Setup
  - [ ] Physical Setup
1. Check-out the corresponding branch in its respective environment
1. Install its requirements
1. Perform QuantumExperiment
1. Add the notebook(s) you tested this with to this issue
1. Enjoy your new QuantumExperiment! ^_^
-->
