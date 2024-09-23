<!--
Implementation issues are used break-up a large piece of (new) software work into small, discrete tasks. For instance, when developing a new calibration routine, distinct tasks could be: (1) preparatory work such as new features on lower layers of pycqed (e.g., pulse library, hardware drivers, ...), which are needed to enable the new kinds of measurements (2) a QuantumExperiment class that enables the measurement of the parameters of interest, (3) An analysis class extracting/fitting the relevant parameters, (4) The integration of (2) and (3) within an automated calibration routine.

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

- MR 1: preparatory work on lower abstraction layers
- [ ] ~qcodes Step 1: driver for ...
- [ ] ~pycqed-core Step 2: new feature ... in ...
- MR 2: basic functionality
- [ ] ~critical Step 1: Implement basic measurement class (child of QuantumExperiment)
- [ ] ~big Step 2: write analysis class for the basic features
- MR 3: Implement additional features and improve user experience
- [ ] ~enhancement Step 1: add the additional feature to measurement class
- [ ] ~enhancement Step 2: extend analysis class for the additional feature
- [ ] ~gui ~usability Step 3: integration into GUI
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
