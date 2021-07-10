# Projects

This folder contains `projects` - essentially collections of code and
configurations necessary for computing results in a paper. The primary emphasis
is on reproducibility - please try to document the complete environment
necessary for running the code, including appropriate package versions and
configs.

Each folder contains a separate project, which should live in its own conda or
pip environment and might have additional dependencies.

The code is expected to be hackable/experimental. Reviewers should note that we
do not expect to see all type annotations or complete documentation in
`projects`. We do run `flake8`, `isort`, and `black` on the folder to promote
a little bit of sanity and help our fellow humans easily parse what other
people write.

You can fork this repository and add your examples as sub-folders here.
