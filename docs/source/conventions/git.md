# Git convensions

## Branches

The branching model is based on the branching model described in the post
[A successful Git branching model](http://nvie.com/posts/a-successful-git-branching-model/),
but differs in that the development branch is replaced by several "project"
branches. 

### Branch naming

TODO: points to discuss: 
* I would recommend to avoid any uppercase in branch naming (i.e. 
`Proj` -> `proj`), but rather only use snake case.

Here is the branch naming scheme used, which will be explained in the following
subsections:
*  `qudev_master`: Main branch of the repository.
*  `Proj/<branch_name>`: Project branch.
*  `devel/<optional_proj>/<branch_name>`: Development branch.
*  `feature/<branch_name>`: Feature branch.
*  `enh/<branch_name>`: Enhancement branch.
*  `doc/<branch_name>`: Documentation branch.
*  `cleanup/<branch_name>`: Clean-up branch.
*  `tmp/<branch_name>`: Temporary branch.

#### Main branch

* The main branch of this repository is `qudev_master`.
* It is the reference branch which should only contain valid code.
* No one is allowed to directly push on it. It is only possible to modify it
through merge requests.
* When working on long-lasting branches, be sure to update them regularly with
the latest changes from this branch. This can be done as follows:
```bash
git checkout master
git pull origin master
git checkout <your_branch>
git merge qudev_master
# git push origin <your_branch>
```

#### Project branches

* There is usually one project branch per experiment setup, however closely
related setups might share the same project branch.
* These branches should be used to merge required feature branches into the
project branch when they are needed before they are merged in `qudev_master`.
* Individual commits on project branch are allowed only if they are definitely
project-specific and will never be needed anywhere else.

For practical reasons, the following scenarios are tolerated:
* Commit to the project branch when developing directly on the experiment setup.
TODO: Maybe I forgot what Christoph exactly  meant with this sentence. I would
think that it is actually bad to commit on the project branch when developping
directly on the setup. I would rather recommend to commit on a tmp branch.
* Developing something that is based on functionality that is currently only
available on the project branch. When ready for a merge request, cherry-pick
onto a new feature branch.

#### Development branches

* They are used for work in progress, not necessarily related to a particular
feature.
* Commits on them can be cherry-picked to create a feature branch.

#### Feature branches

* They are used to implement new features that need to be integrated in
`qudev_master`.
* Keep the feature branch focused on a single well-scoped feature.

#### Enhancement branches

* They are used for minor improvements on existing features/functions.
* The changes introduced should not be too large, otherwise a feature branch
would be more suited.
* Example: adding an additional parameter to an existing function.

#### Documentation branches

* They are used when the changes only relate to documentation.
* Use these when cleaning/improving existing docstrings.
* Use these when writing new documentation pages.

#### Clean-up branches

* Used to remove/delete obsolete modules/classes/functions.

#### Temporary branches

* Use this to store temporary changes, that may or may not be reused later.
* Can be useful for work-in-progress, or to commit unfinished code.
* Don't hesitate to prefix the temp branches with your name (e.g.
`tmp/thomas/something`) so that people know who cares about this branch.
* Should be deleted regularly.

### Branch archiving

TODO: Check with Christoph what the strategy will be. Still planning to have
a mirror repo with archived branches ? Tags ?

## Commits

Here are the basic rules for commit:
* Commits must be atomic. They might contain code changes in multiple files, but
they should all be related to the same aspect. For instance, if you correct a
bug somewhere on the codebase while implementing a new feature, commit those
changes separetely, not as part of a bunch of unrelated changes.
* Always commit your work to avoid loosing it! It is better (**but not
encouraged!**) to commit unfinished work or not-atomic commits than not
committing at all.
* Push your changes frequently to the GitLab.
* Never ever force-push to GitLab! This will rewrite the commit history for your
coworkers and result in a lot of manual cleanup.

### Commit messages

* Always write proper commit messages describing precisely the changes
introduced.
* The first line should be a summary, and further lines should contain a more
detailed description if need. The commit history should be readable as much as
possible without having to inspect the diffs.
* Note that you can mention Gitlab merge requests (e.g. `!3`) and issues (e.g.
`#7`) within a commit message. You are encouraged to do this, as GitLab will
in turn reference the commits in question inside the MR/issue..
* You can amend or squash local commits if you haven't pushed yet to the origin.
This is useful to correct errounous commit messages, or to group together
semantically related commits.
* If your commit is to later be integrated to a feature branch, prefix its
commit message with a tag `[f/<feature_branch_name>]`. This will facilitate the
effort to find the relevant commits to cherry-pick into the feature branch.

## Merge requests

### Concept

Once you are satisfied with the changes you have made on your branch, you can
open a merge request on GitLab, to signal that you would like it to be merged
in `qudev_master` (or another branch if applicable).

The merge requests are an opportunity to receive/give feedback on the changes
introduced, the quality of the code, alternative implementations, etc.
Typically, another person will review your changes, and propose a list of
modifications to be made before the merge request is accepted.

### Guidelines

* Create and review merge requests ASAP, so as to:
  1. Have less branches, resulting in a cleaner repository.
  2. Reduce the likelihood and seriouness of merge conflicts.
* Keep the scope of the merge request as small as possible, to make it easier
to review:
  * 1 bugfix per MR, instead of multiple bugfixes (unless all related).
  * 1 new feature per MR, unless the new features do not make sense individually.
* Fill in the template on GitLab (see
[Merge request template](./merge_Request_template.md)).
* Provide background when necessary to facilitate the review, e.g. justify some
implementation choices, or document the limitations of your implementation.
* Reference related GitLab merge requests (e.g. `!3`) and issues (e.g. `#7`).
* Mention people (e.g. `@thhavy`) in discussion if you need feedback from them.
* Delete the source branch after it is merged, unless there is a reason to keep
it alive (e.g. for archiving purposes).

## Issues tracking

* When encountering problems with PycQED, make sure to open an issue on GitLab,
so that it can be properly tracked and eventually resolved.
* Issues may be reopenned later if the problem reoccurs or if a bug is
re-introduced.
* Fill in the template on GitLab (see [Issue template](./issue_template.md)).
* Add adequate labels to classify issues.
* You can create a merge request to fix and issue directly from the issue
description in GitLab (see {numref}`mr_from_issue`).

(mr_from_issue)=
```{figure} /images/conventions/mr_from_issue.png
:align: center

Create a merge request from an issue in GitLab
```
