# Nilearn on boarding

Here is some information and links that may be helpful if you don't already know
about it.

- Neurostars: https://neurostars.org/tag/nilearn

  - where most of the Nilearn usage related questions get posted
  - to make monitoring easier, you can set up "alerts" to get updated about new
    posts

- Discord: https://discord.gg/bMBhb7w

  - sometimes used for direct communications between devs or with some users

- Twitter: https://twitter.com/nilearn

  - mostly for announcements

- Mastodon: https://fosstodon.org/@nilearn

  - mostly for announcements

- google drive:
  https://drive.google.com/drive/folders/1sTjTyM7ezeWwmbtDvOKCW2H7qFOUIOg9?usp=drive_link

  - This is a public google drive where we keep some slide deck or a few things
    where needed to work colaboratively on for some events
  - Not supposed to be central to our workflows (amongst other things because
    google tools cannot be accessed by everyone - China, Iran...)

- nilearn gmail: nilearn.events@gmail.com

- Weekly drop-in hours:
  https://arewemeetingyet.com/UTC/2023-01-18/16:00/w/Nilearn%20Drop-in%20Hours#eyJ1cmwiOiJodHRwczovL21lZXQuaml0LnNpL25pbGVhcm4tZHJvcC1pbi1ob3VycyJ9
  - Not expected to attend always but a few of us are usually there. We don't
    have a lot of users/contributors coming to ask questions, but we can use it
    as a weekly meeting to discuss ongoing issues/PRs.
  - Shortened url for tweeting: https://tinyurl.com/nilearn-drop-in-hour

## Adding new developpers to the "doc"

- add yourself instead:
  https://github.com/nilearn/nilearn/tree/main/maint_tools/citation_cff_maint.py#L22
- make sure to add yourself and your detail to the CITATION.cff
  https://github.com/nilearn/nilearn/tree/main/CITATION.cff#L42

## Access the Nilearn organization

Make sure that new developpers are granted access to the following:

- [ ] give acces to organization (`core`, `triage`)
  - [ ] make sure that new maintainers have access to
    - [ ] ni
- access to:
  - [ ] social media: give access to
    - [ ] twitter / X
    - [ ] fostodon
  - [ ] discord:
    - give the role `core-developper`
  - [ ] nilearn gmail: ask Elisabeth to grant accces "remote access"
  - [ ] add new developper to pypi:
    - [ ] make sure they have got a pypi account with 2FA and the proper token
          for releasing to pypi
  - [ ] make sure they are on the python neuroimaging mailing list:
        https://mail.python.org/mailman/listinfo/neuroimaging

## Tour of different repos in the organization

### `nilearn`: main code base

### `sandbox`: for draft ideas

### `admin`: where we put the dev meeting notes (and a few other things)

the meeting notes are uploaded from
[this hackmd](https://hackmd.io/hl-vZfHwTkayO5AiT9vvkQ) to this repo after each
meeting.

### `nilearn.github.io`:

The repository that hosts / serves the HTML version of the documentation.

## Tour of the nilearn code base

```
├── .binder
├── .circleci           # configuration for continuous integration on circle-ci (mostly hosting of the documentation)
│   └── config.yml
├── .github
│   ├── ISSUE_TEMPLATE
│   └── workflows       # configuration for continuous integration on github actions
├── build_tools         # scripts used in during continuous integrations (related to building and hosting of the documentation)
├── doc                 # sphinx based documentation: configuration, makefile, scripts and actual documentation
├── examples            # examples that will be rendered in the sphinx gallery
├── maint_tools         # set of scripts to help with maintenance (should be runnable)
└── nilearn             # actualy nilearn codebase
```

## code

All the codebase is in the `nilearn` folder.

Tests are distributed
[as part of the application](https://docs.pytest.org/en/8.0.x/explanation/goodpractices.html#tests-as-part-of-application-code).

## packaging

All the packaging details are in `pyproject.toml`.

We rely on [hatchling](https://hatch.pypa.io/latest/) for packaging and
[hatch-vcs](https://pypi.org/project/hatch-vcs/) for keeping track of the
package version. Note that this relies on the presence of the git tags in your
repository for the generated version string to make sense (See the
"recommendation" admonition in
[this section of the contributing doc](https://nilearn.github.io/stable/development.html#installing)).

For more info see our
[maintenance doc](https://nilearn.github.io/stable/maintenance.html#how-to-make-a-release).

## formatting / linting

We use a series of tools to keep the "code" properly formated.

- `black`: format python code
- `isort`: sort python imports
- `flake8`: check python best practice
- `codespell`: check for spelling mistakes
- ...

All those tools (except flake8) have their configuration in `pyproject.toml`.
For flake8 it is in the `.flake8` file.

### Continuous integration checks

All those tools have dedicated github action workflows to make sure they are
implemented.

### `pre-commit`

You can also decide to use `pre-commit` locally to ensure that all those rules
are implemented with every commit.

Check
[our documentation](https://nilearn.github.io/stable/development.html#pre-commit)
to see how to install it.

`pre-commit` is configured in `.pre-commit-config.yaml`. This also contains a
few extra rules (hooks) that are not enforced in continuous integration (for
example proper formatting of toml and yaml files, avoid trailing whitespace in
ANY file...).

## `tox`

Tox is used to simplify setting up different python environments for testing.

For more info see
[our maintenance doc](https://nilearn.github.io/stable/maintenance.html#using-tox).

## Continuous integration

### Github actions

### Circle-CI

## Documentation

## Tests

## handling new issues

- reply as soon as possible to let users know they have been heard
- consider if the issue should not be better opened as a post on neurostars
- add relevant labels
- BUG reports:
  - is there enough information to reproduce?
  - can it be reproduced with either nilearn dataset or 'mocking' data?
  - was a solution provided by the user?
  - is this a regression?
  - fix using test driven development
    - create a failing test for the bug
    - fix the bug
    - eventually refactor

## Explain github projects

- doc / videos to pytest
