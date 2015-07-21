# nilearn_sandbox
Playground for nilearn compatible features.

This python package aims at sharing beta features that are meant to be merged into nilearn but can't be for the moment because they have strong external dependencies or because they are not directly in the scope of nilearn. It also is the perfect location to advertise a beta feature.

## Rules

This repository is meant to be a playground but we still need to impose rules to avoid anarchy.
* The structure of the nilearn package must be mirrorred as much as possible.
* Integration of code is made through pull request system.
* Import of external python packages should be done locally in your functions so that people who wants to use a particular feature are not blocked by an unwanted dependency.
* This repository requires less quality in the code but please try to document and test it.
* 

### *Happy hacking!*
