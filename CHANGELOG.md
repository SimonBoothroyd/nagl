# Changelog

## [0.0.3](https://github.com/SimonBoothroyd/nagl/tree/0.0.3) (2021-12-22)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.2...0.0.3)

**Merged pull requests:**

- Handle uncaught exceptions when filtering [\#32](https://github.com/SimonBoothroyd/nagl/pull/32) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.2](https://github.com/SimonBoothroyd/nagl/tree/0.0.2) (2021-12-21)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.1...0.0.2)

**Implemented enhancements:**

- Add initial protomer enumeration support [\#30](https://github.com/SimonBoothroyd/nagl/pull/30) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Expose option to include all resonance transfer pathways [\#29](https://github.com/SimonBoothroyd/nagl/pull/29) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add `normalize\_molecule` utility [\#28](https://github.com/SimonBoothroyd/nagl/pull/28) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Replace enumerate resonance arg with callable [\#27](https://github.com/SimonBoothroyd/nagl/pull/27) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Significantly optimize resonance enumeration [\#26](https://github.com/SimonBoothroyd/nagl/pull/26) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1](https://github.com/SimonBoothroyd/nagl/tree/0.0.1) (2021-12-03)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.1-rc.2...0.0.1)

**Implemented enhancements:**

- Support loading multiple datasets in data module [\#25](https://github.com/SimonBoothroyd/nagl/pull/25) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-rc.2](https://github.com/SimonBoothroyd/nagl/tree/0.0.1-rc.2) (2021-12-01)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.1-rc.1...0.0.1-rc.2)

**Implemented enhancements:**

- Don't one-hot encode booleans [\#23](https://github.com/SimonBoothroyd/nagl/pull/23) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add a general lightning data module [\#22](https://github.com/SimonBoothroyd/nagl/pull/22) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-rc.1](https://github.com/SimonBoothroyd/nagl/tree/0.0.1-rc.1) (2021-11-29)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.1-alpha.4...0.0.1-rc.1)

**Implemented enhancements:**

- Split labelling into separate module [\#21](https://github.com/SimonBoothroyd/nagl/pull/21) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add `DGLMolecule` to device [\#20](https://github.com/SimonBoothroyd/nagl/pull/20) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Remove superfluous configs and clean-up API [\#19](https://github.com/SimonBoothroyd/nagl/pull/19) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Improve performance of retrieving from SQL DB [\#18](https://github.com/SimonBoothroyd/nagl/pull/18) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Order of magnitudes speed up of conformer comparison [\#17](https://github.com/SimonBoothroyd/nagl/pull/17) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Improve label CLI performance [\#16](https://github.com/SimonBoothroyd/nagl/pull/16) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Simplify the molecule DB store [\#15](https://github.com/SimonBoothroyd/nagl/pull/15) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add initial support for resonance structures [\#14](https://github.com/SimonBoothroyd/nagl/pull/14) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Adds data set from molecule store function [\#12](https://github.com/SimonBoothroyd/nagl/pull/12) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Make the label CLI more resilient and informative [\#11](https://github.com/SimonBoothroyd/nagl/pull/11) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Add Basic provenance to Labelled Molecule Store [\#7](https://github.com/SimonBoothroyd/nagl/pull/7) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- Replace OE calls with OFF toolkit calls [\#6](https://github.com/SimonBoothroyd/nagl/pull/6) ([SimonBoothroyd](https://github.com/SimonBoothroyd))
- SQL Based Storage of Annotated Molecules [\#5](https://github.com/SimonBoothroyd/nagl/pull/5) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

**Fixed bugs:**

- Temporary fix guessing stereochemistry of pyramidal N [\#10](https://github.com/SimonBoothroyd/nagl/pull/10) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.4](https://github.com/SimonBoothroyd/nagl/tree/0.0.1-alpha.4) (2021-01-03)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.1-alpha.3...0.0.1-alpha.4)

**Fixed bugs:**

- Add Explicit Hydrogens When Computing AM1 Labels [\#4](https://github.com/SimonBoothroyd/nagl/pull/4) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.3](https://github.com/SimonBoothroyd/nagl/tree/0.0.1-alpha.3) (2020-12-31)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.1-alpha.2...0.0.1-alpha.3)

**Fixed bugs:**

- Store Conformers on Molecule After Labelling. [\#3](https://github.com/SimonBoothroyd/nagl/pull/3) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.2](https://github.com/SimonBoothroyd/nagl/tree/0.0.1-alpha.2) (2020-12-30)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/0.0.1-alpha.1...0.0.1-alpha.2)

**Fixed bugs:**

- Only Retain Unique Enumerated Tautomers [\#2](https://github.com/SimonBoothroyd/nagl/pull/2) ([SimonBoothroyd](https://github.com/SimonBoothroyd))

## [0.0.1-alpha.1](https://github.com/SimonBoothroyd/nagl/tree/0.0.1-alpha.1) (2020-12-29)

[Full Changelog](https://github.com/SimonBoothroyd/nagl/compare/b6126177167ae4cbea6705dc2398cc3d7fb84034...0.0.1-alpha.1)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
