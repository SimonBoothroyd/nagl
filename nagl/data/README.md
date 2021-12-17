# Manifest

* `normalizations.json`: A file containing reaction SMIRKS that define a set of 'normalisations' to apply to a molecule,
                         such as changing nitro groups from `N(=O)=O` to `N(=O)[O-]`. These were taken directly from the
                         RDKit [`rdkit/Code/GraphMol/MolStandardize/TransformCatalog/normalizations.in`](https://github.com/rdkit/rdkit/blob/67aa9f9062779500b87e774019d45ff19c34aba5/Code/GraphMol/MolStandardize/TransformCatalog/normalizations.in)
                         file at commit `67aa9f9` (see the `LICENSE-3RD-PARTY` file for more details), which contained 
                         the following license header:
    ```
     //
     //  Copyright (C) 2021 Greg Landrum
     //
     //   @@ All Rights Reserved @@
     //  This file is part of the RDKit.
     //  The contents are covered by the terms of the BSD license
     //  which is included in the file license.txt, found at the root
     //  of the RDKit source tree.
     //
    ```